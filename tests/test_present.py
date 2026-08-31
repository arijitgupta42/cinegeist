"""End-to-end tests for the assembled PRESENT phase.

A mini in-memory catalog with hand-built genome rows drives the whole pipeline — filter, score,
diversify, (optionally) rerank, explain. The offline path must produce real picks and grounded
templates with no LLM; the online path runs both model calls against an ``httpx.MockTransport``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.config import load_settings
from cinegeist.feedback import record_feedback
from cinegeist.llm.client import OpenRouterClient
from cinegeist.profile.model import TagAffinity, TasteProfile
from cinegeist.recommend import present

MODELS = ("stub/model",)

# 8-dim genome. Films 1–3 sit near the profile (tags 0,1); film 4 is far but still loads on 0,1
# (the wildcard); films 5,6 are low. Rows are indexed by genome_row.
_MATRIX = np.array(
    [
        [1.0, 1.0, 0, 0, 0, 0, 0, 0],  # 1
        [0.95, 0.9, 0, 0, 0, 0, 0, 0],  # 2
        [0.9, 0.95, 0, 0, 0, 0, 0, 0],  # 3
        [0.6, 0.6, 1, 1, 1, 1, 1, 1],  # 4  wildcard: far, shares tags 0 and 1
        [0.0, 0.0, 1, 1, 0, 0, 0, 0],  # 5  orthogonal
        [0.2, 0.0, 0.5, 0, 0, 0, 0, 0],  # 6  weak
    ],
    dtype=np.float32,
)


@pytest.fixture
def catalog() -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(i + 1, i, f"t{i}") for i in range(8)],
    )
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, 'measured')",
        [(mid, f"F{mid} ({2000 + mid})", f"F{mid}", 2000 + mid, mid - 1) for mid in range(1, 7)],
    )
    conn.commit()
    return conn


def _profile() -> TasteProfile:
    axes = (
        TagAffinity(position=0, name="t0", weight=0.8, source="'F1'", evidence="loved the quiet"),
        TagAffinity(position=1, name="t1", weight=0.6, source="'F1'", evidence=None),
    )
    return TasteProfile(
        user_id="default",
        genome_vector=np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        total_weight=3.0,
        event_count=2,
        session_count=1,
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
        axes=axes,
    )


def _make_client(handler) -> OpenRouterClient:
    settings = load_settings(
        env={"OPENROUTER_API_KEY": "sk-or-stub"}, config_file=Path("nope.toml")
    )
    return OpenRouterClient(settings, transport=httpx.MockTransport(handler), sleep=lambda _s: None)


def _queue_handler(*texts: str):
    queue = list(texts)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": queue.pop(0)}}], "model": "stub"}
        )

    return handler, queue


# -- offline -------------------------------------------------------------------------


def test_offline_present_produces_picks_and_a_wildcard(catalog) -> None:
    result = present.present(catalog, _MATRIX, _profile(), offline=True, n_confident=3)
    assert result.pool_size == 6
    assert {p.film.movie_id for p in result.picks} == {1, 2, 3}  # the three near the profile
    assert result.wildcard is not None
    assert result.wildcard.film.movie_id == 4  # far, but shares tags 0 and 1
    assert result.degraded_rerank and result.degraded_explain  # offline uses no model


def test_offline_explanations_are_grounded_templates(catalog) -> None:
    result = present.present(catalog, _MATRIX, _profile(), offline=True)
    for pick in result.picks:
        assert pick.explanation.templated
        assert pick.explanation.text  # never empty
    # F1 loads on t0 and t1, and t0 carried a quote, so its template cites the user's words.
    f1 = next(p for p in result.picks if p.film.movie_id == 1)
    assert "loved the quiet" in f1.explanation.text


def test_seen_films_are_never_recommended(catalog) -> None:
    record_feedback(catalog, 1, "already_seen")  # F1 is now seen
    result = present.present(catalog, _MATRIX, _profile(), offline=True, n_confident=3)
    ids = {p.film.movie_id for p in result.picks}
    assert 1 not in ids
    assert ids == {2, 3, 4}  # F4 is promoted into the picks now that F1 is gone


def test_empty_pool_gives_an_empty_presentation(catalog) -> None:
    # Exclude everything genome-covered.
    result = present.present(
        catalog, _MATRIX, _profile(), offline=True, also_exclude=frozenset(range(1, 7))
    )
    assert result.is_empty
    assert result.pool_size == 0


# -- online --------------------------------------------------------------------------


def test_online_present_reranks_and_explains(catalog) -> None:
    import json

    rerank_reply = json.dumps({"order": [3, 2, 1]})  # model reorders the near-profile films
    explain_reply = json.dumps(
        {
            "1": "The quiet you loved.",
            "2": "Same register.",
            "3": "Closest of all.",
            "4": "A stretch.",
        }
    )
    handler, queue = _queue_handler(rerank_reply, explain_reply)
    result = present.present(
        catalog, _MATRIX, _profile(), client=_make_client(handler), models=MODELS, n_confident=3
    )
    assert [p.film.movie_id for p in result.picks] == [3, 2, 1]  # the model's order
    assert not result.degraded_rerank
    assert not result.degraded_explain
    assert result.picks[0].explanation.text == "Closest of all."
    assert result.wildcard.film.movie_id == 4
    assert result.wildcard.explanation.text == "A stretch."
    assert queue == []  # exactly two calls: one rerank, one explain


def test_online_degrades_cleanly_when_the_model_fails(catalog) -> None:
    # Both calls return garbage; the phase falls back to MMR order and templated explanations.
    handler, _ = _queue_handler("nonsense", "nope", "nonsense", "nope")
    result = present.present(
        catalog, _MATRIX, _profile(), client=_make_client(handler), models=MODELS
    )
    assert {p.film.movie_id for p in result.picks} == {1, 2, 3}
    assert result.degraded_rerank
    assert result.degraded_explain
    assert all(p.explanation.templated for p in result.picks)


# -- the profile sketch --------------------------------------------------------------


def test_profile_summary_names_tags_and_quotes() -> None:
    text = present.profile_summary(_profile())
    assert "Drawn toward" in text
    assert "t0" in text
    assert "loved the quiet" in text  # the quote is woven in


def test_profile_summary_handles_an_empty_profile() -> None:
    empty = TasteProfile(
        user_id="default",
        genome_vector=np.zeros(8, dtype=np.float32),
        total_weight=0.0,
        event_count=0,
        session_count=0,
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert "No strong taste" in present.profile_summary(empty)
