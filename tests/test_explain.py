"""Unit tests for evidence-grounded explanations.

``evidence_for_picks`` and the template are pure, so they're tested against a hand-built profile
with known axes and quotes. The full ``explain`` runs against an ``httpx.MockTransport``: a clean
reply, a partial one (missing picks keep their template), and garbage that must fall back to the
template for every pick without crashing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from cinegeist.config import load_settings
from cinegeist.llm.client import OpenRouterClient
from cinegeist.profile.model import TagAffinity, TasteProfile
from cinegeist.recommend import explain
from cinegeist.recommend.score import ScoredFilm

MODELS = ["stub/model"]


def _profile() -> TasteProfile:
    axes = (
        TagAffinity(
            position=0,
            name="slow burn",
            weight=0.8,
            source="'Hereditary'",
            evidence="loved the creeping dread in Hereditary",
        ),
        TagAffinity(position=1, name="dread", weight=0.6, source="'Hereditary'", evidence=None),
        TagAffinity(position=2, name="loud", weight=-0.5, source="'Tenet'", evidence="too loud"),
    )
    return TasteProfile(
        user_id="default",
        genome_vector=np.array([0.8, 0.6, -0.5], dtype=np.float32),
        total_weight=3.0,
        event_count=3,
        session_count=1,
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
        axes=axes,
    )


def _film(movie_id: int, title: str) -> ScoredFilm:
    return ScoredFilm(
        movie_id=movie_id,
        genome_row=movie_id,
        pool_index=movie_id,
        title=title,
        year=1980 + movie_id,
        score=0.0,
        cosine=0.0,
        quality=0.0,
        session_fit=0.0,
        facet_match=0.0,
        popularity_penalty=0.0,
        confidence=1.0,
    )


# The three picks and the genome rows the engine would hand in:
#   1 loads on slow burn and dread; 2 (wildcard) loads on dread only; 3 loads on nothing strong.
PICKS = [_film(1, "The Fog"), _film(2, "Paddington"), _film(3, "Beige")]
VECTORS = {
    1: np.array([0.9, 0.7, 0.1], dtype=np.float32),
    2: np.array([0.1, 0.6, 0.0], dtype=np.float32),
    3: np.array([0.2, 0.2, 0.2], dtype=np.float32),
}


def _make_client(handler) -> OpenRouterClient:
    settings = load_settings(
        env={"OPENROUTER_API_KEY": "sk-or-stub"}, config_file=Path("nope.toml")
    )
    return OpenRouterClient(settings, transport=httpx.MockTransport(handler), sleep=lambda _s: None)


def _texts(*texts: str):
    queue = list(texts)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": queue.pop(0)}}], "model": "stub"}
        )

    return handler, queue


# -- evidence assembly (pure) --------------------------------------------------------


def test_evidence_picks_out_shared_tags_and_quotes() -> None:
    evidence = explain.evidence_for_picks(_profile(), PICKS, VECTORS, wildcard_id=2)
    by_id = {e.movie_id: e for e in evidence}
    assert by_id[1].shared_tags == ["slow burn", "dread"]
    assert by_id[1].quotes == ["loved the creeping dread in Hereditary"]  # only slow burn had one
    assert not by_id[1].is_wildcard
    assert by_id[2].shared_tags == ["dread"]  # loads on dread, not slow burn
    assert by_id[2].is_wildcard
    assert by_id[3].shared_tags == []  # nothing at or above the relevance floor


def test_aversion_axes_are_not_offered_as_shared_tags() -> None:
    # A film that loads hard on the disliked "loud" axis still shares nothing positive.
    picks = [_film(9, "Loud One")]
    vectors = {9: np.array([0.0, 0.0, 0.95], dtype=np.float32)}
    (evidence,) = explain.evidence_for_picks(_profile(), picks, vectors)
    assert evidence.shared_tags == []


# -- the template --------------------------------------------------------------------


def test_templated_explanation_cites_tags_and_quote() -> None:
    (a, b, c) = explain.evidence_for_picks(_profile(), PICKS, VECTORS, wildcard_id=2)
    text_a = explain.templated_explanation(a)
    assert "slow burn and dread" in text_a
    assert "loved the creeping dread in Hereditary" in text_a

    text_b = explain.templated_explanation(b)  # the wildcard
    assert "Further from your usual" in text_b
    assert "dread" in text_b

    text_c = explain.templated_explanation(c)  # no shared tags — honest, not marketing
    assert "overall" in text_c


def test_join_tags_reads_naturally() -> None:
    assert explain._join_tags(["a"]) == "a"
    assert explain._join_tags(["a", "b"]) == "a and b"
    assert explain._join_tags(["a", "b", "c"]) == "a, b, and c"


# -- the LLM call --------------------------------------------------------------------


def test_explain_uses_the_models_sentences() -> None:
    evidence = explain.evidence_for_picks(_profile(), PICKS, VECTORS, wildcard_id=2)
    reply = json.dumps(
        {"1": "Creeping dread, drawn out slow.", "2": "A wilder pick.", "3": "Quietly your speed."}
    )
    handler, queue = _texts(reply)
    result = explain.explain(_make_client(handler), evidence, MODELS)
    assert result[1].text == "Creeping dread, drawn out slow."
    assert not result[1].templated
    assert queue == []  # one call for all three picks


def test_explain_keeps_the_template_for_picks_the_model_skips() -> None:
    evidence = explain.evidence_for_picks(_profile(), PICKS, VECTORS, wildcard_id=2)
    handler, _ = _texts(json.dumps({"1": "Creeping dread, drawn out slow."}))  # 2 and 3 missing
    result = explain.explain(_make_client(handler), evidence, MODELS)
    assert not result[1].templated
    assert result[2].templated and result[3].templated  # fell back, still grounded
    assert set(result) == {1, 2, 3}  # every pick is covered


def test_explain_falls_back_entirely_on_garbage() -> None:
    evidence = explain.evidence_for_picks(_profile(), PICKS, VECTORS, wildcard_id=2)
    handler, _ = _texts("not json", "still not json")
    result = explain.explain(_make_client(handler), evidence, MODELS)
    assert all(e.templated for e in result.values())
    assert "slow burn and dread" in result[1].text  # the grounded template, not a crash


def test_explain_falls_back_when_model_unreachable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no model"})

    evidence = explain.evidence_for_picks(_profile(), PICKS, VECTORS, wildcard_id=2)
    result = explain.explain(_make_client(handler), evidence, MODELS)
    assert all(e.templated for e in result.values())


def test_explain_ignores_ids_it_did_not_ask_about() -> None:
    evidence = explain.evidence_for_picks(_profile(), PICKS, VECTORS, wildcard_id=2)
    handler, _ = _texts(json.dumps({"1": "Good.", "999": "Invented film.", "3": ""}))
    result = explain.explain(_make_client(handler), evidence, MODELS)
    assert 999 not in result
    assert result[1].text == "Good." and not result[1].templated
    assert result[3].templated  # empty string ignored, template kept


def test_explain_on_no_picks_is_empty() -> None:
    handler, _ = _texts("{}")
    assert explain.explain(_make_client(handler), [], MODELS) == {}
