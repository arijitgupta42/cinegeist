"""Unit tests for the LLM rerank and, above all, its id validation (CLAUDE.md hard rule 1).

The pure ``apply_order`` is tested directly — it is where hallucinated ids are dropped and omitted
films are kept. The full ``rerank`` runs against an ``httpx.MockTransport`` so the whole client
path exercises without a network: clean replies, bare arrays, fenced JSON, invented ids, and
garbage that must fall back to the deterministic order without crashing.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from cinegeist.config import load_settings
from cinegeist.llm.client import OpenRouterClient
from cinegeist.recommend import rerank
from cinegeist.recommend.score import ScoredFilm

MODELS = ["stub/model"]


def _film(movie_id: int, title: str = "", year: int | None = 2000) -> ScoredFilm:
    return ScoredFilm(
        movie_id=movie_id,
        genome_row=movie_id,
        pool_index=movie_id,
        title=title or f"Film {movie_id}",
        year=year,
        score=0.0,
        cosine=0.0,
        quality=0.0,
        session_fit=0.0,
        facet_match=0.0,
        popularity_penalty=0.0,
        confidence=1.0,
    )


SHORTLIST = [_film(1), _film(2), _film(3), _film(4)]


def _make_client(handler) -> OpenRouterClient:
    settings = load_settings(
        env={"OPENROUTER_API_KEY": "sk-or-stub"}, config_file=Path("nope.toml")
    )
    return OpenRouterClient(settings, transport=httpx.MockTransport(handler), sleep=lambda _s: None)


def _texts(*texts: str):
    """A handler returning each canned reply in turn; the leftover queue is exposed for asserts."""
    queue = list(texts)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": queue.pop(0)}}], "model": "stub"}
        )

    return handler, queue


def _order(outcome: rerank.RerankOutcome) -> list[int]:
    return [f.movie_id for f in outcome.ordered]


# -- apply_order: the id-validation core ---------------------------------------------


def test_apply_order_reorders_and_keeps_every_film_once() -> None:
    ordered, dropped = rerank.apply_order(SHORTLIST, [3, 1])
    # 3 and 1 go first as ranked; 2 and 4 keep their original order after.
    assert [f.movie_id for f in ordered] == [3, 1, 2, 4]
    assert dropped == []


def test_apply_order_drops_hallucinated_ids() -> None:
    ordered, dropped = rerank.apply_order(SHORTLIST, [2, 999, 1, 424242])
    assert [f.movie_id for f in ordered] == [2, 1, 3, 4]  # invented ids never appear
    assert dropped == [999, 424242]


def test_apply_order_ignores_repeats() -> None:
    ordered, dropped = rerank.apply_order(SHORTLIST, [2, 2, 2])
    assert [f.movie_id for f in ordered] == [2, 1, 3, 4]
    assert dropped == []


# -- rerank: the full call -----------------------------------------------------------


def test_rerank_applies_a_clean_order() -> None:
    handler, queue = _texts(json.dumps({"order": [4, 2, 1, 3]}))
    outcome = rerank.rerank(_make_client(handler), SHORTLIST, MODELS)
    assert _order(outcome) == [4, 2, 1, 3]
    assert not outcome.degraded
    assert outcome.dropped_ids == ()
    assert queue == []  # exactly one call


def test_rerank_accepts_a_bare_array_and_numeric_strings() -> None:
    handler, _ = _texts('[3, "1", 2]')  # bare list, one id as a string
    outcome = rerank.rerank(_make_client(handler), SHORTLIST, MODELS)
    assert _order(outcome) == [3, 1, 2, 4]
    assert not outcome.degraded


def test_rerank_strips_code_fences() -> None:
    handler, _ = _texts('```json\n{"order": [2, 1]}\n```')
    outcome = rerank.rerank(_make_client(handler), SHORTLIST, MODELS)
    assert _order(outcome) == [2, 1, 3, 4]


def test_rerank_drops_ids_not_in_the_shortlist() -> None:
    handler, _ = _texts(json.dumps({"order": [2, 777, 1]}))
    outcome = rerank.rerank(_make_client(handler), SHORTLIST, MODELS)
    assert _order(outcome) == [2, 1, 3, 4]
    assert outcome.dropped_ids == (777,)
    assert not outcome.degraded  # a dropped hallucination is not a degraded turn


def test_rerank_retries_once_then_succeeds() -> None:
    handler, queue = _texts("not json at all", json.dumps({"order": [4, 3, 2, 1]}))
    outcome = rerank.rerank(_make_client(handler), SHORTLIST, MODELS)
    assert _order(outcome) == [4, 3, 2, 1]
    assert not outcome.degraded
    assert queue == []  # the retry was spent


def test_rerank_falls_back_to_deterministic_order_on_garbage() -> None:
    handler, _ = _texts("nope", "still nope")  # both attempts unusable
    outcome = rerank.rerank(_make_client(handler), SHORTLIST, MODELS)
    assert _order(outcome) == [1, 2, 3, 4]  # the shortlist's own order, untouched
    assert outcome.degraded
    assert outcome.model is None


def test_rerank_degrades_when_the_model_is_unreachable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no such model"})

    outcome = rerank.rerank(_make_client(handler), SHORTLIST, MODELS)
    assert _order(outcome) == [1, 2, 3, 4]
    assert outcome.degraded


def test_rerank_short_circuits_without_calling_the_model() -> None:
    # An empty or single-item shortlist has nothing to reorder, so no call is made.
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    client = _make_client(handler)
    assert rerank.rerank(client, [], MODELS).ordered == []
    single = rerank.rerank(client, [_film(1)], MODELS)
    assert _order(single) == [1]
    assert not called


def test_rerank_passes_tags_into_the_prompt() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "[1,2,3,4]"}}], "model": "stub"}
        )

    rerank.rerank(
        _make_client(handler),
        SHORTLIST,
        MODELS,
        profile_summary="Drawn toward: slow burn.",
        tags_by_id={1: ["slow burn", "dread"]},
    )
    assert "slow burn" in captured["body"]
    assert "Drawn toward" in captured["body"]
