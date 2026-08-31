"""Unit tests for free-text signal extraction.

The LLM is mocked with an ``httpx.MockTransport`` so the whole client path runs without a network:
we script the exact replies a model would send — clean JSON, fenced JSON, JSON buried in prose,
garbage — and assert the extractor parses, coerces, retries, and falls back as designed. A single
``@pytest.mark.network`` test at the end exercises a real free model and is excluded from CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cinegeist.config import load_settings
from cinegeist.convo import extract
from cinegeist.convo.extract import Extraction, extract_signals
from cinegeist.llm.client import OpenRouterClient

VALID = json.dumps(
    {
        "mentioned_titles": [
            {"title": "Arrival", "year": 2016, "sentiment": "loved", "quote": "loved Arrival"},
            {"title": "Tenet", "year": None, "sentiment": "bounced", "quote": "turned off Tenet"},
        ],
        "axis_signals": [{"axis": "Slow", "value": -0.6, "quote": "too slow"}],
        "constraints": [{"kind": "max_runtime", "value": 120}],
        "session_mood": "wants something light",
    }
)

MODELS = ["stub/model"]


def _make_client(handler) -> OpenRouterClient:
    settings = load_settings(
        env={"OPENROUTER_API_KEY": "sk-or-stub"}, config_file=Path("nope.toml")
    )
    # sleep is a no-op so the client's retry backoff doesn't slow the tests down.
    return OpenRouterClient(settings, transport=httpx.MockTransport(handler), sleep=lambda _s: None)


def _texts(*texts: str):
    """A handler returning each canned reply in turn; the leftover queue is exposed for asserts."""
    queue = list(texts)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": queue.pop(0)}}], "model": "stub"}
        )

    return handler, queue


def test_parses_a_clean_reply() -> None:
    handler, queue = _texts(VALID)
    result = extract_signals(_make_client(handler), "some message", MODELS)
    assert not result.degraded
    assert [m.title for m in result.mentions] == ["Arrival", "Tenet"]
    assert result.mentions[0].year == 2016
    assert result.mentions[0].sentiment == "loved"
    assert result.mentions[0].evidence == "loved Arrival"
    assert result.mentions[1].sentiment == "bounced"
    assert result.axis_signals[0].axis == "slow"  # lower-cased
    assert result.axis_signals[0].value == pytest.approx(-0.6)
    assert result.constraints[0].kind == "max_runtime"
    assert result.session_mood == "wants something light"
    assert queue == []  # exactly one call


def test_strips_code_fences() -> None:
    handler, _ = _texts(f"```json\n{VALID}\n```")
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert [m.title for m in result.mentions] == ["Arrival", "Tenet"]


def test_finds_json_buried_in_prose() -> None:
    handler, _ = _texts(f"Sure! Here you go:\n{VALID}\nHope that helps.")
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert not result.degraded
    assert result.mentions[0].title == "Arrival"


def test_missing_lists_default_to_empty() -> None:
    handler, _ = _texts('{"session_mood": "tired"}')
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert not result.degraded
    assert result.mentions == []
    assert result.session_mood == "tired"


def test_out_of_range_axis_value_is_clamped() -> None:
    handler, _ = _texts('{"axis_signals": [{"axis": "violent", "value": 5}]}')
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert result.axis_signals[0].value == 1.0


def test_a_noisy_year_does_not_lose_the_title() -> None:
    handler, _ = _texts(
        '{"mentioned_titles": [{"title": "Dune", "year": "two thousand", "sentiment": "liked"}]}'
    )
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert result.mentions[0].title == "Dune"
    assert result.mentions[0].year is None  # coerced, not rejected


def test_evidence_falls_back_to_the_whole_message() -> None:
    handler, _ = _texts('{"mentioned_titles": [{"title": "Heat", "sentiment": "loved"}]}')
    message = "Heat is a masterpiece"
    result = extract_signals(_make_client(handler), message, MODELS)
    assert result.mentions[0].evidence == message


def test_retries_once_then_succeeds() -> None:
    handler, queue = _texts("not json at all", VALID)
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert not result.degraded
    assert result.mentions[0].title == "Arrival"
    assert queue == []  # two calls: the retry consumed the second reply


def test_falls_back_to_degraded_after_two_failures() -> None:
    handler, queue = _texts("garbage one", "garbage two")
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert result.degraded
    assert result.is_empty
    assert queue == []  # tried exactly twice, then gave up


def test_non_object_json_is_treated_as_malformed() -> None:
    handler, _ = _texts("[1, 2, 3]", "[4, 5, 6]")
    result = extract_signals(_make_client(handler), "m", MODELS)
    assert result.degraded


def test_empty_message_does_not_call_the_model() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("the model must not be called for an empty message")

    result = extract_signals(_make_client(handler), "   ", MODELS)
    assert result == Extraction()
    assert not result.degraded


def test_unreachable_model_degrades_rather_than_raising() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    result = extract_signals(_make_client(handler), "m", MODELS)
    assert result.degraded


def test_context_is_added_to_the_system_prompt() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID}}]})

    extract_signals(_make_client(handler), "m", MODELS, context="films they loved")
    system = captured["body"]["messages"][0]["content"]
    assert "films they loved" in system


@pytest.mark.network
def test_extract_against_a_real_free_model() -> None:
    from cinegeist.llm.registry import free_models

    settings = load_settings()
    if not settings.has_api_key:
        pytest.skip("OPENROUTER_API_KEY is not set")
    models = free_models(settings)
    with OpenRouterClient(settings) as client:
        result = extract_signals(
            client,
            "I loved Arrival and Prisoners, but I bounced off Tenet — too loud.",
            models,
            context="films they loved",
        )
    # Always: the real path runs and never raises for a bad reply. Model availability rotates and
    # some free slugs can't chat, so a degraded result here is an environment issue, not a bug.
    assert isinstance(result, Extraction)
    if not result.degraded:
        titles = " ".join(m.title.lower() for m in result.mentions)
        assert any(name in titles for name in ("arrival", "prisoners", "tenet"))


def test_prompt_file_loads() -> None:
    # The prompt is a real resource in the package, not an inline string.
    assert "JSON" in extract.load_prompt("extract")
