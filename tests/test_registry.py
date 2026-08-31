"""Unit tests for free-model discovery. HTTP is mocked; the cache path is a tmp file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from cinegeist.config import Settings
from cinegeist.llm.registry import (
    CACHE_TTL_SECONDS,
    FALLBACK_FREE_MODELS,
    best_free_model,
    free_models,
)


def settings() -> Settings:
    return Settings(api_base_url="https://openrouter.ai/api/v1", request_timeout=5.0)


def model(
    model_id: str,
    prompt: str = "0",
    completion: str = "0",
    context: int = 8192,
    *,
    input_modalities: list[str] | None = None,
    output_modalities: list[str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model_id,
        "pricing": {"prompt": prompt, "completion": completion},
        "context_length": context,
    }
    # Only attach an architecture block when a test cares about modality; models without one keep
    # the old shape, so the discovery must treat "no metadata" as usable (see the leniency test).
    if input_modalities is not None or output_modalities is not None:
        entry["architecture"] = {
            "input_modalities": input_modalities if input_modalities is not None else ["text"],
            "output_modalities": output_modalities if output_modalities is not None else ["text"],
        }
    return entry


def transport_returning(*models: dict[str, Any], status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(status, json={"data": list(models)})

    return httpx.MockTransport(handler)


def test_keeps_only_models_that_are_free_on_both_sides(tmp_path: Path) -> None:
    ids = free_models(
        settings(),
        transport=transport_returning(
            model("free/a:free"),
            model("paid/b", prompt="0.001", completion="0.002"),
            model("half/c", prompt="0", completion="0.01"),
            model("free/d:free"),
        ),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert set(ids) == {"free/a:free", "free/d:free"}


def test_drops_media_models_that_cannot_chat(tmp_path: Path) -> None:
    # Google's Lyria is priced at zero but outputs audio alongside text, so it can't answer a chat
    # completion. It must not surface above real chat models on the free list.
    ids = free_models(
        settings(),
        transport=transport_returning(
            model(
                "google/lyria-3:free",
                input_modalities=["text", "image"],
                output_modalities=["text", "audio"],
            ),
            model(
                "minimax/minimax-m3:free",
                input_modalities=["text", "image", "video"],
                output_modalities=["text"],
            ),
            model("meta/llama-3.3-70b:free", output_modalities=["text"]),
        ),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert "google/lyria-3:free" not in ids
    assert set(ids) == {"minimax/minimax-m3:free", "meta/llama-3.3-70b:free"}


def test_drops_image_output_models(tmp_path: Path) -> None:
    ids = free_models(
        settings(),
        transport=transport_returning(
            model("img/generator:free", output_modalities=["image"]),
            model("chat/words:free", output_modalities=["text"]),
        ),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert ids == ["chat/words:free"]


def test_modality_string_is_used_when_lists_are_absent(tmp_path: Path) -> None:
    # Some entries carry only a "text->image" style string; the output half still filters them.
    def with_modality(model_id: str, modality: str) -> dict[str, Any]:
        entry = model(model_id)
        entry["architecture"] = {"modality": modality}
        return entry

    ids = free_models(
        settings(),
        transport=transport_returning(
            with_modality("gen/pics:free", "text->image"),
            with_modality("chat/words:free", "text->text"),
        ),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert ids == ["chat/words:free"]


def test_models_without_architecture_metadata_are_kept(tmp_path: Path) -> None:
    # Trust metadata only to exclude: a model that reports no modalities is still usable.
    ids = free_models(
        settings(),
        transport=transport_returning(model("legacy/no-meta:free")),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert ids == ["legacy/no-meta:free"]


def test_ranks_by_preference_then_context(tmp_path: Path) -> None:
    ids = free_models(
        settings(),
        transport=transport_returning(
            model("x/mistral-7b:free", context=8000),
            model("meta/llama-3.3-70b:free", context=131072),
            model("z/unknown-big:free", context=200000),
            model("y/unknown-small:free", context=4096),
        ),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert ids == [
        "meta/llama-3.3-70b:free",  # curated preference wins
        "x/mistral-7b:free",  # next curated preference
        "z/unknown-big:free",  # unknowns fall back to larger-context-first
        "y/unknown-small:free",
    ]


def test_caches_and_reuses_within_the_ttl(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    first = free_models(
        settings(),
        transport=transport_returning(model("a:free")),
        cache_path=cache,
        now=1000.0,
    )
    assert first == ["a:free"]
    assert cache.exists()

    # Within the TTL the cache is used even though this transport would 500.
    def boom(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    second = free_models(
        settings(),
        transport=httpx.MockTransport(boom),
        cache_path=cache,
        now=1000.0 + 100,
    )
    assert second == ["a:free"]


def test_a_stale_cache_is_refetched(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    free_models(
        settings(),
        transport=transport_returning(model("old:free")),
        cache_path=cache,
        now=1000.0,
    )
    fresh = free_models(
        settings(),
        transport=transport_returning(model("new:free")),
        cache_path=cache,
        now=1000.0 + CACHE_TTL_SECONDS + 1,
    )
    assert fresh == ["new:free"]


def test_force_refresh_bypasses_a_fresh_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    free_models(
        settings(),
        transport=transport_returning(model("old:free")),
        cache_path=cache,
        now=1000.0,
    )
    fresh = free_models(
        settings(),
        transport=transport_returning(model("new:free")),
        cache_path=cache,
        now=1000.0 + 1,
        force_refresh=True,
    )
    assert fresh == ["new:free"]


def test_falls_back_when_the_endpoint_is_unreachable(tmp_path: Path) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    ids = free_models(
        settings(),
        transport=httpx.MockTransport(unreachable),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert ids == list(FALLBACK_FREE_MODELS)


def test_falls_back_on_a_server_error(tmp_path: Path) -> None:
    ids = free_models(
        settings(),
        transport=transport_returning(status=503),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert ids == list(FALLBACK_FREE_MODELS)


def test_falls_back_when_nothing_is_free(tmp_path: Path) -> None:
    ids = free_models(
        settings(),
        transport=transport_returning(model("paid", prompt="0.01", completion="0.02")),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert ids == list(FALLBACK_FREE_MODELS)
    # The fallback is not cached, so a later run can rediscover.
    assert not (tmp_path / "cache.json").exists()


def test_best_free_model_returns_the_top_ranked(tmp_path: Path) -> None:
    best = best_free_model(
        settings(),
        transport=transport_returning(model("z/other:free"), model("meta/llama-3.3:free")),
        cache_path=tmp_path / "cache.json",
        now=1000.0,
    )
    assert best == "meta/llama-3.3:free"
