"""Discover which OpenRouter models are free right now.

Free-model ids rotate constantly, so this project never hardcodes one as the default
(hard rule 3). Instead it fetches the live list from ``{base}/models``, keeps the entries
priced at zero for both prompt and completion *and* usable for chat — text in, text-only out,
which drops the odd free media model like Google's Lyria that would otherwise top the list —
ranks them by a small curated preference order, and caches the result for a day.

``FALLBACK_FREE_MODELS`` is a last resort used only when the endpoint can't be reached.
It is assumed **stale** — any entry may have lost its free tier — so it is never cached and
discovery always wins when it can run.

The ``/models`` endpoint is public, so this needs no API key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings, cache_dir

CACHE_TTL_SECONDS: float = 24 * 60 * 60
_CACHE_FILENAME = "models.json"

# Preference by substring: better instruction-following / JSON reliability first. This only
# orders models already confirmed free; it never introduces an id that discovery didn't see.
_PREFERENCE_ORDER: tuple[str, ...] = (
    "llama-3.3",
    "llama-3.1",
    "qwen-2.5",
    "qwen2.5",
    "gemma-2",
    "mistral",
    "phi-3",
)

# Emergency fallback only (endpoint unreachable). ASSUMED STALE: any of these may no longer
# be free. Discovery is the real source of truth; this just keeps the CLI usable offline.
FALLBACK_FREE_MODELS: tuple[str, ...] = (
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
)


def _is_zero(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _is_free(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    return _is_zero(pricing.get("prompt")) and _is_zero(pricing.get("completion"))


def _as_modality_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip().lower() for item in value if str(item).strip()}


def _modalities(model: dict[str, Any], list_key: str, side: int) -> set[str]:
    """The input (``side=0``) or output (``side=1``) modalities of a model.

    Prefers the explicit ``input_modalities`` / ``output_modalities`` list; falls back to the
    half of a ``"text+image->text+audio"`` ``modality`` string on the requested side.
    """
    arch = model.get("architecture") or {}
    explicit = _as_modality_set(arch.get(list_key))
    if explicit:
        return explicit
    modality = arch.get("modality")
    if isinstance(modality, str) and "->" in modality:
        half = modality.split("->", 1)[side]
        return {part.strip().lower() for part in half.split("+") if part.strip()}
    return set()


def _is_text_model(model: dict[str, Any]) -> bool:
    """A model we can prompt with text and that replies with text only.

    OpenRouter lists the occasional free entry that is really a media model — Google's Lyria, for
    one, declares an ``audio`` output alongside ``text`` — and those can't answer a chat
    completion, so they must be dropped before ranking or they surface at the top of the free list
    and break every LLM feature. Metadata is trusted only to *exclude*: a model is rejected only
    when the data positively says it emits a non-text modality (audio, image, video) or can't take
    text in. A model with no architecture info is kept, so a missing field never hides a good one.
    """
    outputs = _modalities(model, "output_modalities", side=1)
    inputs = _modalities(model, "input_modalities", side=0)
    output_is_text_only = outputs == {"text"} if outputs else True
    input_accepts_text = "text" in inputs if inputs else True
    return output_is_text_only and input_accepts_text


def _rank_key(model: dict[str, Any]) -> tuple[int, int, str]:
    model_id = model.get("id", "")
    preference = len(_PREFERENCE_ORDER)  # unknown models sort after every known one
    for index, needle in enumerate(_PREFERENCE_ORDER):
        if needle in model_id:
            preference = index
            break
    try:
        context = int(model.get("context_length") or 0)
    except (TypeError, ValueError):
        context = 0
    return (preference, -context, model_id)  # larger context wins within a preference tier


def _rank_free_models(models: Iterable[dict[str, Any]]) -> list[str]:
    usable = (m for m in models if _is_free(m) and _is_text_model(m))
    free = sorted(usable, key=_rank_key)
    return [m["id"] for m in free if m.get("id")]


def _cache_file() -> Path:
    return cache_dir() / _CACHE_FILENAME


def _read_cache(path: Path, now: float, ttl: float) -> list[str] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = float(data["fetched_at"])
        models = [str(m) for m in data["models"]]
    except (ValueError, KeyError, TypeError, OSError):
        return None
    if now - fetched_at > ttl:
        return None
    return models or None


def _write_cache(path: Path, models: list[str], now: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"fetched_at": now, "models": models})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)  # atomic swap so a crash mid-write can't leave a torn cache


def _fetch_models(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None,
    http_client: httpx.Client | None,
) -> list[dict[str, Any]]:
    url = f"{settings.api_base_url.rstrip('/')}/models"
    client = http_client or httpx.Client(timeout=settings.request_timeout, transport=transport)
    try:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
    finally:
        if http_client is None:
            client.close()
    models = data.get("data", [])
    return list(models) if isinstance(models, list) else []


def free_models(
    settings: Settings,
    *,
    force_refresh: bool = False,
    transport: httpx.BaseTransport | None = None,
    http_client: httpx.Client | None = None,
    now: float | None = None,
    cache_path: Path | None = None,
    ttl: float = CACHE_TTL_SECONDS,
) -> list[str]:
    """Return currently-free model ids, best first.

    Uses the day-old cache unless ``force_refresh`` is set. On any network/endpoint error,
    or when the endpoint reports nothing free, returns the (stale) fallback list without
    caching it, so the next run tries discovery again.
    """
    now = time.time() if now is None else now
    cache_path = cache_path or _cache_file()

    if not force_refresh:
        cached = _read_cache(cache_path, now, ttl)
        if cached is not None:
            return cached

    try:
        raw = _fetch_models(settings, transport=transport, http_client=http_client)
    except (httpx.HTTPError, ValueError, KeyError):
        return list(FALLBACK_FREE_MODELS)

    ranked = _rank_free_models(raw)
    if not ranked:
        return list(FALLBACK_FREE_MODELS)

    _write_cache(cache_path, ranked, now)
    return ranked


def best_free_model(settings: Settings, **kwargs: Any) -> str | None:
    """The single best free model id, or None if the list is somehow empty."""
    models = free_models(settings, **kwargs)
    return models[0] if models else None
