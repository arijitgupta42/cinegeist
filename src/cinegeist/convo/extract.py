"""Turn a free-text answer into structured taste signals with one LLM call.

The model's only job here is reading comprehension: it reads what the user wrote and reports the
films they named, how they felt, any taste descriptors (pace, tone), and any constraints — as a
strict JSON object. It never recommends and never invents; titles it emits are resolved against
the catalog elsewhere (:mod:`cinegeist.convo.resolve`), so a hallucinated one is simply dropped.

Small free models misformat JSON often, so this is defensive by design (CLAUDE.md testing):

* the reply is stripped of code fences and prose before parsing;
* it is validated against a schema, with noisy per-field values coerced rather than rejected;
* a structurally broken reply is retried once with the error appended;
* a second failure falls back to an empty, ``degraded`` result — the conversation never crashes.

Budget: the happy path is one call (hard rule 4). The single retry is the exception, spent only
to rescue a turn a model garbled, not on every turn.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ..llm.client import LLMError, OpenRouterClient
from ..llm.prompt_loader import load_prompt
from .resolve import Mention

_PROMPT_NAME = "extract"
_MAX_ATTEMPTS = 2  # one try, then one retry with the error appended
_MAX_TOKENS = 512  # the JSON is small; cap so a rambling model can't burn the budget
_RETRY_NUDGE = (
    "Your previous reply was not valid JSON for the schema ({error}). "
    "Reply with ONLY the JSON object — all four keys, nothing else."
)

_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?")
_TRAILING_FENCE_RE = re.compile(r"\n?```$")


# -- the structured result -----------------------------------------------------------


@dataclass(frozen=True)
class AxisSignal:
    """A taste descriptor the message expressed, signed in ``[-1, 1]``, with the user's words."""

    axis: str
    value: float
    quote: str | None = None


@dataclass(frozen=True)
class Constraint:
    """A session constraint ("under two hours", "no subtitles tonight"). Not long-term taste."""

    kind: str
    value: float | int | str


@dataclass(frozen=True)
class Extraction:
    """Everything one message yielded. ``degraded`` marks a fallback after the model failed."""

    mentions: list[Mention] = field(default_factory=list)
    axis_signals: list[AxisSignal] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    session_mood: str | None = None
    degraded: bool = False

    @property
    def is_empty(self) -> bool:
        return not (self.mentions or self.axis_signals or self.constraints or self.session_mood)


# -- the raw schema the model must produce -------------------------------------------
#
# Per-field noise (a year written as a word, a value out of range) is coerced to something sane
# rather than raising, so one sloppy field can't cost a whole turn. Only structurally broken JSON
# — not an object, or missing everything — falls through to a retry.


class _RawTitle(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    year: int | None = None
    sentiment: str | None = None
    quote: str | None = None

    @field_validator("year", mode="before")
    @classmethod
    def _coerce_year(cls, value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None


class _RawAxis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    axis: str
    value: float = 0.0
    quote: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, value: object) -> float:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return max(-1.0, min(1.0, number))


class _RawConstraint(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: str
    value: int | float | str


class _RawExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mentioned_titles: list[_RawTitle] = []
    axis_signals: list[_RawAxis] = []
    constraints: list[_RawConstraint] = []
    session_mood: str | None = None


# -- parsing -------------------------------------------------------------------------


def _loads(text: str) -> object:
    """Parse a model reply into JSON, tolerating code fences and surrounding prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _TRAILING_FENCE_RE.sub("", _LEADING_FENCE_RE.sub("", stripped)).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if 0 <= start < end:
            return json.loads(stripped[start : end + 1])  # may still raise — caught by caller
        raise


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip()
    return cleaned or None


def _to_mention(title: _RawTitle, message: str) -> Mention:
    sentiment = _clean(title.sentiment)
    return Mention(
        title=title.title.strip(),
        year=title.year,
        sentiment=sentiment.lower() if sentiment else None,
        evidence=_clean(title.quote) or message,  # cite the quote, else the whole message
    )


def _to_extraction(raw: _RawExtraction, message: str) -> Extraction:
    """Convert the validated raw schema into the public result, filling evidence from the text."""
    mentions = [
        _to_mention(title, message) for title in raw.mentioned_titles if title.title.strip()
    ]
    axis_signals = [
        AxisSignal(axis=axis.axis.strip().lower(), value=axis.value, quote=_clean(axis.quote))
        for axis in raw.axis_signals
        if axis.axis.strip()
    ]
    constraints = [
        Constraint(kind=c.kind.strip().lower(), value=c.value)
        for c in raw.constraints
        if str(c.kind).strip()
    ]
    return Extraction(
        mentions=mentions,
        axis_signals=axis_signals,
        constraints=constraints,
        session_mood=_clean(raw.session_mood),
    )


# -- the one call --------------------------------------------------------------------


def extract_signals(
    client: OpenRouterClient,
    message: str,
    models: Sequence[str],
    *,
    context: str | None = None,
    temperature: float = 0.0,
    max_attempts: int = _MAX_ATTEMPTS,
) -> Extraction:
    """Extract structured signals from ``message``. Never raises for a bad model reply.

    ``context`` is a short note about what the user was asked (e.g. "naming films they loved"), to
    steer sentiment. On a malformed reply the call is retried once with the error appended; if that
    also fails — or the LLM is unreachable — an empty ``degraded`` extraction is returned so the
    caller can carry on rather than crash.
    """
    if not message.strip():
        return Extraction()

    system = load_prompt(_PROMPT_NAME)
    if context:
        system = f"{system}\n\nContext: the user was just asked about {context.strip()}."
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]

    for _attempt in range(max_attempts):
        try:
            result = client.chat_with_failover(
                conversation, models, temperature=temperature, max_tokens=_MAX_TOKENS
            )
        except LLMError:
            break  # auth failure or every model down — nothing to recover this turn
        try:
            raw = _RawExtraction.model_validate(_loads(result.text))
            return _to_extraction(raw, message)
        except (ValueError, ValidationError) as error:
            nudge = _RETRY_NUDGE.format(error=client.redact(str(error))[:300])
            conversation = [
                *conversation,
                {"role": "assistant", "content": result.text[:2000]},
                {"role": "user", "content": nudge},
            ]

    return Extraction(degraded=True)
