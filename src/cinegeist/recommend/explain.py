"""Explain each pick in the user's own evidence, never in marketing copy (plan.md §6).

A recommendation the user can't trace is a recommendation they won't trust. So every pick gets one
sentence that points back at *why we think they'll like it* — the tags it shares with their taste,
and, when they gave us words, their own quote behind those tags:

    "The same creeping dread you loved in Hereditary, drawn out slow."   ✓  cites their evidence
    "A gripping thriller that will keep you on the edge of your seat!"     ✗  could be any film

Two layers, same shape. :func:`evidence_for_picks` is pure: it reads the profile's strong axes and
each film's genome row to work out which tags a pick actually shares and which quotes produced
them. :func:`templated_explanation` turns that into a deterministic sentence — the ``--offline``
path and the fallback. :func:`explain` spends one LLM call (hard rule 4) to phrase all the picks
at once, more naturally, and drops back to the template for any pick the model fumbles. Either
way the sentence is grounded in the user's own evidence, never invented.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..llm.client import LLMError, OpenRouterClient
from ..llm.prompt_loader import load_prompt
from ..profile.model import TasteProfile
from .score import ScoredFilm

_PROMPT_NAME = "explain"
_MAX_ATTEMPTS = 2
# Room for one or two textured sentences per pick (the prompt asks the model to vary its angle),
# with headroom so a verbose model isn't truncated into invalid JSON before the closing brace.
_MAX_TOKENS = 600
_RETRY_NUDGE = (
    "Your previous reply was not valid JSON ({error}). Reply with ONLY a JSON object mapping "
    'each id (as a string) to one sentence, e.g. {{"12": "..."}}.'
)

# A pick "shares" a profile tag when it loads at least this strongly on that tag's axis.
EXPLAIN_TAG_RELEVANCE = 0.5
# How many shared tags / quotes to hand the phrasing, per pick. A little generous so the model has
# distinct material to differentiate picks with — several tags to lead on, and every quote the pick
# earned rather than only the top one (which is the same across picks and drives repetition).
_MAX_TAGS = 4
_MAX_QUOTES = 3
_QUOTE_CHARS = 90  # trim a rambling quote so the sentence stays a sentence

_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?")
_TRAILING_FENCE_RE = re.compile(r"\n?```$")


@dataclass(frozen=True)
class PickEvidence:
    """Why one pick is being recommended: the tags it shares with the user, and their words.

    ``shared_tags`` are the profile's own strong affinities that this film also loads on, strongest
    first; ``quotes`` are the verbatim things the user said that produced those axes (deduped).
    ``is_wildcard`` marks the deliberate stretch, phrased as one. When ``shared_tags`` is empty the
    film matched on the overall centroid rather than any single strong axis — still honest, just
    less quotable.
    """

    movie_id: int
    title: str
    year: int | None
    shared_tags: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    is_wildcard: bool = False


@dataclass(frozen=True)
class Explanation:
    """One pick's finished sentence. ``templated`` is true when it came from the deterministic
    fallback rather than the model."""

    movie_id: int
    text: str
    templated: bool


def evidence_for_picks(
    profile: TasteProfile,
    picks: Sequence[ScoredFilm],
    vectors: Mapping[int, np.ndarray],
    *,
    wildcard_id: int | None = None,
    max_tags: int = _MAX_TAGS,
    max_quotes: int = _MAX_QUOTES,
) -> list[PickEvidence]:
    """Work out which of the user's strong tags each pick shares, and the quotes behind them.

    Pure: it needs only the profile's ranked axes (which already carry each axis's position, name,
    and best evidence quote) and each film's genome row (``vectors[movie_id]``). A film shares a
    tag when it loads at or above :data:`EXPLAIN_TAG_RELEVANCE` on one of the profile's positive
    affinities; the strongest affinities are considered first, so the most defining tags win the
    limited slots.
    """
    affinities = profile.affinities  # strong positive axes, strongest first
    out: list[PickEvidence] = []
    for pick in picks:
        vector = vectors.get(pick.movie_id)
        shared: list[str] = []
        quotes: list[str] = []
        if vector is not None:
            for axis in affinities:
                if axis.position >= vector.shape[0]:
                    continue
                if float(vector[axis.position]) < EXPLAIN_TAG_RELEVANCE:
                    continue
                shared.append(axis.name)
                if axis.evidence:
                    quote = " ".join(axis.evidence.split())[:_QUOTE_CHARS]
                    if quote and quote not in quotes:
                        quotes.append(quote)
                if len(shared) >= max_tags:
                    break
        out.append(
            PickEvidence(
                movie_id=pick.movie_id,
                title=pick.title,
                year=pick.year,
                shared_tags=shared,
                quotes=quotes[:max_quotes],
                is_wildcard=pick.movie_id == wildcard_id,
            )
        )
    return out


def _join_tags(tags: Sequence[str]) -> str:
    """A natural-language join: "a", "a and b", "a, b, and c"."""
    tags = list(tags)
    if len(tags) == 1:
        return tags[0]
    if len(tags) == 2:
        return f"{tags[0]} and {tags[1]}"
    return f"{', '.join(tags[:-1])}, and {tags[-1]}"


def templated_explanation(pick: PickEvidence) -> str:
    """A deterministic, evidence-grounded sentence — the offline path and the per-pick fallback.

    Built from the shared tags and, when present, the user's own quote, so it names something
    specific rather than reaching for marketing words. When nothing strong is shared it stays
    honest about that instead of inventing a reason.
    """
    if not pick.shared_tags:
        if pick.is_wildcard:
            return "A wilder pick — further from your usual taste, offered to widen the net."
        return "Close to where your taste sits overall, without leaning on one single trait."

    tags = _join_tags(pick.shared_tags)
    if pick.is_wildcard:
        sentence = f"Further from your usual, but it shares the {tags} you're drawn to."
    else:
        sentence = f"Leans into the {tags} you keep coming back to."
    if pick.quotes:
        sentence += f' You said: "{pick.quotes[0]}".'
    return sentence


def templated_explanations(picks: Sequence[PickEvidence]) -> dict[int, Explanation]:
    """Every pick explained deterministically — the ``--offline`` recommender's explanations."""
    return {
        pick.movie_id: Explanation(pick.movie_id, templated_explanation(pick), templated=True)
        for pick in picks
    }


def _pick_line(pick: PickEvidence) -> str:
    year = f" ({pick.year})" if pick.year else ""
    tag_part = ", ".join(pick.shared_tags) if pick.shared_tags else "no single strong tag"
    line = f'- id {pick.movie_id} — "{pick.title}"{year}'
    if pick.is_wildcard:
        line += " — wildcard"
    line += f" — shares: {tag_part}"
    if pick.quotes:
        # Hand over every quote the pick earned, not just the first — the first is often the same
        # across picks, so a model given only that repeats it. With several, the prompt can spread
        # a different one over each pick.
        quoted = "; ".join(f'"{q}"' for q in pick.quotes)
        line += f" — they said: {quoted}"
    else:
        line += " — (no quote)"
    return line


def _loads(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _TRAILING_FENCE_RE.sub("", _LEADING_FENCE_RE.sub("", stripped)).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if 0 <= start < end:
            return json.loads(stripped[start : end + 1])
        raise


def _parse_explanations(payload: object, valid_ids: set[int]) -> dict[int, str]:
    """Pull ``{id: sentence}`` out of the reply, keeping only ids we asked about."""
    if not isinstance(payload, Mapping):
        raise ValueError("explanations reply is not a JSON object")
    out: dict[int, str] = {}
    for key, value in payload.items():
        try:
            movie_id = int(key)
        except (TypeError, ValueError):
            continue
        if movie_id not in valid_ids or not isinstance(value, str):
            continue
        sentence = " ".join(value.split())
        if sentence:
            out[movie_id] = sentence
    return out


def explain(
    client: OpenRouterClient,
    picks: Sequence[PickEvidence],
    models: Sequence[str],
    *,
    temperature: float = 0.3,
    max_attempts: int = _MAX_ATTEMPTS,
) -> dict[int, Explanation]:
    """Phrase every pick in one LLM call, falling back to the template for any it fumbles.

    Never raises: a malformed reply is retried once with the error appended, and a second failure
    (or an unreachable model) leaves every pick on its deterministic template. Any pick the model
    skips or returns empty also keeps its template, so the result always covers every pick.
    """
    if not picks:
        return {}
    result = templated_explanations(picks)  # start with the grounded fallback for every pick
    valid_ids = {p.movie_id for p in picks}

    system = load_prompt(_PROMPT_NAME)
    user = "Picks:\n" + "\n".join(_pick_line(p) for p in picks)
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for _attempt in range(max_attempts):
        try:
            reply = client.chat_with_failover(
                conversation, models, temperature=temperature, max_tokens=_MAX_TOKENS
            )
        except LLMError:
            break
        try:
            phrased = _parse_explanations(_loads(reply.text), valid_ids)
        except (ValueError, json.JSONDecodeError) as error:
            nudge = _RETRY_NUDGE.format(error=client.redact(str(error))[:200])
            conversation = [
                *conversation,
                {"role": "assistant", "content": reply.text[:1000]},
                {"role": "user", "content": nudge},
            ]
            continue
        for movie_id, sentence in phrased.items():
            result[movie_id] = Explanation(movie_id, sentence, templated=False)
        return result

    return result
