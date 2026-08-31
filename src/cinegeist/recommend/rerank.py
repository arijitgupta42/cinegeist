"""Let the model put the shortlist in order — but never let it invent a film (plan.md §6, step 4).

The deterministic scorer hands up a diversified shortlist of real catalog ids (:mod:`score`). The
LLM's whole job here is to *reorder* that list using the user's taste evidence: it reads the
candidates and the profile, and returns ids, best first. It phrases nothing and names nothing.

This module is where CLAUDE.md hard rule 1 is enforced. Whatever the model returns, we keep only
ids that were in the shortlist we sent — a hallucinated id (or a title, or a number it made up) is
dropped, with a log line, and never reaches the user. Ids the model omitted are appended in their
original order, so the shortlist comes back reordered but whole: same films, new sequence.

Budget is one call (hard rule 4), defended like every other LLM touch (CLAUDE.md testing): the
reply is parsed leniently, retried once with the error appended if it is unusable, and on a second
failure — or an unreachable model — we fall back to the deterministic order rather than crash. A
bad model reply costs us the reranking, not the recommendation.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..llm.client import LLMError, OpenRouterClient
from ..llm.prompt_loader import load_prompt
from .score import ScoredFilm

logger = logging.getLogger(__name__)

_PROMPT_NAME = "rerank"
_MAX_ATTEMPTS = 2  # one try, then one retry with the error appended
_MAX_TOKENS = 512  # a list of ids is small; cap so a rambling model can't burn the budget
_RETRY_NUDGE = (
    "Your previous reply was not usable ({error}). Reply with ONLY a JSON object like "
    '{{"order": [12, 88, 47]}}, using only ids from the candidate list.'
)

_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?")
_TRAILING_FENCE_RE = re.compile(r"\n?```$")


@dataclass(frozen=True)
class RerankOutcome:
    """The reordered shortlist plus what happened, so the engine can log and explain honestly.

    ``ordered`` is the full input shortlist in the new sequence — every film present exactly once,
    nothing added. ``degraded`` is true when we fell back to the deterministic order (the model was
    unusable or unreachable). ``dropped_ids`` are ids the model returned that weren't ours — the
    hallucinations hard rule 1 exists to catch.
    """

    ordered: list[ScoredFilm]
    model: str | None
    degraded: bool
    dropped_ids: tuple[int, ...] = ()


def _candidate_line(film: ScoredFilm, tags: Sequence[str] | None) -> str:
    year = f" ({film.year})" if film.year else ""
    line = f'- id {film.movie_id} — "{film.title}"{year}'
    if tags:
        line += f" — tags: {', '.join(tags)}"
    return line


def _build_messages(
    shortlist: list[ScoredFilm],
    profile_summary: str,
    tags_by_id: Mapping[int, Sequence[str]] | None,
) -> list[dict[str, str]]:
    lines = [
        _candidate_line(f, tags_by_id.get(f.movie_id) if tags_by_id else None) for f in shortlist
    ]
    evidence = profile_summary.strip() or "No strong taste on record yet — order by overall fit."
    user = f"Taste evidence:\n{evidence}\n\nCandidates:\n" + "\n".join(lines)
    return [
        {"role": "system", "content": load_prompt(_PROMPT_NAME)},
        {"role": "user", "content": user},
    ]


def _loads(text: str) -> object:
    """Parse a model reply into JSON, tolerating code fences and surrounding prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _TRAILING_FENCE_RE.sub("", _LEADING_FENCE_RE.sub("", stripped)).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Fall back to the outermost object, then the outermost array, whichever is present.
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start, end = stripped.find(open_ch), stripped.rfind(close_ch)
            if 0 <= start < end:
                return json.loads(stripped[start : end + 1])  # may still raise — caught by caller
        raise


def _parse_ids(payload: object) -> list[int]:
    """Pull the ordered id list out of ``{"order": [...]}`` or a bare ``[...]``.

    Ids may arrive as ints or as numeric strings; anything non-numeric is skipped here and the
    shortlist validation drops it regardless. Raises ``ValueError`` when there is no list at all,
    so the caller can retry a structurally broken reply.
    """
    if isinstance(payload, Mapping):
        payload = payload.get("order", payload.get("ids"))
    if not isinstance(payload, list):
        raise ValueError("no id list in reply")
    ids: list[int] = []
    for item in payload:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def apply_order(
    shortlist: list[ScoredFilm], model_ids: Sequence[int]
) -> tuple[list[ScoredFilm], list[int]]:
    """Reorder ``shortlist`` by ``model_ids``, keeping only our ids and appending the rest.

    The heart of hard rule 1, and pure so it is trivially testable: an id the model returned that
    is not in the shortlist is collected as ``dropped`` and never appears in the output; a film the
    model didn't mention keeps its place, after the ones it ranked. Every input film appears in the
    output exactly once.
    """
    by_id = {f.movie_id: f for f in shortlist}
    ordered: list[ScoredFilm] = []
    placed: set[int] = set()
    dropped: list[int] = []
    for movie_id in model_ids:
        if movie_id in placed:
            continue  # a repeat; ignore the second mention
        film = by_id.get(movie_id)
        if film is None:
            dropped.append(movie_id)  # not one of ours — a hallucination, hard rule 1
            continue
        ordered.append(film)
        placed.add(movie_id)
    for film in shortlist:  # whatever the model left out, in its original order
        if film.movie_id not in placed:
            ordered.append(film)
    return ordered, dropped


def rerank(
    client: OpenRouterClient,
    shortlist: list[ScoredFilm],
    models: Sequence[str],
    *,
    profile_summary: str = "",
    tags_by_id: Mapping[int, Sequence[str]] | None = None,
    temperature: float = 0.0,
    max_attempts: int = _MAX_ATTEMPTS,
) -> RerankOutcome:
    """Reorder ``shortlist`` with one LLM call, validated against the ids we sent.

    Never raises for a bad model reply: a malformed answer is retried once with the error appended,
    and a second failure (or an unreachable model) returns the shortlist in its deterministic order
    with ``degraded=True``. Hallucinated ids are dropped and logged.
    """
    if not shortlist:
        return RerankOutcome(ordered=[], model=None, degraded=False)
    if len(shortlist) == 1:
        return RerankOutcome(ordered=list(shortlist), model=None, degraded=False)

    conversation = _build_messages(shortlist, profile_summary, tags_by_id)
    for _attempt in range(max_attempts):
        try:
            result = client.chat_with_failover(
                conversation, models, temperature=temperature, max_tokens=_MAX_TOKENS
            )
        except LLMError:
            break  # auth failure or every model down — nothing to recover this turn
        try:
            model_ids = _parse_ids(_loads(result.text))
        except (ValueError, json.JSONDecodeError) as error:
            nudge = _RETRY_NUDGE.format(error=client.redact(str(error))[:200])
            conversation = [
                *conversation,
                {"role": "assistant", "content": result.text[:1000]},
                {"role": "user", "content": nudge},
            ]
            continue
        ordered, dropped = apply_order(shortlist, model_ids)
        if dropped:
            logger.warning(
                "rerank dropped %d id(s) not in the shortlist: %s", len(dropped), dropped
            )
        return RerankOutcome(
            ordered=ordered,
            model=result.model,
            degraded=False,
            dropped_ids=tuple(dropped),
        )

    return RerankOutcome(ordered=list(shortlist), model=None, degraded=True)
