"""Learn from what the user did after we recommended something (plan.md §5.1, FEEDBACK).

The payoff of a recommendation is the reaction to it, and a reaction to a film we picked is the
sharpest evidence there is — it's revealed, not stated. So after the picks are shown the user can
say what happened, and each verdict becomes one more ``post_watch_feedback`` event in the same
append-only log everything else lives in. The profile is a derived view, so the next run simply
recomputes over the longer log and comes out visibly different: a "not for me" pushes the centroid
away from that film, and the film never comes back (it's now a seen id, so retrieval excludes it).

This is deterministic and LLM-free by design. The verdicts are a small fixed vocabulary the user
picks from — a click, or a phrase we match — so a feedback turn costs no model call and works
offline. "Show me three more" isn't a verdict at all; it's a request for the next page of picks,
recognised here but handled by the engine, which just re-presents with the shown films excluded.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from .profile import store
from .profile.model import PreferenceEvent


@dataclass(frozen=True)
class Verdict:
    """One post-watch reaction: what it's called, and the signed evidence it records.

    ``value`` is the direction and strength in ``[-1, 1]``; ``weight`` is the initial confidence,
    set above 1.0 for the decisive verdicts because actually watching a film is stronger evidence
    than a hypothetical this-or-that click. ``already_seen`` carries a zero value: it adds no taste
    direction, it just marks the film seen so we stop recommending it.
    """

    key: str
    label: str
    value: float
    weight: float


# The feedback menu, in the order the plan lists it (plan.md §5.1). Shown as choices in the CLI;
# also matched from typed text by :func:`match_feedback`.
VERDICTS: tuple[Verdict, ...] = (
    Verdict("loved", "watched and loved it", value=1.0, weight=1.5),
    Verdict("fine", "watched, it was fine", value=0.2, weight=1.0),
    Verdict("not_for_me", "not for me", value=-0.8, weight=1.5),
    Verdict("already_seen", "already seen it", value=0.0, weight=1.0),
)

_BY_KEY: dict[str, Verdict] = {v.key: v for v in VERDICTS}

# Phrase fragments that identify each verdict in free text, checked most-specific first so a
# negation ("not for me") is never misread as its opposite ("for me" / "loved"). Each entry is
# (verdict key, patterns); order within the tuple is the match priority.
_FEEDBACK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("already_seen", ("already seen", "seen it", "seen this", "already watched")),
    (
        "not_for_me",
        (
            "not for me",
            "wasn't for me",
            "was not for me",
            "didn't like",
            "did not like",
            "hated",
            "turned it off",
            "turned off",
            "not really",
            "nope",
        ),
    ),
    (
        "loved",
        ("loved it", "loved", "adored", "amazing", "brilliant", "really liked", "great pick"),
    ),
    (
        "fine",
        ("it was fine", "was fine", "fine", "it was ok", "was ok", "okay", "alright", "decent"),
    ),
)

# "Give me different ones" — a request for the next page of picks, not a taste verdict.
_MORE_PATTERNS: tuple[str, ...] = (
    "three more",
    "show me more",
    "show me another",
    "some more",
    "more options",
    "more picks",
    "anything else",
    "something else",
    "what else",
    "next",
    "others",
)


def get_verdict(key: str) -> Verdict:
    """The :class:`Verdict` for a key, or ``KeyError`` if it isn't one of the four."""
    return _BY_KEY[key]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def match_feedback(text: str) -> Verdict | None:
    """Read a typed reaction as one of the verdicts, or ``None`` when it isn't clearly one.

    Most-specific patterns are tried first so "not for me" resolves to the negative verdict rather
    than tripping the "for me"-adjacent positives. Returning ``None`` lets the caller treat the
    input as something else (a "show me more", or a fresh free-text answer) instead of guessing.
    """
    lowered = _normalize(text)
    if not lowered:
        return None
    for key, patterns in _FEEDBACK_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return _BY_KEY[key]
    return None


def wants_more(text: str) -> bool:
    """True when the user is asking for the next page of picks rather than giving feedback."""
    lowered = _normalize(text)
    return any(pattern in lowered for pattern in _MORE_PATTERNS)


def record_feedback(
    conn: sqlite3.Connection,
    movie_id: int,
    verdict: Verdict | str,
    *,
    session_id: str | None = None,
    evidence: str | None = None,
    now: datetime | None = None,
) -> PreferenceEvent:
    """Append a post-watch reaction to a film as one immutable event, and return it.

    Accepts a :class:`Verdict` or its key. The event's ``value``/``weight`` come from the verdict;
    ``evidence`` is the user's own words when they gave any. Appending invalidates the cached
    snapshot, so the next profile read reflects this reaction — that is the whole learning loop.
    """
    if isinstance(verdict, str):
        verdict = _BY_KEY[verdict]
    event = PreferenceEvent(
        kind="post_watch_feedback",
        subject_kind="movie",
        subject=str(movie_id),
        value=verdict.value,
        weight=verdict.weight,
        evidence=evidence,
        session_id=session_id,
    )
    return store.append_event(conn, event, now=now)
