"""The fixed opening of every conversation, and how its answers become events.

The seed is three questions, always the same (plan.md §5.1): name a few films you loved, name
something you bounced off, then one this-or-that pair. Question two is the most valuable in the
system — a film someone turned off tells you more about their pacing tolerance than ten films
they loved — so it is asked of everyone, every time.

Questions one and two are free text: the LLM extracts the titles (session 3, PR 3), those titles
are resolved against the catalog (:mod:`cinegeist.convo.resolve`), and the resolved films become
``liked_movie`` / ``disliked_movie`` events here. The pair question's two films are chosen by the
probe layer (session 3, PR 4); this module only declares its place in the opening.
"""

from __future__ import annotations

from ..profile.model import PreferenceEvent
from .resolve import Mention, Resolution


class SeedQuestion:
    """One fixed opening question. ``style`` is 'free_text' (Q1, Q2) or 'pair' (Q3)."""

    __slots__ = ("key", "style", "prompt")

    def __init__(self, key: str, style: str, prompt: str) -> None:
        self.key = key
        self.style = style
        self.prompt = prompt


# The three seed questions, in order. The phrasings are fixed so the opening is reproducible and
# works in --offline mode; the LLM only rephrases later, adaptive questions.
SEED_QUESTIONS: tuple[SeedQuestion, ...] = (
    SeedQuestion(
        "loved",
        "free_text",
        "Name two or three films you've genuinely loved — any era, any genre.",
    ),
    SeedQuestion(
        "disliked",
        "free_text",
        "Anything you started and turned off, or that everyone loves and you just didn't?",
    ),
    SeedQuestion(
        "pair",
        "pair",
        "Two to get us going — which would you rather put on tonight?",
    ),
)

# The sentiment vocabulary the extractor emits, and the signed value each maps to. Kept here as
# the single source of truth so PR 3's extraction validates against exactly these keys. 'mixed'
# is deliberately zero — a genuinely ambivalent reaction is not evidence in either direction.
SENTIMENT_VALUE: dict[str, float] = {
    "loved": 1.0,
    "liked": 0.6,
    "mixed": 0.0,
    "disliked": -0.6,
    "bounced": -0.8,  # started and turned off — a sharp negative, usually about pacing
    "hated": -1.0,
}


def sentiment_value(sentiment: str | None) -> float:
    """The signed weight for a sentiment word, or 0.0 if it is unknown or ambivalent."""
    if sentiment is None:
        return 0.0
    return SENTIMENT_VALUE.get(sentiment.strip().lower(), 0.0)


def events_from_resolutions(
    resolutions: list[tuple[Mention, Resolution]],
    *,
    session_id: str | None = None,
    weight: float = 1.0,
) -> list[PreferenceEvent]:
    """Turn resolved seed mentions into like/dislike events, carrying the user's own words.

    Only cleanly resolved films with a directional sentiment become events. Ambiguous or unmatched
    titles, and ambivalent ("mixed") reactions, are skipped — the caller asks the user to
    disambiguate the first and treats the rest as no signal.
    """
    events: list[PreferenceEvent] = []
    for mention, resolution in resolutions:
        if not resolution.is_resolved or resolution.match is None:
            continue
        value = sentiment_value(mention.sentiment)
        if value == 0.0:
            continue
        kind = "liked_movie" if value > 0 else "disliked_movie"
        events.append(
            PreferenceEvent(
                kind=kind,
                subject_kind="movie",
                subject=str(resolution.match.movie_id),
                value=value,
                weight=weight,
                evidence=mention.evidence,
                session_id=session_id,
            )
        )
    return events
