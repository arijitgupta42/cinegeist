"""Unit tests for the seed questions and turning resolved mentions into events."""

from __future__ import annotations

from cinegeist.convo import seed
from cinegeist.convo.resolve import Mention, MovieMatch, Resolution


def _resolved(movie_id: int, sentiment: str, evidence: str) -> tuple[Mention, Resolution]:
    mention = Mention(title="X", sentiment=sentiment, evidence=evidence)
    resolution = Resolution(
        query="X",
        year=None,
        status="resolved",
        match=MovieMatch(movie_id=movie_id, title="X", year=2000, score=1.0),
    )
    return mention, resolution


def test_there_are_three_seed_questions_in_order() -> None:
    assert [q.key for q in seed.SEED_QUESTIONS] == ["loved", "disliked", "pair"]
    assert [q.style for q in seed.SEED_QUESTIONS] == ["free_text", "free_text", "pair"]


def test_sentiment_value_maps_and_defaults() -> None:
    assert seed.sentiment_value("loved") == 1.0
    assert seed.sentiment_value("HATED") == -1.0  # case-insensitive
    assert seed.sentiment_value("mixed") == 0.0
    assert seed.sentiment_value("who knows") == 0.0  # unknown → no signal
    assert seed.sentiment_value(None) == 0.0


def test_events_are_built_from_resolved_directional_mentions() -> None:
    events = seed.events_from_resolutions(
        [
            _resolved(10, "loved", "loved this"),
            _resolved(20, "hated", "could not stand it"),
        ],
        session_id="s1",
    )
    assert len(events) == 2
    liked, disliked = events
    assert liked.kind == "liked_movie"
    assert liked.movie_id == 10
    assert liked.value == 1.0
    assert liked.evidence == "loved this"
    assert liked.session_id == "s1"
    assert disliked.kind == "disliked_movie"
    assert disliked.value == -1.0


def test_mixed_and_unresolved_mentions_produce_no_event() -> None:
    ambiguous = (
        Mention(title="Solaris", sentiment="loved"),
        Resolution(query="Solaris", year=None, status="ambiguous"),
    )
    mixed = _resolved(30, "mixed", "it was fine")
    events = seed.events_from_resolutions([ambiguous, mixed])
    assert events == []
