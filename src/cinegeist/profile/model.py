"""Dataclasses for the taste profile: the immutable event and the derived view.

A :class:`PreferenceEvent` is one row of the append-only log — a single reaction, carrying the
user's own words. A :class:`TasteProfile` is what the recommender reads: the decayed centroid
in tag-genome space plus, for display and explanation, the strongest signed axes with the piece
of evidence that most produced each. The maths that turns the first into the second lives in
:mod:`cinegeist.profile.update`; this module is just the shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# The event vocabularies, mirroring the CHECK constraints in migration_0003_profile.sql. Kept
# here too so construction and tests share one source of truth rather than hard-coding strings.
EVENT_KINDS: frozenset[str] = frozenset(
    {
        "liked_movie",
        "disliked_movie",
        "pair_choice",
        "axis_answer",
        "constraint",
        "post_watch_feedback",
    }
)
SUBJECT_KINDS: frozenset[str] = frozenset({"movie", "tag", "facet"})


@dataclass(frozen=True)
class PreferenceEvent:
    """One immutable reaction in the log.

    ``subject`` is polymorphic and read according to ``subject_kind``: a ``movies.movie_id``
    for ``'movie'``, a ``genome_tags.tag_id`` for ``'tag'``, or a facet key for ``'facet'``.
    ``value`` is a signed strength in ``[-1, 1]`` for taste events; a ``constraint`` overloads
    it to carry the facet's raw value (e.g. a runtime ceiling in minutes). ``ts`` is ``None``
    until the event is persisted, at which point the store stamps it.
    """

    kind: str
    subject_kind: str
    subject: str
    value: float
    weight: float = 1.0
    evidence: str | None = None
    session_id: str | None = None
    user_id: str = "default"
    ts: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")
        if self.subject_kind not in SUBJECT_KINDS:
            raise ValueError(f"unknown subject_kind {self.subject_kind!r}")

    # -- construction helpers ---------------------------------------------------------
    #
    # The conversation builds these; centralising the subject_kind/value conventions here keeps
    # callers from re-deriving "a like is value +1 on a movie" every time.

    @classmethod
    def liked_movie(
        cls,
        movie_id: int,
        *,
        value: float = 1.0,
        weight: float = 1.0,
        evidence: str | None = None,
        session_id: str | None = None,
    ) -> PreferenceEvent:
        """A film the user liked. Positive pull toward that film's genome vector."""
        return cls(
            kind="liked_movie",
            subject_kind="movie",
            subject=str(movie_id),
            value=value,
            weight=weight,
            evidence=evidence,
            session_id=session_id,
        )

    @classmethod
    def disliked_movie(
        cls,
        movie_id: int,
        *,
        value: float = -1.0,
        weight: float = 1.0,
        evidence: str | None = None,
        session_id: str | None = None,
    ) -> PreferenceEvent:
        """A film the user disliked. Negative push away from that film's genome vector."""
        return cls(
            kind="disliked_movie",
            subject_kind="movie",
            subject=str(movie_id),
            value=value,
            weight=weight,
            evidence=evidence,
            session_id=session_id,
        )

    @classmethod
    def axis_answer(
        cls,
        tag_id: int,
        value: float,
        *,
        weight: float = 1.0,
        evidence: str | None = None,
        session_id: str | None = None,
    ) -> PreferenceEvent:
        """A direct answer about one taste axis (a genome tag), signed in ``[-1, 1]``."""
        return cls(
            kind="axis_answer",
            subject_kind="tag",
            subject=str(tag_id),
            value=value,
            weight=weight,
            evidence=evidence,
            session_id=session_id,
        )

    @property
    def movie_id(self) -> int | None:
        """The subject as a movie id, or ``None`` when this event is not about a movie."""
        return int(self.subject) if self.subject_kind == "movie" else None

    @property
    def tag_id(self) -> int | None:
        """The subject as a genome tag id, or ``None`` when this event is not about a tag."""
        return int(self.subject) if self.subject_kind == "tag" else None


@dataclass(frozen=True)
class TagAffinity:
    """One taste axis in the derived profile: a signed strength plus where it came from.

    ``weight`` is this axis's value in the decayed centroid (positive = drawn toward, negative =
    pushed away). ``evidence`` is the single verbatim quote that contributed most to it, when the
    driving event had words; ``source`` is a short provenance label (a film title or the axis
    itself) shown when there is no quote.
    """

    position: int
    name: str
    weight: float
    source: str
    evidence: str | None = None


@dataclass(frozen=True)
class TasteProfile:
    """The decayed taste vector plus the display-ready axes derived from the event log.

    ``genome_vector`` is the weighted centroid over the tag genome — the thing the recommender
    scores against. ``total_weight`` is Σ|w_i|, the evidence mass behind it (our confidence
    signal), already decayed to the ``computed_at`` instant. ``axes`` holds the strongest signed
    axes, most positive first through to most negative, each with its best evidence.
    """

    user_id: str
    genome_vector: np.ndarray
    total_weight: float
    event_count: int
    session_count: int
    computed_at: datetime
    axes: tuple[TagAffinity, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        """True when no evidence has shaped this profile yet."""
        return self.event_count == 0 or self.total_weight == 0.0

    @property
    def affinities(self) -> list[TagAffinity]:
        """Axes the user is drawn toward, strongest first."""
        return [axis for axis in self.axes if axis.weight > 0]

    @property
    def aversions(self) -> list[TagAffinity]:
        """Axes the user is pushed away from, strongest (most negative) first."""
        return sorted((axis for axis in self.axes if axis.weight < 0), key=lambda a: a.weight)
