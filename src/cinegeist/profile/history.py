"""Reconstruct how a taste profile evolved over time, straight from the event log.

Every other view of the profile is a snapshot of *now*. This one is the trajectory: sample the
persisted profile at each past session and watch it move — the evidence mass rising, the strongest
axes climbing and slipping, the centroid drifting, and old evidence fading as it ages past the
270-day half-life. It reads only the append-only ``preference_events`` log and the decay function,
so it invents no new storage and stays exact: the profile at a past instant is just
:func:`~cinegeist.profile.update.profile_from_events` over the events up to then, decayed to then.

This is the full-version answer to "why are tonight's picks different from last month's?" — the
picks follow the centroid, and here you can see the centroid move. It is deliberately full-mode
only: the browser demo keeps nothing between sessions, so it has no honest history to draw.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..catalog import genome
from . import store, update
from .model import PreferenceEvent

# The current top axes we trace back through time. A handful keeps the sparkline table legible.
_TRACKED_AXES = 6
# How many aging events to list. The strongest evidence is what a reader recognises fading.
_FADING_SHOWN = 8
# Cap on sampled points so a heavy log stays a readable table; first and last are always kept.
_MAX_POINTS = 12


@dataclass(frozen=True)
class TimelinePoint:
    """The profile as it stood at one past moment (the end of one session)."""

    when: datetime
    session_id: str | None
    event_count: int  # cumulative events up to and including this point
    total_weight: float  # decayed evidence mass at `when`
    drift: float | None  # 1 − cosine(previous vector, this vector): how far taste moved since


@dataclass(frozen=True)
class AxisTrack:
    """One currently-strong axis and its weight at each sampled point, oldest → newest."""

    name: str
    weights: tuple[float, ...]


@dataclass(frozen=True)
class FadingEvidence:
    """One piece of evidence and how much time has worn it down by ``now``."""

    event: PreferenceEvent
    decay: float  # the current 0.5**(age/half_life) multiplier, in (0, 1]
    age_days: float


@dataclass(frozen=True)
class Timeline:
    """A profile's whole trajectory: sampled points, the top axes' tracks, and aging evidence."""

    user_id: str
    now: datetime
    points: tuple[TimelinePoint, ...]
    axis_tracks: tuple[AxisTrack, ...]
    fading: tuple[FadingEvidence, ...]

    @property
    def is_empty(self) -> bool:
        return not self.points


def _sample_times(events: list[PreferenceEvent]) -> list[tuple[datetime, str | None]]:
    """One (time, session_id) per distinct session, at that session's last event, oldest first.

    Sessions are the natural granularity of "a time you came back". Events with no session id are
    folded into one implicit bucket so a hand-seeded log still yields a point. If there are more
    sessions than we can show, keep an evenly-spaced subset that always includes the first and last.
    """
    last_by_session: dict[str | None, datetime] = {}
    for event in events:  # events arrive oldest-first, so the last write per session wins
        ts = event.ts
        if ts is None:
            continue
        last_by_session[event.session_id] = ts
    ordered = sorted(last_by_session.items(), key=lambda kv: kv[1])
    points = [(ts, sid) for sid, ts in ordered]
    if len(points) <= _MAX_POINTS:
        return points
    # Thin to _MAX_POINTS, keeping the endpoints: pick evenly-spaced indices across the range.
    step = (len(points) - 1) / (_MAX_POINTS - 1)
    idx = sorted({round(i * step) for i in range(_MAX_POINTS)})
    return [points[i] for i in idx]


def build_timeline(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    *,
    user_id: str = store.DEFAULT_USER,
    now: datetime | None = None,
    half_life: float = update.HALF_LIFE_DAYS,
) -> Timeline:
    """Reconstruct the profile's evolution across sessions from the event log."""
    now = now or store.now_utc()
    events = store.iter_events(conn, user_id)
    if not events:
        return Timeline(user_id=user_id, now=now, points=(), axis_tracks=(), fading=())

    samples = _sample_times(events)

    points: list[TimelinePoint] = []
    vectors: list[np.ndarray] = []
    prev_vector: np.ndarray | None = None
    for when, session_id in samples:
        upto = [e for e in events if e.ts is not None and e.ts <= when]
        profile = update.profile_from_events(
            conn, matrix, upto, user_id=user_id, now=when, half_life=half_life
        )
        vectors.append(profile.genome_vector)
        drift = None
        if prev_vector is not None:
            cos = float(genome.cosine_scores(profile.genome_vector[np.newaxis, :], prev_vector)[0])
            drift = max(0.0, 1.0 - cos)
        points.append(
            TimelinePoint(
                when=when,
                session_id=session_id,
                event_count=len(upto),
                total_weight=profile.total_weight,
                drift=drift,
            )
        )
        prev_vector = profile.genome_vector

    axis_tracks = _axis_tracks(conn, matrix, events, samples, vectors, now, half_life)
    fading = _fading_evidence(conn, events, now, half_life)
    return Timeline(
        user_id=user_id,
        now=now,
        points=tuple(points),
        axis_tracks=axis_tracks,
        fading=fading,
    )


def _axis_tracks(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    events: list[PreferenceEvent],
    samples: list[tuple[datetime, str | None]],
    vectors: list[np.ndarray],
    now: datetime,
    half_life: float,
) -> tuple[AxisTrack, ...]:
    """Trace today's strongest axes back through the sampled vectors, so you see them rise and fall.

    The axes are chosen from the profile as it stands *now* (the full log decayed to now), then read
    at each historical vector by column position — a fixed set of lines makes the movement legible.
    """
    current = update.profile_from_events(
        conn, matrix, events, user_id="", now=now, half_life=half_life
    )
    ranked = sorted(current.axes, key=lambda a: abs(a.weight), reverse=True)[:_TRACKED_AXES]
    ranked.sort(key=lambda a: a.weight, reverse=True)
    tracks: list[AxisTrack] = []
    for axis in ranked:
        weights = tuple(float(vector[axis.position]) for vector in vectors)
        tracks.append(AxisTrack(name=axis.name, weights=weights))
    return tuple(tracks)


def _fading_evidence(
    conn: sqlite3.Connection,
    events: list[PreferenceEvent],
    now: datetime,
    half_life: float,
) -> tuple[FadingEvidence, ...]:
    """The strongest evidence and how far it has decayed by now — the oldest has faded the most."""
    scored: list[FadingEvidence] = []
    for event in events:
        ts = event.ts or now
        age_days = max(0.0, (now - ts).total_seconds() / 86_400.0)
        decay = update.decay_factor(age_days, half_life)
        scored.append(FadingEvidence(event=event, decay=decay, age_days=age_days))
    # Show the events carrying the most *original* weight (|value×weight|), so the list is the
    # evidence a reader recognises, ranked by how much it still counts after decay.
    scored.sort(key=lambda f: abs(f.event.value * f.event.weight) * f.decay, reverse=True)
    return tuple(scored[:_FADING_SHOWN])
