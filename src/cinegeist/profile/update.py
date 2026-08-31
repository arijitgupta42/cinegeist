"""Turn the event log into a decayed taste profile.

The profile is a weighted centroid in tag-genome space (plan.md §4.2):

    w_i    = value_i × weight_i × 0.5 ** (age_days_i / HALF_LIFE)
    vector = Σ (w_i × genome_i) / Σ |w_i|

where ``genome_i`` is the film's genome row for a movie event, or a one-hot on the answered
axis for a tag event. Constraints (facets) are session filters, not directions in taste space,
so they never enter the centroid. Old evidence fades rather than being deleted, which gives
taste drift for free and lets a returning user's quiet-but-recoverable history come back.

A useful consequence of the ratio-of-sums form: uniform time decay scales every term by the
same factor, which cancels top and bottom, so the centroid's *direction* does not move as time
passes on its own — only new or forgotten evidence moves it. Only ``total_weight`` (the evidence
mass, our confidence signal) shrinks with time. That is what makes the cached snapshot valid
until the event set changes, and it is why we can decay a cached ``total_weight`` forward with a
single scalar instead of replaying the log.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np

from . import store
from .model import PreferenceEvent, TagAffinity, TasteProfile

# Default half-life for the decay: evidence loses half its weight every 270 days.
HALF_LIFE_DAYS = 270.0

# How many axes per sign to keep (with attributed evidence) for display and explanation.
_AXES_PER_SIGN = 15
# Below this magnitude an axis is treated as noise rather than a real affinity.
_AXIS_EPSILON = 1e-6


def decay_factor(age_days: float, half_life: float = HALF_LIFE_DAYS) -> float:
    """The multiplier ``0.5 ** (age_days / half_life)`` applied to an event's weight.

    1.0 at age zero, 0.5 at one half-life, and never above 1.0 (a future timestamp from clock
    skew is clamped to the present so it can't amplify).
    """
    return 0.5 ** (max(0.0, age_days) / half_life)


def _age_days(then: datetime, now: datetime) -> float:
    then = then if then.tzinfo is not None else then.replace(tzinfo=UTC)
    now = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return (now - then).total_seconds() / 86_400.0


# One event's contribution to the centroid: its signed decayed weight, the unit/genome vector it
# acts on, the event itself (for evidence), and a short label of where it came from.
_Contribution = tuple[float, np.ndarray, PreferenceEvent, str]


def _resolve_movie_vectors(
    conn: sqlite3.Connection, matrix: np.ndarray, movie_ids: set[int]
) -> dict[int, tuple[np.ndarray, str]]:
    """Map each known, genome-covered movie id to (its genome row, a display title)."""
    if not movie_ids:
        return {}
    placeholders = ",".join("?" for _ in movie_ids)
    rows = conn.execute(
        f"SELECT movie_id, genome_row, clean_title, title FROM movies "
        f"WHERE movie_id IN ({placeholders})",
        tuple(movie_ids),
    ).fetchall()
    resolved: dict[int, tuple[np.ndarray, str]] = {}
    for row in rows:
        if row["genome_row"] is None:
            continue  # un-vectored film: still real evidence, but no direction to contribute
        vector = np.asarray(matrix[row["genome_row"]], dtype=np.float32)
        resolved[row["movie_id"]] = (vector, row["clean_title"] or row["title"])
    return resolved


def _resolve_tag_axes(conn: sqlite3.Connection, tag_ids: set[int]) -> dict[int, tuple[int, str]]:
    """Map each known genome tag id to (its column position, its name)."""
    if not tag_ids:
        return {}
    placeholders = ",".join("?" for _ in tag_ids)
    rows = conn.execute(
        f"SELECT tag_id, position, name FROM genome_tags WHERE tag_id IN ({placeholders})",
        tuple(tag_ids),
    ).fetchall()
    return {row["tag_id"]: (row["position"], row["name"]) for row in rows}


def _accumulate(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    events: list[PreferenceEvent],
    now: datetime,
    half_life: float,
) -> tuple[np.ndarray, float, list[_Contribution]]:
    """Fold the events into (centroid, total_weight, per-event contributions)."""
    n_tags = int(matrix.shape[1])
    movie_ids = {e.movie_id for e in events if e.subject_kind == "movie" and e.movie_id is not None}
    tag_ids = {e.tag_id for e in events if e.subject_kind == "tag" and e.tag_id is not None}
    movies = _resolve_movie_vectors(conn, matrix, movie_ids)
    tags = _resolve_tag_axes(conn, tag_ids)

    numerator = np.zeros(n_tags, dtype=np.float32)
    total_weight = 0.0
    contributions: list[_Contribution] = []
    for event in events:
        if event.subject_kind == "movie":
            resolved = movies.get(event.movie_id) if event.movie_id is not None else None
            if resolved is None:
                continue
            vector, source = resolved
        elif event.subject_kind == "tag":
            axis = tags.get(event.tag_id) if event.tag_id is not None else None
            if axis is None:
                continue
            position, name = axis
            vector = np.zeros(n_tags, dtype=np.float32)
            vector[position] = 1.0
            source = f"'{name}'"
        else:
            continue  # a facet/constraint is a filter, not a taste-space direction

        w = event.value * event.weight * decay_factor(_age_days(event.ts or now, now), half_life)
        if w == 0.0:
            continue
        numerator += w * vector
        total_weight += abs(w)
        contributions.append((w, vector, event, source))

    centroid = numerator / total_weight if total_weight > 0 else numerator
    return centroid, total_weight, contributions


def _best_evidence(contributions: list[_Contribution], position: int) -> tuple[str | None, str]:
    """The verbatim quote and source label of the event that most drove one axis."""
    best: tuple[str | None, str] | None = None
    best_magnitude = 0.0
    for weight, vector, event, source in contributions:
        magnitude = abs(weight * float(vector[position]))
        if magnitude > best_magnitude:
            best_magnitude = magnitude
            best = (event.evidence, source)
    return best if best is not None else (None, "")


def _rank_axes(
    conn: sqlite3.Connection, centroid: np.ndarray, contributions: list[_Contribution]
) -> tuple[TagAffinity, ...]:
    """The strongest signed axes, each with the evidence that most produced it."""
    if not contributions:
        return ()
    names = {
        row["position"]: row["name"]
        for row in conn.execute("SELECT position, name FROM genome_tags")
    }
    order = np.argsort(np.abs(centroid))[::-1]
    positives = 0
    negatives = 0
    chosen: list[TagAffinity] = []
    for position in order:
        weight = float(centroid[position])
        if abs(weight) < _AXIS_EPSILON:
            break  # everything past here is noise
        if weight > 0 and positives >= _AXES_PER_SIGN:
            continue
        if weight < 0 and negatives >= _AXES_PER_SIGN:
            continue
        evidence, source = _best_evidence(contributions, int(position))
        chosen.append(
            TagAffinity(
                position=int(position),
                name=names.get(int(position), f"tag#{position}"),
                weight=weight,
                source=source,
                evidence=evidence,
            )
        )
        if weight > 0:
            positives += 1
        else:
            negatives += 1
        if positives >= _AXES_PER_SIGN and negatives >= _AXES_PER_SIGN:
            break
    chosen.sort(key=lambda a: a.weight, reverse=True)
    return tuple(chosen)


def compute_profile(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    *,
    user_id: str = store.DEFAULT_USER,
    now: datetime | None = None,
    half_life: float = HALF_LIFE_DAYS,
) -> TasteProfile:
    """Recompute a user's profile from their whole event log and refresh the cached snapshot.

    This is the full path used by ``profile show`` and explanations: it returns the decayed
    centroid *and* the ranked axes with evidence. The lighter :func:`load_vector` is for the
    hot recommendation path, where only the vector is needed.
    """
    now = now or store.now_utc()
    events = store.iter_events(conn, user_id)
    event_count = len(events)
    session_count = store.count_sessions(conn, user_id)

    centroid, total_weight, contributions = _accumulate(conn, matrix, events, now, half_life)
    store.write_snapshot(
        conn,
        user_id,
        computed_at=now,
        event_count=event_count,
        total_weight=total_weight,
        vector=centroid,
    )
    axes = _rank_axes(conn, centroid, contributions)
    return TasteProfile(
        user_id=user_id,
        genome_vector=centroid,
        total_weight=total_weight,
        event_count=event_count,
        session_count=session_count,
        computed_at=now,
        axes=axes,
    )


def load_vector(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    *,
    user_id: str = store.DEFAULT_USER,
    now: datetime | None = None,
    half_life: float = HALF_LIFE_DAYS,
) -> tuple[np.ndarray, float]:
    """Return ``(centroid, total_weight)``, reusing the snapshot when it is still valid.

    The snapshot is invalidated on every write, so an existing one whose ``event_count`` still
    matches the log is exact: the centroid is reused as-is and ``total_weight`` is decayed
    forward to ``now`` by a single scalar. Otherwise the profile is recomputed (and re-cached).
    """
    now = now or store.now_utc()
    snapshot = store.read_snapshot(conn, user_id)
    if (
        snapshot is not None
        and snapshot.event_count == store.count_events(conn, user_id)
        and snapshot.vector.shape[0] == int(matrix.shape[1])
    ):
        factor = decay_factor(_age_days(snapshot.computed_at, now), half_life)
        return snapshot.vector.copy(), snapshot.total_weight * factor
    profile = compute_profile(conn, matrix, user_id=user_id, now=now, half_life=half_life)
    return profile.genome_vector, profile.total_weight
