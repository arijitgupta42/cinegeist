"""Unit tests for the taste-over-time reconstruction (profile/history.py).

Reuses the hand-built three-tag catalog shape from the update tests — Toy Story (animation) and
Solaris (cerebral/space) — so a viewer who likes one then the other has a taste that visibly
*moves*, which is what the timeline exists to show.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.profile import history, store, update
from cinegeist.profile.model import PreferenceEvent

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
TOY_STORY = np.array([0.9, 0.1, 0.2], dtype=np.float32)
SOLARIS = np.array([0.05, 0.95, 0.8], dtype=np.float32)


@pytest.fixture
def catalog() -> tuple[sqlite3.Connection, np.ndarray]:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(10, 0, "animation"), (20, 1, "cerebral"), (30, 2, "space")],
    )
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "Toy Story (1995)", "Toy Story", 1995, 0, "measured"),
            (2, "Solaris (1972)", "Solaris", 1972, 1, "measured"),
        ],
    )
    conn.commit()
    return conn, np.vstack([TOY_STORY, SOLARIS])


def _seed_two_sessions(conn: sqlite3.Connection) -> None:
    """Session one likes animation; session two, weeks later, swings to cerebral space."""
    store.append_event(
        conn,
        PreferenceEvent.liked_movie(1, evidence="loved this cartoon", session_id="s1"),
        now=T0,
    )
    store.append_event(
        conn,
        PreferenceEvent.liked_movie(2, evidence="loved this cerebral one", session_id="s2"),
        now=T0 + timedelta(days=40),
    )


# -- profile_from_events (the read-only historical primitive) ------------------------


def test_profile_from_events_matches_compute_over_the_full_log(catalog) -> None:
    conn, matrix = catalog
    _seed_two_sessions(conn)
    now = T0 + timedelta(days=60)
    events = store.iter_events(conn)
    subset = update.profile_from_events(conn, matrix, events, now=now)
    full = update.compute_profile(conn, matrix, now=now)
    assert np.allclose(subset.genome_vector, full.genome_vector, atol=1e-6)
    assert subset.total_weight == pytest.approx(full.total_weight)


def test_profile_from_events_does_not_write_a_snapshot(catalog) -> None:
    conn, matrix = catalog
    _seed_two_sessions(conn)
    # A historical recompute (past instant) must not clobber the live cache.
    update.profile_from_events(conn, matrix, store.iter_events(conn), now=T0)
    assert store.read_snapshot(conn) is None


# -- the timeline --------------------------------------------------------------------


def test_empty_log_is_an_empty_timeline(catalog) -> None:
    conn, matrix = catalog
    timeline = history.build_timeline(conn, matrix, now=T0)
    assert timeline.is_empty
    assert timeline.points == ()


def test_one_point_per_session_in_time_order(catalog) -> None:
    conn, matrix = catalog
    _seed_two_sessions(conn)
    timeline = history.build_timeline(conn, matrix, now=T0 + timedelta(days=60))
    assert [p.session_id for p in timeline.points] == ["s1", "s2"]
    assert [p.event_count for p in timeline.points] == [1, 2]  # cumulative
    assert timeline.points[0].when < timeline.points[1].when


def test_a_past_point_excludes_later_events(catalog) -> None:
    # The core correctness property: the profile at the first session is built from only that
    # session's evidence, not the whole log decayed backwards (which would fold in future likes).
    conn, matrix = catalog
    _seed_two_sessions(conn)
    timeline = history.build_timeline(conn, matrix, now=T0 + timedelta(days=60))
    first = timeline.points[0]
    assert first.event_count == 1
    # One like of Toy Story at age zero → mass 1.0 exactly, nothing from the later Solaris like.
    assert first.total_weight == pytest.approx(1.0)


def test_drift_is_larger_when_taste_actually_moves(catalog) -> None:
    conn, matrix = catalog
    # s1 animation, s2 cerebral (a real swing), s3 cerebral again (settling).
    store.append_event(conn, PreferenceEvent.liked_movie(1, session_id="s1"), now=T0)
    store.append_event(
        conn, PreferenceEvent.liked_movie(2, session_id="s2"), now=T0 + timedelta(days=30)
    )
    store.append_event(
        conn, PreferenceEvent.liked_movie(2, session_id="s3"), now=T0 + timedelta(days=60)
    )
    timeline = history.build_timeline(conn, matrix, now=T0 + timedelta(days=90))
    assert timeline.points[0].drift is None  # nothing to compare the first point against
    swing = timeline.points[1].drift
    settle = timeline.points[2].drift
    assert swing is not None and settle is not None
    assert swing > settle  # moving toward Solaris is a bigger step than staying there


def test_axis_tracks_follow_the_current_top_axes(catalog) -> None:
    conn, matrix = catalog
    _seed_two_sessions(conn)
    timeline = history.build_timeline(conn, matrix, now=T0 + timedelta(days=60))
    names = {track.name for track in timeline.axis_tracks}
    # By now cerebral/space dominate (the recent Solaris like), so they must be traced.
    assert "cerebral" in names or "space" in names
    for track in timeline.axis_tracks:
        assert len(track.weights) == len(timeline.points)


def test_older_evidence_has_faded_more(catalog) -> None:
    conn, matrix = catalog
    _seed_two_sessions(conn)
    now = T0 + timedelta(days=60)
    timeline = history.build_timeline(conn, matrix, now=now)
    by_note = {f.event.evidence: f for f in timeline.fading}
    old = by_note["loved this cartoon"]  # 60 days old
    recent = by_note["loved this cerebral one"]  # 20 days old
    assert old.decay < recent.decay
    assert all(0.0 < f.decay <= 1.0 for f in timeline.fading)
    assert old.age_days == pytest.approx(60.0, abs=0.01)
