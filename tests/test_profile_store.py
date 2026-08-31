"""Unit tests for the preference-event store: round-trip, forget, reset, snapshot cache.

Everything runs against a migrated in-memory database — no catalog, no network. The store does
not interpret events (that is update.py's job), so these tests only care that rows go in and
come back faithfully and that the snapshot cache is invalidated on every write.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.profile import store
from cinegeist.profile.model import PreferenceEvent


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = db.connect(":memory:")
    db.migrate(connection)
    return connection


AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


# -- round-trip ----------------------------------------------------------------------


def test_append_assigns_id_and_stamps_time(conn: sqlite3.Connection) -> None:
    stored = store.append_event(conn, PreferenceEvent.liked_movie(862, evidence="loved it"), now=AT)
    assert stored.id is not None
    assert stored.ts == AT
    back = store.get_event(conn, stored.id)
    assert back is not None
    assert back.kind == "liked_movie"
    assert back.subject_kind == "movie"
    assert back.movie_id == 862
    assert back.value == 1.0
    assert back.evidence == "loved it"


def test_append_keeps_an_explicit_timestamp(conn: sqlite3.Connection) -> None:
    when = datetime(2020, 6, 1, tzinfo=UTC)
    event = PreferenceEvent.liked_movie(1, session_id="s1")
    stored = store.append_event(conn, PreferenceEvent(**{**event.__dict__, "ts": when}), now=AT)
    assert store.get_event(conn, stored.id).ts == when


def test_iter_events_is_oldest_first(conn: sqlite3.Connection) -> None:
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=datetime(2022, 1, 1, tzinfo=UTC))
    store.append_event(conn, PreferenceEvent.liked_movie(2), now=datetime(2023, 1, 1, tzinfo=UTC))
    ids = [e.movie_id for e in store.iter_events(conn)]
    assert ids == [1, 2]


def test_counts_events_and_distinct_sessions(conn: sqlite3.Connection) -> None:
    store.append_events(
        conn,
        [
            PreferenceEvent.liked_movie(1, session_id="a"),
            PreferenceEvent.disliked_movie(2, session_id="a"),
            PreferenceEvent.liked_movie(3, session_id="b"),
            PreferenceEvent.liked_movie(4),  # no session id — not counted as a session
        ],
        now=AT,
    )
    assert store.count_events(conn) == 4
    assert store.count_sessions(conn) == 2


# -- forget and reset ----------------------------------------------------------------


def test_forget_removes_one_event(conn: sqlite3.Connection) -> None:
    a = store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    store.append_event(conn, PreferenceEvent.liked_movie(2), now=AT)
    assert store.forget_event(conn, a.id) is True
    assert store.get_event(conn, a.id) is None
    assert store.count_events(conn) == 1


def test_forget_missing_event_returns_false(conn: sqlite3.Connection) -> None:
    assert store.forget_event(conn, 999) is False


def test_reset_clears_the_user(conn: sqlite3.Connection) -> None:
    store.append_events(
        conn, [PreferenceEvent.liked_movie(1), PreferenceEvent.liked_movie(2)], now=AT
    )
    assert store.reset_profile(conn) == 2
    assert store.count_events(conn) == 0


def test_reset_only_touches_the_named_user(conn: sqlite3.Connection) -> None:
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    store.append_event(
        conn,
        PreferenceEvent(
            kind="liked_movie", subject_kind="movie", subject="2", value=1.0, user_id="other"
        ),
        now=AT,
    )
    store.reset_profile(conn, "default")
    assert store.count_events(conn, "default") == 0
    assert store.count_events(conn, "other") == 1


# -- snapshot cache ------------------------------------------------------------------


def _write_a_snapshot(conn: sqlite3.Connection, count: int = 1) -> None:
    store.write_snapshot(
        conn,
        store.DEFAULT_USER,
        computed_at=AT,
        event_count=count,
        total_weight=3.0,
        vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def test_snapshot_round_trips(conn: sqlite3.Connection) -> None:
    _write_a_snapshot(conn, count=5)
    snap = store.read_snapshot(conn)
    assert snap is not None
    assert snap.event_count == 5
    assert snap.total_weight == pytest.approx(3.0)
    assert np.allclose(snap.vector, [0.1, 0.2, 0.3])


def test_append_invalidates_the_snapshot(conn: sqlite3.Connection) -> None:
    _write_a_snapshot(conn)
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    assert store.read_snapshot(conn) is None


def test_forget_invalidates_the_snapshot(conn: sqlite3.Connection) -> None:
    stored = store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    _write_a_snapshot(conn)
    store.forget_event(conn, stored.id)
    assert store.read_snapshot(conn) is None


def test_reset_invalidates_the_snapshot(conn: sqlite3.Connection) -> None:
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    _write_a_snapshot(conn)
    store.reset_profile(conn)
    assert store.read_snapshot(conn) is None


# -- database-level guards -----------------------------------------------------------


def test_bad_kind_is_rejected_by_the_check_constraint(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO preference_events (ts, kind, subject_kind, subject, value) "
            "VALUES ('2024-01-01T00:00:00Z', 'adored', 'movie', '1', 1.0)"
        )


def test_model_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        PreferenceEvent(kind="adored", subject_kind="movie", subject="1", value=1.0)
