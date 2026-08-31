"""Persist and read the preference-event log, and cache the derived snapshot.

This is the only module that writes taste data. It appends immutable events, reads them back,
forgets one by id, resets a user, and keeps the ``profile_snapshots`` cache honest by deleting
it whenever the event set changes (see the table comment in migration_0003_profile.sql for why
that is a complete invalidation). The maths that turns events into a profile is in
:mod:`cinegeist.profile.update`; nothing here interprets ``value`` or applies decay.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from .model import PreferenceEvent

DEFAULT_USER = "default"

# The snapshot's genome_vector is stored as an explicit little-endian float32 blob so a cache
# written on one machine reads back the same on another, regardless of native byte order.
_BLOB_DTYPE = np.dtype("<f4")


def now_utc() -> datetime:
    """Current time as a timezone-aware UTC datetime (the clock the log is stamped with)."""
    return datetime.now(UTC)


def _to_iso(when: datetime) -> str:
    """Format a datetime as ``YYYY-MM-DDTHH:MM:SSZ`` in UTC, matching the rest of the catalog."""
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _from_iso(text: str) -> datetime:
    """Parse a stored timestamp back to an aware UTC datetime."""
    parsed = datetime.fromisoformat(text)  # 3.11+ accepts the trailing 'Z'
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _row_to_event(row: sqlite3.Row) -> PreferenceEvent:
    return PreferenceEvent(
        id=row["id"],
        user_id=row["user_id"],
        ts=_from_iso(row["ts"]),
        session_id=row["session_id"],
        kind=row["kind"],
        subject_kind=row["subject_kind"],
        subject=row["subject"],
        value=row["value"],
        weight=row["weight"],
        evidence=row["evidence"],
    )


# -- writing events ------------------------------------------------------------------


def append_event(
    conn: sqlite3.Connection, event: PreferenceEvent, *, now: datetime | None = None
) -> PreferenceEvent:
    """Append one event, returning it with its assigned ``id`` and stamped ``ts``.

    The event's own ``ts`` is kept when set (so recorded conversations replay at fixed ages);
    otherwise it is stamped with ``now``. Appending invalidates the cached snapshot.
    """
    return append_events(conn, [event], now=now)[0]


def append_events(
    conn: sqlite3.Connection,
    events: list[PreferenceEvent],
    *,
    now: datetime | None = None,
) -> list[PreferenceEvent]:
    """Append several events in one transaction, invalidating the snapshot once at the end."""
    if not events:
        return []
    stamp = now or now_utc()
    stored: list[PreferenceEvent] = []
    users: set[str] = set()
    with conn:
        for event in events:
            ts = event.ts or stamp
            cursor = conn.execute(
                """
                INSERT INTO preference_events
                    (user_id, ts, session_id, kind, subject_kind, subject, value, weight, evidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.user_id,
                    _to_iso(ts),
                    event.session_id,
                    event.kind,
                    event.subject_kind,
                    event.subject,
                    event.value,
                    event.weight,
                    event.evidence,
                ),
            )
            users.add(event.user_id)
            stored.append(
                PreferenceEvent(
                    id=int(cursor.lastrowid),
                    user_id=event.user_id,
                    ts=ts,
                    session_id=event.session_id,
                    kind=event.kind,
                    subject_kind=event.subject_kind,
                    subject=event.subject,
                    value=event.value,
                    weight=event.weight,
                    evidence=event.evidence,
                )
            )
        for user_id in users:
            _delete_snapshot(conn, user_id)
    return stored


# -- reading events ------------------------------------------------------------------


def iter_events(conn: sqlite3.Connection, user_id: str = DEFAULT_USER) -> list[PreferenceEvent]:
    """All of a user's events, oldest first."""
    rows = conn.execute(
        "SELECT * FROM preference_events WHERE user_id = ? ORDER BY ts, id", (user_id,)
    ).fetchall()
    return [_row_to_event(row) for row in rows]


def get_event(conn: sqlite3.Connection, event_id: int) -> PreferenceEvent | None:
    """One event by id, or ``None`` if it does not exist."""
    row = conn.execute("SELECT * FROM preference_events WHERE id = ?", (event_id,)).fetchone()
    return _row_to_event(row) if row is not None else None


def count_events(conn: sqlite3.Connection, user_id: str = DEFAULT_USER) -> int:
    """How many events a user has."""
    row = conn.execute(
        "SELECT COUNT(*) FROM preference_events WHERE user_id = ?", (user_id,)
    ).fetchone()
    return int(row[0])


def count_sessions(conn: sqlite3.Connection, user_id: str = DEFAULT_USER) -> int:
    """How many distinct conversations a user's events span (NULL session_ids don't count)."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM preference_events "
        "WHERE user_id = ? AND session_id IS NOT NULL",
        (user_id,),
    ).fetchone()
    return int(row[0])


# -- forgetting and resetting --------------------------------------------------------


def forget_event(conn: sqlite3.Connection, event_id: int) -> bool:
    """Delete one event by id. Returns whether a row was actually removed."""
    with conn:
        row = conn.execute(
            "SELECT user_id FROM preference_events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM preference_events WHERE id = ?", (event_id,))
        _delete_snapshot(conn, row["user_id"])
    return True


def reset_profile(conn: sqlite3.Connection, user_id: str = DEFAULT_USER) -> int:
    """Delete every event for a user (and their snapshot). Returns how many events were removed."""
    with conn:
        cursor = conn.execute("DELETE FROM preference_events WHERE user_id = ?", (user_id,))
        _delete_snapshot(conn, user_id)
    return int(cursor.rowcount)


# -- snapshot cache ------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """A cached decayed centroid and the bookkeeping needed to know if it is still valid."""

    computed_at: datetime
    event_count: int
    total_weight: float
    vector: np.ndarray


def read_snapshot(conn: sqlite3.Connection, user_id: str = DEFAULT_USER) -> Snapshot | None:
    """The cached snapshot for a user, or ``None`` if none is stored."""
    row = conn.execute(
        "SELECT computed_at, event_count, total_weight, genome_vector "
        "FROM profile_snapshots WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    vector = np.frombuffer(row["genome_vector"], dtype=_BLOB_DTYPE).astype(np.float32)
    return Snapshot(
        computed_at=_from_iso(row["computed_at"]),
        event_count=int(row["event_count"]),
        total_weight=float(row["total_weight"]),
        vector=vector,
    )


def write_snapshot(
    conn: sqlite3.Connection,
    user_id: str,
    *,
    computed_at: datetime,
    event_count: int,
    total_weight: float,
    vector: np.ndarray,
) -> None:
    """Cache a freshly computed centroid, replacing any earlier snapshot for the user."""
    blob = np.asarray(vector, dtype=_BLOB_DTYPE).tobytes()
    with conn:
        conn.execute(
            """
            INSERT INTO profile_snapshots
                (user_id, computed_at, event_count, total_weight, genome_vector)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                computed_at = excluded.computed_at,
                event_count = excluded.event_count,
                total_weight = excluded.total_weight,
                genome_vector = excluded.genome_vector
            """,
            (user_id, _to_iso(computed_at), event_count, total_weight, blob),
        )


def _delete_snapshot(conn: sqlite3.Connection, user_id: str) -> None:
    """Drop a user's cached snapshot. Runs inside the caller's transaction."""
    conn.execute("DELETE FROM profile_snapshots WHERE user_id = ?", (user_id,))
