"""SQLite connection helpers and a small forward-only migration runner.

The catalog's shape lives in ``.sql`` files next to this module (hard rule: SQL in files,
not embedded strings). ``migrate`` applies any migration whose version is newer than the
database's ``PRAGMA user_version`` and bumps the version once each script has run in full.
Because every migration is written with ``IF NOT EXISTS`` guards, an interrupted run is
simply replayed on the next call — the version only advances after a clean pass.

Adding to the schema later means writing a new ``.sql`` file and appending a
``(version, filename)`` pair to :data:`_MIGRATIONS`; never edit an already-shipped
migration, because databases in the wild have already applied it.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

from ..config import data_dir

# The highest schema version this build knows how to produce.
SCHEMA_VERSION = 3

# Migrations in apply order: (target user_version, resource filename within this package).
# schema.sql brings a blank database up to version 1; each later file evolves it. Order
# matters; keep this sorted by version, and never edit a file that has already shipped.
_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, "schema.sql"),
    (2, "migration_0002_tmdb_id_not_unique.sql"),
    (3, "migration_0003_profile.sql"),
)

_PACKAGE = "cinegeist.catalog"


def default_db_path() -> Path:
    """Where the catalog database lives by default (``data/cinegeist.db``)."""
    return data_dir() / "cinegeist.db"


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with the pragmas the catalog expects.

    Foreign keys are enforced, journalling is WAL (so a reader never blocks the long
    build's writer), and rows come back as :class:`sqlite3.Row` for name-based access.
    The parent directory is created if it does not exist, unless this is the special
    in-memory database.
    """
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def user_version(conn: sqlite3.Connection) -> int:
    """The database's current schema version (0 for a brand-new file)."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _migration_sql(filename: str) -> str:
    return resources.files(_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Bring ``conn`` up to :data:`SCHEMA_VERSION`, returning the versions applied.

    Idempotent: calling it on an up-to-date database applies nothing and returns ``[]``.
    Each migration runs as one script; ``user_version`` is advanced only after the script
    completes, so a crash mid-script leaves the version untouched and the next call replays
    the (idempotent) statements.
    """
    applied: list[int] = []
    for version, filename in _MIGRATIONS:
        if version <= user_version(conn):
            continue
        conn.executescript(_migration_sql(filename))
        # PRAGMA values can't be parameterised; version is a trusted int from _MIGRATIONS.
        conn.execute(f"PRAGMA user_version = {int(version)}")
        conn.commit()
        applied.append(version)
    return applied


def open_catalog(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the catalog database and migrate it to the current schema in one step."""
    conn = connect(default_db_path() if path is None else path)
    migrate(conn)
    return conn


def get_state(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    """Read a value from the ``build_state`` scratchpad, or ``default`` if absent."""
    row = conn.execute("SELECT value FROM build_state WHERE key = ?", (key,)).fetchone()
    return default if row is None else row["value"]


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a value into the ``build_state`` scratchpad and stamp ``updated_at``."""
    conn.execute(
        """
        INSERT INTO build_state (key, value, updated_at)
        VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        ON CONFLICT (key) DO UPDATE
            SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value),
    )
    conn.commit()
