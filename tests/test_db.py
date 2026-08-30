"""Unit tests for the catalog database: schema, migrations, and the state scratchpad.

Everything runs against a temporary on-disk SQLite file — no network, no shared state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cinegeist.catalog import db

# Every table schema.sql is expected to create. If you add one, add it here too.
EXPECTED_TABLES = {
    "movies",
    "genome_tags",
    "genres",
    "movie_genres",
    "keywords",
    "movie_keywords",
    "people",
    "credits",
    "movie_countries",
    "watch_providers",
    "movie_watch_providers",
    "collections",
    "build_state",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def _fresh(tmp_path: Path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "catalog.db")
    db.migrate(conn)
    return conn


def test_migrate_creates_every_table(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    assert EXPECTED_TABLES <= _table_names(conn)


def test_migrate_sets_the_user_version(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "catalog.db")
    assert db.user_version(conn) == 0
    applied = db.migrate(conn)
    assert applied == [1, 2]
    assert db.user_version(conn) == db.SCHEMA_VERSION


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    # A second call has nothing new to apply and must not error.
    assert db.migrate(conn) == []
    assert db.user_version(conn) == db.SCHEMA_VERSION


def test_reopening_a_migrated_database_applies_nothing(tmp_path: Path) -> None:
    path = tmp_path / "catalog.db"
    db.migrate(db.connect(path))
    reopened = db.connect(path)
    assert db.migrate(reopened) == []
    assert db.user_version(reopened) == db.SCHEMA_VERSION


def test_connect_enforces_foreign_keys(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    # A genre credit for a movie that does not exist must be rejected.
    conn.execute("INSERT INTO genres (genre_id, name) VALUES (1, 'Drama')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO movie_genres (movie_id, genre_id) VALUES (999, 1)")


def test_genome_source_check_constraint(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    conn.execute(
        "INSERT INTO movies (movie_id, title, genome_source) VALUES (1, 'A (1999)', 'measured')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO movies (movie_id, title, genome_source) VALUES (2, 'B (2000)', 'guessed')"
        )


def test_genome_source_defaults_to_none(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    conn.execute("INSERT INTO movies (movie_id, title) VALUES (1, 'A (1999)')")
    row = conn.execute("SELECT genome_source FROM movies WHERE movie_id = 1").fetchone()
    assert row["genome_source"] == "none"


def test_movies_and_genome_tags_round_trip(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    conn.execute(
        """
        INSERT INTO movies (movie_id, tmdb_id, title, clean_title, year, genome_row, genome_source)
        VALUES (1, 862, 'Toy Story (1995)', 'Toy Story', 1995, 0, 'measured')
        """
    )
    conn.execute("INSERT INTO genome_tags (tag_id, position, name) VALUES (1, 0, 'pixar')")
    movie = conn.execute("SELECT * FROM movies WHERE movie_id = 1").fetchone()
    assert movie["clean_title"] == "Toy Story"
    assert movie["genome_row"] == 0
    tag = conn.execute("SELECT name FROM genome_tags WHERE position = 0").fetchone()
    assert tag["name"] == "pixar"


def test_genome_row_is_unique(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    conn.execute("INSERT INTO movies (movie_id, title, genome_row) VALUES (1, 'A (1)', 5)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO movies (movie_id, title, genome_row) VALUES (2, 'B (2)', 5)")


def test_tmdb_id_is_not_unique(tmp_path: Path) -> None:
    # The real MovieLens links.csv points several movieIds at one tmdbId; both must ingest.
    conn = _fresh(tmp_path)
    conn.execute("INSERT INTO movies (movie_id, tmdb_id, title) VALUES (1, 862, 'A (1)')")
    conn.execute("INSERT INTO movies (movie_id, tmdb_id, title) VALUES (2, 862, 'B (2)')")
    count = conn.execute("SELECT COUNT(*) FROM movies WHERE tmdb_id = 862").fetchone()[0]
    assert count == 2


def test_two_null_genome_rows_are_allowed(tmp_path: Path) -> None:
    # The uniqueness index is partial (WHERE genome_row IS NOT NULL), so many un-vectored
    # films can coexist — most of the catalog, before the genome is attached.
    conn = _fresh(tmp_path)
    conn.execute("INSERT INTO movies (movie_id, title) VALUES (1, 'A (1)')")
    conn.execute("INSERT INTO movies (movie_id, title) VALUES (2, 'B (2)')")
    count = conn.execute("SELECT COUNT(*) FROM movies WHERE genome_row IS NULL").fetchone()[0]
    assert count == 2


def test_cascade_delete_removes_join_rows(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    conn.execute("INSERT INTO movies (movie_id, title) VALUES (1, 'A (1)')")
    conn.execute("INSERT INTO genres (genre_id, name) VALUES (18, 'Drama')")
    conn.execute("INSERT INTO movie_genres (movie_id, genre_id) VALUES (1, 18)")
    conn.execute("DELETE FROM movies WHERE movie_id = 1")
    assert conn.execute("SELECT COUNT(*) FROM movie_genres").fetchone()[0] == 0


def test_state_scratchpad_get_set(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    assert db.get_state(conn, "missing") is None
    assert db.get_state(conn, "missing", "fallback") == "fallback"
    db.set_state(conn, "genome_rows", "13176")
    assert db.get_state(conn, "genome_rows") == "13176"
    # Upsert: writing the same key again overwrites rather than duplicating.
    db.set_state(conn, "genome_rows", "13177")
    assert db.get_state(conn, "genome_rows") == "13177"
    assert conn.execute("SELECT COUNT(*) FROM build_state").fetchone()[0] == 1


def test_default_db_path_lives_under_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CINEGEIST_DATA_DIR", raising=False)
    assert db.default_db_path() == Path("data") / "cinegeist.db"


def test_data_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CINEGEIST_DATA_DIR", str(tmp_path))
    assert db.default_db_path() == tmp_path / "cinegeist.db"
