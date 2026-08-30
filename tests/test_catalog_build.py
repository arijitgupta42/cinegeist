"""End-to-end tests for the catalog build against the synthetic archive.

These exercise the whole pipeline offline: the archive is placed in the data directory so the
download stage reuses it (never reaching the network), then ingest and genome build run for
real. Resumability and ``--force`` are checked here because they are session-2 acceptance
criteria.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console

from cinegeist.catalog import genome
from cinegeist.catalog.build import build_catalog
from cinegeist.catalog.db import connect, get_state
from cinegeist.catalog.sources import movielens

QUIET = Console(quiet=True)


def _seed_archive(data_dir: Path, make_movielens_archive) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    make_movielens_archive(data_dir / movielens.ARCHIVE_NAME)


def test_build_populates_movies(tmp_path: Path, make_movielens_archive) -> None:
    data_dir = tmp_path / "data"
    _seed_archive(data_dir, make_movielens_archive)
    build_catalog(data_dir=data_dir, console=QUIET)

    conn = connect(data_dir / "cinegeist.db")
    rows = {r["movie_id"]: r for r in conn.execute("SELECT * FROM movies ORDER BY movie_id")}
    assert set(rows) == {1, 2, 3}
    assert rows[1]["clean_title"] == "Toy Story"
    assert rows[1]["year"] == 1995
    assert rows[1]["imdb_id"] == "tt0114709"
    assert rows[1]["tmdb_id"] == 862
    # Film 3 has no year and blank links.
    assert rows[3]["year"] is None
    assert rows[3]["imdb_id"] is None
    assert rows[3]["tmdb_id"] is None


def test_build_writes_the_genome_and_links_rows(tmp_path: Path, make_movielens_archive) -> None:
    data_dir = tmp_path / "data"
    _seed_archive(data_dir, make_movielens_archive)
    build_catalog(data_dir=data_dir, console=QUIET)

    matrix = genome.load_genome(genome.default_genome_path(data_dir))
    # Films 1 and 2 are scored; film 3 is not; film 999 (not in movies) is dropped.
    assert matrix.shape == (2, 3)

    conn = connect(data_dir / "cinegeist.db")
    scored = {
        r["movie_id"]: (r["genome_row"], r["genome_source"])
        for r in conn.execute(
            "SELECT movie_id, genome_row, genome_source FROM movies ORDER BY movie_id"
        )
    }
    assert scored[1] == (0, "measured")
    assert scored[2] == (1, "measured")
    assert scored[3] == (None, "none")

    # The row a film points to holds that film's vector.
    assert np.allclose(matrix[scored[1][0]], [0.9, 0.1, 0.2])
    assert np.allclose(matrix[scored[2][0]], [0.05, 0.95, 0.8])


def test_build_records_genome_tags_and_state(tmp_path: Path, make_movielens_archive) -> None:
    data_dir = tmp_path / "data"
    _seed_archive(data_dir, make_movielens_archive)
    build_catalog(data_dir=data_dir, console=QUIET)

    conn = connect(data_dir / "cinegeist.db")
    tags = conn.execute(
        "SELECT tag_id, position, name FROM genome_tags ORDER BY position"
    ).fetchall()
    assert [(t["position"], t["name"]) for t in tags] == [
        (0, "animation"),
        (1, "cerebral"),
        (2, "space"),
    ]
    assert get_state(conn, "genome_rows") == "2"
    assert get_state(conn, "genome_cols") == "3"
    assert get_state(conn, "genome_dtype") == "float32"
    assert get_state(conn, "movielens_ingested_at") is not None
    assert get_state(conn, "genome_built_at") is not None


def test_rebuild_is_idempotent(tmp_path: Path, make_movielens_archive) -> None:
    data_dir = tmp_path / "data"
    _seed_archive(data_dir, make_movielens_archive)
    build_catalog(data_dir=data_dir, console=QUIET)
    # A second run finds every stage done and must neither error nor change the result.
    build_catalog(data_dir=data_dir, console=QUIET)

    conn = connect(data_dir / "cinegeist.db")
    assert conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == 3
    assert genome.load_genome(genome.default_genome_path(data_dir)).shape == (2, 3)


def test_missing_genome_file_triggers_a_rebuild(tmp_path: Path, make_movielens_archive) -> None:
    data_dir = tmp_path / "data"
    _seed_archive(data_dir, make_movielens_archive)
    build_catalog(data_dir=data_dir, console=QUIET)

    # Delete the memmap but leave the "done" flag: the guard requires both, so it rebuilds.
    genome.default_genome_path(data_dir).unlink()
    build_catalog(data_dir=data_dir, console=QUIET)
    assert genome.default_genome_path(data_dir).exists()


def test_force_rebuilds_from_the_existing_archive(tmp_path: Path, make_movielens_archive) -> None:
    data_dir = tmp_path / "data"
    _seed_archive(data_dir, make_movielens_archive)
    build_catalog(data_dir=data_dir, console=QUIET)
    build_catalog(data_dir=data_dir, force=True, console=QUIET)

    conn = connect(data_dir / "cinegeist.db")
    assert (
        conn.execute("SELECT COUNT(*) FROM movies WHERE genome_source = 'measured'").fetchone()[0]
        == 2
    )
