"""Unit tests for catalog search: query parsing, tag resolution, and ranking.

Ranking runs against a real mini catalog built from the synthetic archive (two films, three
tags: animation, cerebral, space), so the cosine path is exercised end to end.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from cinegeist.catalog import genome, search
from cinegeist.catalog.build import build_catalog
from cinegeist.catalog.db import connect, migrate
from cinegeist.catalog.sources import movielens

QUIET = Console(quiet=True)


def _mini_catalog(tmp_path: Path, make_movielens_archive) -> tuple[sqlite3.Connection, np.ndarray]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    make_movielens_archive(data_dir / movielens.ARCHIVE_NAME)
    build_catalog(data_dir=data_dir, enrich=False, console=QUIET)
    conn = connect(data_dir / "cinegeist.db")
    matrix = genome.load_genome(genome.default_genome_path(data_dir))
    return conn, matrix


# -- parse_query ------------------------------------------------------------------


def test_parse_query_two_digit_decade() -> None:
    leftover, years = search.parse_query("bleak 90s")
    assert years == (1990, 1999)
    assert "bleak" in leftover
    assert "90s" not in leftover


def test_parse_query_four_digit_decade() -> None:
    assert search.parse_query("slow 1970s cinema")[1] == (1970, 1979)


def test_parse_query_recent_two_digit_decade_is_2000s() -> None:
    assert search.parse_query("quirky 00s")[1] == (2000, 2009)
    assert search.parse_query("20s")[1] == (2020, 2029)


def test_parse_query_bare_year() -> None:
    assert search.parse_query("heist 1994")[1] == (1994, 1994)


def test_parse_query_no_year() -> None:
    leftover, years = search.parse_query("bleak cerebral")
    assert years is None
    assert leftover == "bleak cerebral"


# -- resolve_tags -----------------------------------------------------------------


def test_resolve_tags_matches_single_words_by_whole_word() -> None:
    conn = connect(":memory:")
    migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(1, 0, "war"), (2, 1, "atmospheric"), (3, 2, "twist ending")],
    )
    # "warm" must not fire "war"; "atmospheric" matches; the phrase tag matches as a substring.
    matched = dict(
        (name, pos) for pos, name in search.resolve_tags(conn, "atmospheric warm twist ending")
    )
    assert "atmospheric" in matched
    assert "twist ending" in matched
    assert "war" not in matched


# -- search -----------------------------------------------------------------------


def test_search_ranks_the_on_theme_film_first(tmp_path: Path, make_movielens_archive) -> None:
    conn, matrix = _mini_catalog(tmp_path, make_movielens_archive)
    result = search.search(conn, matrix, "space", limit=5)
    assert result.tags == ["space"]
    # Solaris (space 0.8) beats Toy Story (space 0.2).
    assert result.hits[0].title == "Solaris"
    assert result.hits[0].matched[0] == ("space", pytest.approx(0.8, abs=1e-6))


def test_search_animation_favours_toy_story(tmp_path: Path, make_movielens_archive) -> None:
    conn, matrix = _mini_catalog(tmp_path, make_movielens_archive)
    result = search.search(conn, matrix, "animation", limit=5)
    assert result.hits[0].title == "Toy Story"


def test_search_year_filter_narrows_results(tmp_path: Path, make_movielens_archive) -> None:
    conn, matrix = _mini_catalog(tmp_path, make_movielens_archive)
    result = search.search(conn, matrix, "cerebral 1970s", limit=5)
    assert result.year_range == (1970, 1979)
    assert [h.title for h in result.hits] == ["Solaris"]  # Toy Story (1995) is filtered out


def test_search_raises_when_no_tag_matches(tmp_path: Path, make_movielens_archive) -> None:
    conn, matrix = _mini_catalog(tmp_path, make_movielens_archive)
    with pytest.raises(search.NoTagsMatched):
        search.search(conn, matrix, "zzzznotarealtag", limit=5)
