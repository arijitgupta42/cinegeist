"""Unit tests for title resolution: exact, article, year, fuzzy, and disambiguation.

A hand-built ``movies`` table gives the cases the mini archive can't — two films that share a
title (the two *Solaris*), a comma-article title ("Matrix, The"), a number that is part of the
name rather than a year ("Blade Runner 2049"), and a two-letter title ("Up").
"""

from __future__ import annotations

import sqlite3

import pytest

from cinegeist.catalog import db
from cinegeist.convo import resolve
from cinegeist.convo.resolve import Mention


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = db.connect(":memory:")
    db.migrate(connection)
    connection.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year) VALUES (?, ?, ?, ?)",
        [
            (1, "Toy Story (1995)", "Toy Story", 1995),
            (2, "Solaris (1972)", "Solaris", 1972),
            (3, "Solaris (2002)", "Solaris", 2002),
            (4, "Matrix, The (1999)", "Matrix, The", 1999),
            (5, "Blade Runner 2049 (2017)", "Blade Runner 2049", 2017),
            (6, "Up (2009)", "Up", 2009),
            (7, "Amelie (2001)", "Amelie", 2001),
        ],
    )
    connection.commit()
    return connection


def test_exact_title_resolves(conn: sqlite3.Connection) -> None:
    result = resolve.resolve_title(conn, "Toy Story")
    assert result.is_resolved
    assert result.match.movie_id == 1


def test_matching_is_case_insensitive(conn: sqlite3.Connection) -> None:
    assert resolve.resolve_title(conn, "toy story").match.movie_id == 1


def test_leading_article_matches_comma_article_form(conn: sqlite3.Connection) -> None:
    # The user types "The Matrix"; MovieLens stores "Matrix, The".
    assert resolve.resolve_title(conn, "The Matrix").match.movie_id == 4
    assert resolve.resolve_title(conn, "matrix").match.movie_id == 4


def test_same_title_two_films_is_ambiguous(conn: sqlite3.Connection) -> None:
    result = resolve.resolve_title(conn, "Solaris")
    assert result.status == "ambiguous"
    assert {c.movie_id for c in result.candidates} == {2, 3}


def test_year_breaks_the_tie(conn: sqlite3.Connection) -> None:
    assert resolve.resolve_title(conn, "Solaris", year=1972).match.movie_id == 2
    assert resolve.resolve_title(conn, "Solaris", year=2002).match.movie_id == 3


def test_year_parsed_from_the_title(conn: sqlite3.Connection) -> None:
    assert resolve.resolve_title(conn, "Solaris (2002)").match.movie_id == 3


def test_a_number_in_the_title_is_not_a_year(conn: sqlite3.Connection) -> None:
    result = resolve.resolve_title(conn, "Blade Runner 2049")
    assert result.is_resolved
    assert result.match.movie_id == 5


def test_short_title_resolves(conn: sqlite3.Connection) -> None:
    assert resolve.resolve_title(conn, "Up").match.movie_id == 6


def test_small_typo_still_lands(conn: sqlite3.Connection) -> None:
    result = resolve.resolve_title(conn, "Amelei")  # transposed letters
    assert result.is_resolved
    assert result.match.movie_id == 7


def test_unknown_title_is_no_match(conn: sqlite3.Connection) -> None:
    result = resolve.resolve_title(conn, "Some Film That Is Not Here At All")
    assert result.status == "no_match"
    assert result.match is None


def test_resolve_mentions_pairs_each_with_its_outcome(conn: sqlite3.Connection) -> None:
    mentions = [Mention(title="Toy Story"), Mention(title="Solaris")]
    resolved = resolve.resolve_mentions(conn, mentions)
    assert [m.title for m, _ in resolved] == ["Toy Story", "Solaris"]
    assert resolved[0][1].is_resolved
    assert resolved[1][1].status == "ambiguous"
