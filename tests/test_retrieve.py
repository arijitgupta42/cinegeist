"""Unit tests for the hard filter that builds the candidate pool.

A small hand-built catalog carries exactly the columns the filter reads — some enriched (runtime,
language, providers), some left NULL to prove an un-enriched film is kept rather than dropped when
a constraint can't be checked against it.
"""

from __future__ import annotations

import sqlite3

import pytest

from cinegeist.catalog import db
from cinegeist.profile import store
from cinegeist.profile.model import PreferenceEvent
from cinegeist.recommend import retrieve


@pytest.fixture
def catalog() -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.migrate(conn)
    # Five films. 1,2,4,5 are genome-covered; 3 has no vector so is never a candidate.
    #   1  90 min  en   fully enriched, streamable in US
    #   2 150 min  fr   enriched, not streamable in US
    #   4  runtime/lang NULL (genome-only), no provider rows
    #   5 100 min  en   enriched, streamable in US
    conn.executemany(
        "INSERT INTO movies "
        "(movie_id, title, clean_title, year, runtime, original_language, "
        " vote_average, vote_count, popularity, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "A (2001)", "A", 2001, 90, "en", 7.5, 1000, 12.0, 0, "measured"),
            (2, "B (1975)", "B", 1975, 150, "fr", 8.1, 500, 6.0, 1, "measured"),
            (3, "C (2010)", "C", 2010, None, None, None, None, None, None, "none"),
            (4, "D (2019)", "D", 2019, None, None, None, None, None, 2, "measured"),
            (5, "E (2020)", "E", 2020, 100, "en", 6.0, 200, 30.0, 3, "measured"),
        ],
    )
    conn.executemany(
        "INSERT INTO watch_providers (provider_id, name) VALUES (?, ?)",
        [(8, "Netflix")],
    )
    conn.executemany(
        "INSERT INTO movie_watch_providers (movie_id, provider_id, region, monetization) "
        "VALUES (?, ?, ?, ?)",
        [(1, 8, "US", "flatrate"), (5, 8, "US", "flatrate"), (2, 8, "FR", "flatrate")],
    )
    conn.commit()
    return conn


def _ids(pool: list[retrieve.Candidate]) -> set[int]:
    return {c.movie_id for c in pool}


def test_only_genome_covered_films_are_candidates(catalog) -> None:
    pool = retrieve.retrieve(catalog)
    assert _ids(pool) == {1, 2, 4, 5}  # film 3 has no genome vector


def test_excluded_ids_are_dropped(catalog) -> None:
    pool = retrieve.retrieve(catalog, retrieve.Constraints(exclude_ids=frozenset({1, 4})))
    assert _ids(pool) == {2, 5}


def test_runtime_ceiling_filters_known_but_keeps_unknown(catalog) -> None:
    # Under 120 min: film 2 (150) is out; film 4 (NULL runtime) is kept — we can't say it violates.
    pool = retrieve.retrieve(catalog, retrieve.Constraints(max_runtime=120))
    assert _ids(pool) == {1, 4, 5}


def test_release_window(catalog) -> None:
    pool = retrieve.retrieve(catalog, retrieve.Constraints(min_year=2000, max_year=2019))
    assert _ids(pool) == {1, 4}  # 2 is 1975, 5 is 2020


def test_language_allow_set_filters_known_but_keeps_unknown(catalog) -> None:
    # English only: film 2 (fr) is out; film 4 (NULL language) is kept.
    pool = retrieve.retrieve(catalog, retrieve.Constraints(languages=frozenset({"en"})))
    assert _ids(pool) == {1, 4, 5}


def test_require_available_filters_by_region(catalog) -> None:
    pool = retrieve.retrieve(catalog, retrieve.Constraints(require_available=True, region="US"))
    assert _ids(pool) == {1, 5}  # only these two stream in US


def test_require_available_keeps_all_when_no_provider_data() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.execute(
        "INSERT INTO movies (movie_id, title, clean_title, genome_row, genome_source) "
        "VALUES (1, 'A', 'A', 0, 'measured')"
    )
    conn.commit()
    # No provider rows at all: "require available" can't be checked, so it keeps the film.
    pool = retrieve.retrieve(conn, retrieve.Constraints(require_available=True, region="US"))
    assert _ids(pool) == {1}


def test_seen_movie_ids_reads_movie_events_only(catalog) -> None:
    store.append_events(
        catalog,
        [
            PreferenceEvent.liked_movie(1, session_id="s1"),
            PreferenceEvent.disliked_movie(2, session_id="s1"),
            PreferenceEvent.axis_answer(999, 0.5, session_id="s1"),  # a tag event, not a film
            PreferenceEvent(
                kind="post_watch_feedback",
                subject_kind="movie",
                subject="5",
                value=1.0,
                session_id="s1",
            ),
        ],
    )
    assert retrieve.seen_movie_ids(catalog) == frozenset({1, 2, 5})


def test_seen_films_are_excluded_end_to_end(catalog) -> None:
    store.append_event(catalog, PreferenceEvent.liked_movie(1))
    seen = retrieve.seen_movie_ids(catalog)
    pool = retrieve.retrieve(catalog, retrieve.Constraints(exclude_ids=seen))
    assert 1 not in _ids(pool)
    assert _ids(pool) == {2, 4, 5}
