"""Unit tests for probe selection and the stopping rules.

The catalog is hand-built with three tags of known spread — ``warm`` (variance 0.09), ``bleak``
(0.25), and a constant ``even`` (0) — over four films, so which axis "teaches most" is arithmetic
we can assert exactly, and so are the grounded poles.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.convo import probes

# rows, by genome_row: [warm, bleak, even]
#   f1 low-warm  low-bleak     f2 high-warm high-bleak
#   f3 high-warm low-bleak     f4 low-warm  high-bleak
_MATRIX = np.array(
    [
        [0.2, 0.0, 0.9],  # f1  (movie 1)
        [0.8, 1.0, 0.9],  # f2  (movie 2)
        [0.8, 0.0, 0.9],  # f3  (movie 3)
        [0.2, 1.0, 0.9],  # f4  (movie 4)
    ],
    dtype=np.float32,
)
ZERO = np.zeros(3, dtype=np.float32)


@pytest.fixture
def catalog() -> tuple[sqlite3.Connection, np.ndarray]:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(100, 0, "warm"), (200, 1, "bleak"), (300, 2, "even")],
    )
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, 'measured')",
        [
            (1, "F1 (2001)", "F1", 2001, 0),
            (2, "F2 (2002)", "F2", 2002, 1),
            (3, "F3 (2003)", "F3", 2003, 2),
            (4, "F4 (2004)", "F4", 2004, 3),
        ],
    )
    conn.commit()
    return conn, _MATRIX


# -- probe selection -----------------------------------------------------------------


def test_cold_start_picks_the_highest_spread_axis(catalog) -> None:
    conn, matrix = catalog
    probe = probes.choose_probe(conn, matrix, ZERO)
    assert probe is not None
    assert probe.axis_name == "bleak"  # variance 0.25 beats warm's 0.09


def test_pair_is_grounded_at_the_poles_and_otherwise_similar(catalog) -> None:
    conn, matrix = catalog
    probe = probes.choose_probe(conn, matrix, ZERO)
    # High pole is a bleak film (2 or 4); the low pole is the not-bleak film closest to it —
    # F3 shares F2's high "warm", so F2/F3 is the contrast that isolates bleak.
    assert probe.film_high.movie_id == 2
    assert probe.film_low.movie_id == 3


def test_uncertainty_weight_can_flip_the_choice(catalog) -> None:
    conn, matrix = catalog
    # Down-weight bleak: warm's 0.09 now beats bleak's 0.25 × 0.1.
    weight = np.array([1.0, 0.1, 1.0], dtype=np.float32)
    probe = probes.choose_probe(conn, matrix, ZERO, uncertainty=weight)
    assert probe.axis_name == "warm"


def test_already_asked_axis_is_not_repeated(catalog) -> None:
    conn, matrix = catalog
    probe = probes.choose_probe(conn, matrix, ZERO, asked_positions=frozenset({1}))
    assert probe.axis_name == "warm"


def test_excluding_films_changes_the_contested_set(catalog) -> None:
    conn, matrix = catalog
    # Drop both high-bleak films; bleak is now constant-zero, so warm wins.
    probe = probes.choose_probe(conn, matrix, ZERO, excluded_movie_ids=frozenset({2, 4}))
    assert probe.axis_name == "warm"
    assert {probe.film_high.movie_id, probe.film_low.movie_id} == {1, 3}


def test_profile_narrows_the_contested_set(catalog) -> None:
    conn, matrix = catalog
    # A profile pointing at the constant "even" axis ranks the two low-bleak films (F1, F3) top.
    # With only those two contested, bleak is constant and warm is the axis that divides them.
    profile = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    probe = probes.choose_probe(conn, matrix, profile, pool_top=2)
    assert probe.axis_name == "warm"
    assert {probe.film_high.movie_id, probe.film_low.movie_id} == {1, 3}


def test_returns_none_when_nothing_divides_the_pool() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.execute("INSERT INTO genome_tags (tag_id, position, name) VALUES (1, 0, 'flat')")
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, 'measured')",
        [(1, "A (1)", "A", 1, 0), (2, "B (2)", "B", 2, 1)],
    )
    conn.commit()
    matrix = np.array([[0.5], [0.5]], dtype=np.float32)  # identical → zero spread
    assert probes.choose_probe(conn, matrix, np.zeros(1, dtype=np.float32)) is None


# -- the escape hatch ----------------------------------------------------------------


def test_wants_to_stop_detects_the_escape_hatch() -> None:
    assert probes.wants_to_stop("just show me something")
    assert probes.wants_to_stop("Just tell me already")
    assert probes.wants_to_stop("ok, STOP ASKING and pick one")
    assert not probes.wants_to_stop("I loved Heat and Prisoners")


# -- stopping rules ------------------------------------------------------------------


def test_user_request_stops_immediately_even_on_turn_zero() -> None:
    decision = probes.should_stop(turn=0, top5_history=[], user_requested=True)
    assert decision.stop
    assert decision.reason == "user_request"


def test_hard_turn_cap() -> None:
    decision = probes.should_stop(turn=9, top5_history=[[1, 2, 3, 4, 5]])
    assert decision.stop
    assert decision.reason == "max_turns"


def test_does_not_stop_before_the_minimum() -> None:
    # Stable top-5 but only turn 1 — too early to declare victory.
    history = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    decision = probes.should_stop(turn=1, top5_history=history)
    assert not decision.stop


def test_stops_when_the_top_five_holds_steady() -> None:
    history = [[9, 8, 7, 6, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    decision = probes.should_stop(turn=4, top5_history=history)
    assert decision.stop
    assert decision.reason == "top5_stable"


def test_stops_on_a_confident_margin() -> None:
    scores = [0.9, 0.5, 0.4, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.2]  # gap 0.7
    history = [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5]]  # not stable, so margin is what fires
    decision = probes.should_stop(turn=4, top5_history=history, top_scores=scores)
    assert decision.stop
    assert decision.reason == "margin"


def test_keeps_going_when_no_rule_fires() -> None:
    scores = [0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41]  # gap 0.09 < 0.15
    history = [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5]]
    decision = probes.should_stop(turn=4, top5_history=history, top_scores=scores)
    assert not decision.stop
    assert decision.reason == "continue"
