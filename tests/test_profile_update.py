"""Unit tests for the decay maths and profile computation.

A hand-built three-tag catalog (animation, cerebral, space) with two genome-covered films —
Toy Story [0.9, 0.1, 0.2] and Solaris [0.05, 0.95, 0.8] — lets us assert exact centroid values
rather than eyeballing them. The maths under test is pure and deterministic; that is the whole
reason it is separable from the store.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.profile import store, update
from cinegeist.profile.model import PreferenceEvent

AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
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
            (3, "Unvectored (2000)", "Unvectored", 2000, None, "none"),
        ],
    )
    conn.commit()
    matrix = np.vstack([TOY_STORY, SOLARIS])
    return conn, matrix


# -- decay_factor --------------------------------------------------------------------


def test_decay_factor_halves_at_the_half_life() -> None:
    assert update.decay_factor(0.0) == pytest.approx(1.0)
    assert update.decay_factor(update.HALF_LIFE_DAYS) == pytest.approx(0.5)
    assert update.decay_factor(2 * update.HALF_LIFE_DAYS) == pytest.approx(0.25)


def test_decay_factor_clamps_future_timestamps() -> None:
    # A timestamp in the future (clock skew) must not amplify past weight 1.0.
    assert update.decay_factor(-100.0) == pytest.approx(1.0)


# -- the centroid --------------------------------------------------------------------


def test_single_like_is_that_films_vector(catalog) -> None:
    conn, matrix = catalog
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    profile = update.compute_profile(conn, matrix, now=AT)
    assert np.allclose(profile.genome_vector, TOY_STORY, atol=1e-6)
    assert profile.total_weight == pytest.approx(1.0)
    assert profile.event_count == 1


def test_a_dislike_pushes_away(catalog) -> None:
    conn, matrix = catalog
    store.append_events(
        conn,
        [PreferenceEvent.liked_movie(1), PreferenceEvent.disliked_movie(2)],
        now=AT,
    )
    profile = update.compute_profile(conn, matrix, now=AT)
    # (1·ToyStory − 1·Solaris) / (|1| + |−1|)
    expected = (TOY_STORY - SOLARIS) / 2.0
    assert np.allclose(profile.genome_vector, expected, atol=1e-6)
    # Only animation stays positive; cerebral and space go negative.
    assert [a.name for a in profile.affinities] == ["animation"]
    assert profile.aversions[0].name == "cerebral"


def test_older_evidence_counts_for_less(catalog) -> None:
    conn, matrix = catalog
    fresh = PreferenceEvent.liked_movie(1)
    old = PreferenceEvent.liked_movie(2)
    store.append_event(conn, PreferenceEvent(**{**fresh.__dict__, "ts": AT}), now=AT)
    store.append_event(
        conn,
        PreferenceEvent(**{**old.__dict__, "ts": AT - timedelta(days=update.HALF_LIFE_DAYS)}),
        now=AT,
    )
    profile = update.compute_profile(conn, matrix, now=AT)
    expected = (TOY_STORY + 0.5 * SOLARIS) / 1.5
    assert np.allclose(profile.genome_vector, expected, atol=1e-6)
    assert profile.total_weight == pytest.approx(1.5)


def test_an_axis_answer_moves_a_single_axis(catalog) -> None:
    conn, matrix = catalog
    store.append_event(conn, PreferenceEvent.axis_answer(20, 1.0), now=AT)  # tag 20 = cerebral
    profile = update.compute_profile(conn, matrix, now=AT)
    assert np.allclose(profile.genome_vector, [0.0, 1.0, 0.0], atol=1e-6)


def test_unvectored_film_is_evidence_but_no_direction(catalog) -> None:
    conn, matrix = catalog
    store.append_event(conn, PreferenceEvent.liked_movie(3), now=AT)  # genome_row IS NULL
    profile = update.compute_profile(conn, matrix, now=AT)
    assert profile.event_count == 1
    assert profile.total_weight == pytest.approx(0.0)
    assert profile.is_empty  # nothing to point at, even though there is an event


def test_empty_profile(catalog) -> None:
    conn, matrix = catalog
    profile = update.compute_profile(conn, matrix, now=AT)
    assert profile.is_empty
    assert profile.event_count == 0
    assert profile.axes == ()


# -- evidence attribution ------------------------------------------------------------


def test_axes_cite_the_event_that_produced_them(catalog) -> None:
    conn, matrix = catalog
    store.append_events(
        conn,
        [
            PreferenceEvent.liked_movie(1, evidence="loved Toy Story"),
            PreferenceEvent.disliked_movie(2, evidence="hated Solaris"),
        ],
        now=AT,
    )
    profile = update.compute_profile(conn, matrix, now=AT)
    animation = profile.affinities[0]
    assert animation.name == "animation"
    assert animation.evidence == "loved Toy Story"
    assert animation.source == "Toy Story"
    cerebral = profile.aversions[0]
    assert cerebral.name == "cerebral"
    assert cerebral.evidence == "hated Solaris"


# -- the snapshot cache --------------------------------------------------------------


def test_compute_profile_writes_a_snapshot(catalog) -> None:
    conn, matrix = catalog
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    update.compute_profile(conn, matrix, now=AT)
    snap = store.read_snapshot(conn)
    assert snap is not None
    assert snap.event_count == 1
    assert np.allclose(snap.vector, TOY_STORY, atol=1e-6)


def test_load_vector_reuses_a_valid_snapshot(catalog) -> None:
    conn, matrix = catalog
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)  # invalidates any snapshot
    # A sentinel snapshot whose count matches the one event: if load_vector returns it, it reused.
    store.write_snapshot(
        conn,
        store.DEFAULT_USER,
        computed_at=AT,
        event_count=1,
        total_weight=2.0,
        vector=np.array([7.0, 7.0, 7.0], dtype=np.float32),
    )
    vector, total_weight = update.load_vector(conn, matrix, now=AT)
    assert np.allclose(vector, [7.0, 7.0, 7.0])
    assert total_weight == pytest.approx(2.0)


def test_load_vector_decays_total_weight_forward(catalog) -> None:
    conn, matrix = catalog
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    store.write_snapshot(
        conn,
        store.DEFAULT_USER,
        computed_at=AT,
        event_count=1,
        total_weight=2.0,
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    later = AT + timedelta(days=update.HALF_LIFE_DAYS)
    vector, total_weight = update.load_vector(conn, matrix, now=later)
    assert np.allclose(vector, [1.0, 0.0, 0.0])  # direction is time-invariant
    assert total_weight == pytest.approx(1.0)  # mass halves over one half-life


def test_load_vector_recomputes_on_a_count_mismatch(catalog) -> None:
    conn, matrix = catalog
    store.append_event(conn, PreferenceEvent.liked_movie(1), now=AT)
    # A snapshot claiming five events no longer matches the one real event, so it is discarded.
    store.write_snapshot(
        conn,
        store.DEFAULT_USER,
        computed_at=AT,
        event_count=5,
        total_weight=2.0,
        vector=np.array([7.0, 7.0, 7.0], dtype=np.float32),
    )
    vector, _ = update.load_vector(conn, matrix, now=AT)
    assert np.allclose(vector, TOY_STORY, atol=1e-6)  # recomputed from the log
