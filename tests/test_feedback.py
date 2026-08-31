"""Unit tests for the post-watch feedback loop.

Verdict matching is pinned phrase by phrase (negations must not read as their opposite), and the
learning loop is proved end to end: recording a "not for me" against a film measurably pushes the
recomputed profile away from that film — the acceptance test for this PR (plan.md §10).
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from cinegeist import feedback
from cinegeist.catalog import db
from cinegeist.profile import store, update
from cinegeist.profile.model import PreferenceEvent

# Two films: A at [1, 0], B at [0.7, 0.7] (so B starts fairly close to a profile built on A).
_MATRIX = np.array([[1.0, 0.0], [0.7, 0.7]], dtype=np.float32)


@pytest.fixture
def catalog() -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(1, 0, "warm"), (2, 1, "bleak")],
    )
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, 'measured')",
        [(10, "A (2001)", "A", 2001, 0), (20, "B (2002)", "B", 2002, 1)],
    )
    conn.commit()
    return conn


# -- reading a verdict from text -----------------------------------------------------


def test_match_feedback_reads_each_verdict() -> None:
    assert feedback.match_feedback("watched and loved it").key == "loved"
    assert feedback.match_feedback("it was fine I guess").key == "fine"
    assert feedback.match_feedback("nah, not for me").key == "not_for_me"
    assert feedback.match_feedback("already seen it").key == "already_seen"


def test_negation_is_not_read_as_a_positive() -> None:
    # "not for me" must land on the negative verdict, never trip the "for me"-adjacent positives.
    assert feedback.match_feedback("that one wasn't for me").key == "not_for_me"
    assert feedback.match_feedback("didn't like it").key == "not_for_me"


def test_match_feedback_returns_none_for_unrelated_text() -> None:
    assert feedback.match_feedback("") is None
    assert feedback.match_feedback("tell me about the director") is None
    # Matching is phrase-based, so it only runs in the feedback phase — the engine never asks it to
    # disambiguate a film someone is naming. Within that phase these clearly aren't verdicts.
    assert feedback.match_feedback("why did you pick that") is None


def test_wants_more_is_distinct_from_a_verdict() -> None:
    assert feedback.wants_more("show me three more")
    assert feedback.wants_more("what else have you got")
    assert not feedback.wants_more("watched and loved it")
    assert not feedback.wants_more("not for me")


def test_get_verdict_and_vocabulary() -> None:
    assert feedback.get_verdict("loved").value == 1.0
    assert feedback.get_verdict("not_for_me").value < 0
    assert feedback.get_verdict("already_seen").value == 0.0  # marks seen, no taste direction
    with pytest.raises(KeyError):
        feedback.get_verdict("nonsense")


# -- recording an event --------------------------------------------------------------


def test_record_feedback_appends_a_post_watch_event(catalog) -> None:
    event = feedback.record_feedback(
        catalog, 10, "loved", session_id="s1", evidence="stuck with me for days"
    )
    assert event.id is not None
    assert event.kind == "post_watch_feedback"
    assert event.subject_kind == "movie" and event.subject == "10"
    assert event.value == 1.0 and event.weight == 1.5
    assert event.evidence == "stuck with me for days"

    stored = store.get_event(catalog, event.id)
    assert stored is not None and stored.kind == "post_watch_feedback"


def test_record_feedback_accepts_a_verdict_object(catalog) -> None:
    verdict = feedback.get_verdict("not_for_me")
    event = feedback.record_feedback(catalog, 20, verdict)
    assert event.value == verdict.value and event.weight == verdict.weight


def test_a_recorded_film_becomes_a_seen_id(catalog) -> None:
    from cinegeist.recommend import retrieve

    feedback.record_feedback(catalog, 20, "already_seen")
    assert 20 in retrieve.seen_movie_ids(catalog)  # never recommend it again


# -- the learning loop ---------------------------------------------------------------


def test_negative_feedback_pushes_the_profile_away_from_the_film(catalog) -> None:
    vector_b = _MATRIX[1]
    # Start from a profile that likes film A, so film B (nearby) has a positive cosine.
    store.append_event(catalog, PreferenceEvent.liked_movie(10, weight=1.5))
    before = update.compute_profile(catalog, _MATRIX)
    cos_before = float(
        vector_b
        @ before.genome_vector
        / (np.linalg.norm(vector_b) * np.linalg.norm(before.genome_vector))
    )
    assert cos_before > 0

    # "Not for me" on B should drag the centroid away from B.
    feedback.record_feedback(catalog, 20, "not_for_me")
    after = update.compute_profile(catalog, _MATRIX)
    cos_after = float(
        vector_b
        @ after.genome_vector
        / (np.linalg.norm(vector_b) * np.linalg.norm(after.genome_vector))
    )
    assert cos_after < cos_before  # measurably shifted
    assert cos_after < 0  # actively pushed away, not merely reduced


def test_already_seen_does_not_move_taste(catalog) -> None:
    store.append_event(catalog, PreferenceEvent.liked_movie(10, weight=1.5))
    before = update.compute_profile(catalog, _MATRIX).genome_vector.copy()
    feedback.record_feedback(catalog, 20, "already_seen")  # value 0 → no direction
    after = update.compute_profile(catalog, _MATRIX).genome_vector
    assert np.allclose(before, after)
