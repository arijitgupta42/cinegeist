"""Unit tests for predicted genome vectors (catalog/predict.py).

A tiny catalog with two measured "clusters" — P films (genre 1, keyword 10, English) with genome
vector [0.9, 0.1, 0, 0], and Q films (genre 2, keyword 20, French) with [0, 0, 0.9, 0.9] — plus
target films that share P's or Q's features but have no measured vector. A working ridge map must
hand each target a vector closer to the cluster it resembles, flag it predicted, and leave a truly
featureless film untouched. Everything is offline and deterministic.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from cinegeist.catalog import db, genome, predict
from cinegeist.catalog.genome import cosine_scores

P_VECTOR = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
Q_VECTOR = np.array([0.0, 0.0, 0.9, 0.9], dtype=np.float32)


def _rows(path: Path) -> np.ndarray:
    """Read the whole genome memmap into a plain array and release the file handle."""
    matrix = genome.load_genome(path)
    try:
        return np.array(matrix, dtype=np.float32)
    finally:
        del matrix


@pytest.fixture
def catalog(tmp_path: Path) -> tuple[sqlite3.Connection, Path]:
    conn = db.connect(str(tmp_path / "cinegeist.db"))
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(1, 0, "a"), (2, 1, "b"), (3, 2, "c"), (4, 3, "d")],
    )
    conn.executemany(
        "INSERT INTO genres (genre_id, name) VALUES (?, ?)", [(1, "drama"), (2, "action")]
    )
    conn.executemany(
        "INSERT INTO keywords (keyword_id, name) VALUES (?, ?)", [(10, "slow"), (20, "boom")]
    )
    # 1-3 measured P; 4-6 measured Q; 7 target like P; 8 target like Q; 9 featureless (no year,
    # no language, no genre, no keyword) — nothing to predict from, so it must stay uncovered.
    movies = [
        (1, "P1", 1995, "en"),
        (2, "P2", 1998, "en"),
        (3, "P3", 2001, "en"),
        (4, "Q1", 1980, "fr"),
        (5, "Q2", 1982, "fr"),
        (6, "Q3", 1985, "fr"),
        (7, "TargetP", 2000, "en"),
        (8, "TargetQ", 1983, "fr"),
        (9, "Featureless", None, None),
    ]
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, original_language) "
        "VALUES (?, ?, ?, ?, ?)",
        [(mid, title, title, year, lang) for mid, title, year, lang in movies],
    )
    conn.executemany(
        "INSERT INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
        [(mid, 1) for mid in (1, 2, 3, 7)] + [(mid, 2) for mid in (4, 5, 6, 8)],
    )
    conn.executemany(
        "INSERT INTO movie_keywords (movie_id, keyword_id) VALUES (?, ?)",
        [(mid, 10) for mid in (1, 2, 3, 7)] + [(mid, 20) for mid in (4, 5, 6, 8)],
    )

    scores = []
    for mid in (1, 2, 3):
        scores += [(mid, 0, 0.9), (mid, 1, 0.1)]
    for mid in (4, 5, 6):
        scores += [(mid, 2, 0.9), (mid, 3, 0.9)]
    genome_path = tmp_path / "genome.npy"
    row_of = genome.build_memmap(genome_path, 4, scores)
    with conn:
        conn.executemany(
            "UPDATE movies SET genome_row = ?, genome_source = 'measured' WHERE movie_id = ?",
            [(row, mid) for mid, row in row_of.items()],
        )
    conn.commit()
    return conn, genome_path


# -- the pieces ----------------------------------------------------------------------


def test_write_matrix_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "g.npy"
    matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    genome.write_matrix(path, matrix)
    assert np.allclose(_rows(path), matrix)
    # A rewrite replaces rather than grows the file.
    genome.write_matrix(path, matrix[:1])
    assert _rows(path).shape == (1, 2)


def test_feature_matrix_is_one_hot_with_a_bias(catalog) -> None:
    conn, _ = catalog
    space = predict.build_feature_space(conn, min_keyword_films=1)
    features = predict.feature_matrix(conn, [1], space)  # P1: genre 1, keyword 10, 1990s, en
    assert features[0, space.bias_position] == 1.0
    assert features[0, space.genre_index[1]] == 1.0
    assert features[0, space.genre_index[2]] == 0.0
    assert features[0, space.keyword_index[10]] == 1.0
    assert features[0, space.decade_index[1990]] == 1.0
    assert features[0, space.language_index["en"]] == 1.0


def test_fit_ridge_has_the_right_shape(catalog) -> None:
    conn, _ = catalog
    space = predict.build_feature_space(conn, min_keyword_films=1)
    x = predict.feature_matrix(conn, [1, 2, 3, 4, 5, 6], space)
    y = np.random.default_rng(0).random((6, 4)).astype(np.float32)
    weights = predict.fit_ridge(x, y, alpha=1.0)
    assert weights.shape == (space.dimension, 4)


# -- the prediction ------------------------------------------------------------------


def test_predicts_vectors_that_resemble_the_matching_cluster(catalog) -> None:
    conn, genome_path = catalog
    result = predict.predict_missing(conn, genome_path, min_keyword_films=1)
    assert result.trained_on == 6
    assert result.predicted == 2  # targets 7 and 8; the featureless film 9 is left out

    def source_and_row(movie_id: int) -> tuple[str, int | None]:
        row = conn.execute(
            "SELECT genome_source, genome_row FROM movies WHERE movie_id = ?", (movie_id,)
        ).fetchone()
        return row["genome_source"], row["genome_row"]

    matrix = _rows(genome_path)
    # Target 7 shares P's features, so its predicted vector must sit closer to P than to Q.
    src7, row7 = source_and_row(7)
    assert src7 == "predicted"
    vec7 = matrix[row7]
    assert cosine_scores(P_VECTOR[None, :], vec7)[0] > cosine_scores(Q_VECTOR[None, :], vec7)[0]
    # Target 8 shares Q's features.
    src8, row8 = source_and_row(8)
    assert src8 == "predicted"
    vec8 = matrix[row8]
    assert cosine_scores(Q_VECTOR[None, :], vec8)[0] > cosine_scores(P_VECTOR[None, :], vec8)[0]


def test_a_featureless_film_is_left_uncovered(catalog) -> None:
    conn, genome_path = catalog
    predict.predict_missing(conn, genome_path, min_keyword_films=1)
    source, row = conn.execute(
        "SELECT genome_source, genome_row FROM movies WHERE movie_id = 9"
    ).fetchone()
    assert source == "none"
    assert row is None


def test_measured_vectors_survive_and_stay_measured(catalog) -> None:
    conn, genome_path = catalog
    predict.predict_missing(conn, genome_path, min_keyword_films=1)
    matrix = _rows(genome_path)
    for mid, expected in ((1, P_VECTOR), (4, Q_VECTOR)):
        source, row = conn.execute(
            "SELECT genome_source, genome_row FROM movies WHERE movie_id = ?", (mid,)
        ).fetchone()
        assert source == "measured"
        assert np.allclose(matrix[row], expected, atol=1e-6)


def test_predicted_vectors_are_clipped_to_unit_range(catalog) -> None:
    conn, genome_path = catalog
    predict.predict_missing(conn, genome_path, min_keyword_films=1)
    matrix = _rows(genome_path)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_re_running_is_idempotent_and_does_not_grow_the_genome(catalog) -> None:
    conn, genome_path = catalog
    first = predict.predict_missing(conn, genome_path, min_keyword_films=1)
    rows_after_first = _rows(genome_path).shape[0]
    second = predict.predict_missing(conn, genome_path, min_keyword_films=1)
    rows_after_second = _rows(genome_path).shape[0]
    assert first.predicted == second.predicted == 2
    assert rows_after_first == rows_after_second == 8  # 6 measured + 2 predicted, not 10


def test_no_measured_vectors_predicts_nothing(tmp_path: Path) -> None:
    # A genome-only-less catalog (or one built before the genome stage) has nothing to learn from.
    conn = db.connect(str(tmp_path / "cinegeist.db"))
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)", [(1, 0, "a")]
    )
    conn.execute("INSERT INTO movies (movie_id, title, year) VALUES (1, 'x', 2000)")
    conn.commit()
    genome_path = tmp_path / "genome.npy"
    genome.write_matrix(genome_path, np.zeros((1, 1), dtype=np.float32))
    result = predict.predict_missing(conn, genome_path)
    assert result.predicted == 0
