"""Predicted genome vectors for films the tag genome doesn't cover (plan.md §2.2, Phase 2).

The MovieLens tag genome only covers well-rated, well-tagged films, and it regenerates
periodically rather than daily — so a recent or obscure film that TMDB knows about often has no
genome vector, and without a vector the recommender can't score it at all. This bridges the gap:
fit a linear map from cheap TMDB features (genres, the commonest keywords, release decade, and
original language, as a sparse one-hot) to genome vectors, using the films where we have *both*,
then predict a vector for the films where we have only the features.

It's ridge regression, solved with numpy's normal equations — no GPU, no scikit-learn, and it
runs in one matmul-and-solve over the catalog. Predicted vectors go into the same genome memmap as
the measured ones (the recommender indexes a single matrix), and the film is flagged
``genome_source='predicted'``, which the scorer trusts less than a measured one
(:func:`cinegeist.recommend.score.confidence_for`). The honesty principle applies as everywhere
else: a guess is labelled a guess and discounted, never passed off as measured signal.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from . import genome

# Defaults that bound the feature space so the normal equations stay small regardless of catalog
# size: only keywords common enough to generalise become features, capped at the most frequent.
DEFAULT_MAX_KEYWORDS = 2000
DEFAULT_MIN_KEYWORD_FILMS = 5
# Ridge penalty. One-hot features and targets in [0, 1] make a modest penalty about right; it keeps
# a rare feature from swinging the fit and keeps the normal-equation matrix well-conditioned.
DEFAULT_ALPHA = 1.0


@dataclass(frozen=True)
class FeatureSpace:
    """The one-hot columns the regression uses, and where each lives in a film's feature vector.

    The layout is ``[genres | keywords | decades | languages | bias]``. ``bias`` is a single
    always-on column so the model has an intercept — the base vector a film with no known features
    would get — while a film with at least one real feature is what we actually predict for.
    """

    genre_index: dict[int, int]
    keyword_index: dict[int, int]
    decade_index: dict[int, int]
    language_index: dict[str, int]

    @property
    def content_dimension(self) -> int:
        """Width of the genre+keyword block — the features that describe a film's content."""
        return len(self.genre_index) + len(self.keyword_index)

    @property
    def bias_position(self) -> int:
        return self.content_dimension + len(self.decade_index) + len(self.language_index)

    @property
    def dimension(self) -> int:
        return self.bias_position + 1


def build_feature_space(
    conn: sqlite3.Connection,
    *,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    min_keyword_films: int = DEFAULT_MIN_KEYWORD_FILMS,
) -> FeatureSpace:
    """Decide the feature columns from what the catalog actually contains.

    Genres, decades, and languages are all kept (small, controlled vocabularies); keywords are
    thousands-strong and long-tailed, so only those appearing in at least ``min_keyword_films``
    films qualify, capped at the ``max_keywords`` most frequent. A keyword seen on three films
    teaches the model nothing it can reuse.
    """
    genres = [row[0] for row in conn.execute("SELECT DISTINCT genre_id FROM movie_genres")]
    keyword_rows = conn.execute(
        "SELECT keyword_id FROM movie_keywords GROUP BY keyword_id "
        "HAVING COUNT(DISTINCT movie_id) >= ? ORDER BY COUNT(DISTINCT movie_id) DESC, keyword_id "
        "LIMIT ?",
        (min_keyword_films, max_keywords),
    ).fetchall()
    keywords = [row[0] for row in keyword_rows]
    decades = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT (year / 10) * 10 AS decade FROM movies "
            "WHERE year IS NOT NULL ORDER BY decade"
        )
    ]
    languages = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT original_language FROM movies "
            "WHERE original_language IS NOT NULL ORDER BY original_language"
        )
    ]

    base = 0
    genre_index = {gid: base + i for i, gid in enumerate(sorted(genres))}
    base += len(genre_index)
    keyword_index = {kid: base + i for i, kid in enumerate(keywords)}
    base += len(keyword_index)
    decade_index = {dec: base + i for i, dec in enumerate(sorted(decades))}
    base += len(decade_index)
    language_index = {lang: base + i for i, lang in enumerate(sorted(languages))}
    return FeatureSpace(genre_index, keyword_index, decade_index, language_index)


def feature_matrix(
    conn: sqlite3.Connection, movie_ids: list[int], space: FeatureSpace
) -> np.ndarray:
    """Build the dense one-hot feature matrix for ``movie_ids``, one row each, in that order.

    Bulk-reads the join tables once rather than querying per film, so it stays linear in the
    catalog. The bias column is always 1; every other column is 1 when the film has that genre,
    keyword, decade, or language and 0 otherwise.
    """
    row_of = {movie_id: i for i, movie_id in enumerate(movie_ids)}
    features = np.zeros((len(movie_ids), space.dimension), dtype=np.float32)
    if not movie_ids:
        return features
    features[:, space.bias_position] = 1.0

    placeholders = ",".join("?" for _ in movie_ids)
    ids = tuple(movie_ids)

    for movie_id, genre_id in conn.execute(
        f"SELECT movie_id, genre_id FROM movie_genres WHERE movie_id IN ({placeholders})", ids
    ):
        column = space.genre_index.get(genre_id)
        if column is not None:
            features[row_of[movie_id], column] = 1.0

    for movie_id, keyword_id in conn.execute(
        f"SELECT movie_id, keyword_id FROM movie_keywords WHERE movie_id IN ({placeholders})", ids
    ):
        column = space.keyword_index.get(keyword_id)
        if column is not None:
            features[row_of[movie_id], column] = 1.0

    for movie_id, year, language in conn.execute(
        f"SELECT movie_id, year, original_language FROM movies WHERE movie_id IN ({placeholders})",
        ids,
    ):
        row = row_of[movie_id]
        if year is not None:
            column = space.decade_index.get((year // 10) * 10)
            if column is not None:
                features[row, column] = 1.0
        if language is not None:
            column = space.language_index.get(language)
            if column is not None:
                features[row, column] = 1.0
    return features


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, *, alpha: float = DEFAULT_ALPHA
) -> np.ndarray:
    """Ridge regression by the normal equations: ``W = (XᵀX + αI)⁻¹ XᵀY``.

    ``features`` is ``n × d``, ``targets`` is ``n × tags``; the returned weights are ``d × tags``.
    Solving the ``d × d`` system (d bounded by the feature cap) is milliseconds, and the α on the
    diagonal both regularises and keeps the matrix invertible when two features always co-occur.
    """
    d = features.shape[1]
    gram = features.T @ features + alpha * np.eye(d, dtype=np.float64)
    rhs = features.T @ targets
    return np.linalg.solve(gram, rhs).astype(np.float32)


@dataclass(frozen=True)
class PredictionResult:
    """What a prediction pass did: how many films trained it, how many it filled in, its width."""

    trained_on: int
    predicted: int
    n_features: int


def _has_content_feature(features: np.ndarray, space: FeatureSpace) -> np.ndarray:
    """A boolean mask of rows with at least one genre or keyword set.

    Decade and language help *shape* a prediction but don't describe a film on their own, so a film
    with only those (and no genre or keyword) isn't worth predicting for — its vector would be
    little more than the decade's average. We fill in a film only when there's real content signal.
    """
    if space.content_dimension == 0:
        return np.zeros(features.shape[0], dtype=bool)
    return features[:, : space.content_dimension].any(axis=1)


def predict_missing(
    conn: sqlite3.Connection,
    genome_path,
    *,
    alpha: float = DEFAULT_ALPHA,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    min_keyword_films: int = DEFAULT_MIN_KEYWORD_FILMS,
) -> PredictionResult:
    """Fit on measured films and write a predicted vector for the featured-but-uncovered ones.

    Trains only on ``genome_source='measured'`` films (an existing predicted row never feeds the
    fit), predicts for ``genome_source='none'`` films that have at least one real feature, appends
    the predictions to the genome memmap, and flags those films ``predicted`` with their new row.
    Idempotent: any film already flagged ``predicted`` is reset to ``none`` first, so a re-run
    (for example after a genome rebuild) recomputes cleanly rather than orphaning rows.
    """
    with conn:
        conn.execute(
            "UPDATE movies SET genome_row = NULL, genome_source = 'none' "
            "WHERE genome_source = 'predicted'"
        )

    space = build_feature_space(
        conn, max_keywords=max_keywords, min_keyword_films=min_keyword_films
    )
    measured = conn.execute(
        "SELECT movie_id, genome_row FROM movies "
        "WHERE genome_source = 'measured' AND genome_row IS NOT NULL ORDER BY genome_row"
    ).fetchall()
    # Nothing to learn from, or no content features to predict from (a genome-only catalog with no
    # TMDB genres or keywords): there is nothing to fill in, so skip the fit entirely.
    if not measured or space.content_dimension == 0:
        return PredictionResult(trained_on=len(measured), predicted=0, n_features=space.dimension)

    measured_ids = [row[0] for row in measured]
    measured_rows = [row[1] for row in measured]
    matrix = genome.load_genome(genome_path)
    try:
        measured_vectors = np.asarray(matrix[measured_rows], dtype=np.float32)
    finally:
        del matrix  # release the read handle before we rewrite the file below
    x_train = feature_matrix(conn, measured_ids, space)
    weights = fit_ridge(x_train, measured_vectors, alpha=alpha)

    candidates = [
        row[0] for row in conn.execute("SELECT movie_id FROM movies WHERE genome_source = 'none'")
    ]
    x_all = feature_matrix(conn, candidates, space)
    mask = _has_content_feature(x_all, space)
    target_ids = [movie_id for movie_id, keep in zip(candidates, mask, strict=True) if keep]
    if not target_ids:
        return PredictionResult(len(measured_ids), 0, space.dimension)
    predictions = np.clip(x_all[mask] @ weights, 0.0, 1.0).astype(np.float32)

    # Rewrite the genome as [measured; predicted] so the file stays exactly the rows we reference —
    # measured first (their order preserved), predictions after — and re-map every genome_row to
    # its new index. Rewriting (not appending) is what makes a re-run idempotent.
    combined = np.vstack([measured_vectors, predictions]).astype(np.float32)
    genome.write_matrix(genome_path, combined)
    updates = [(i, movie_id) for i, movie_id in enumerate(measured_ids)]
    updates += [(len(measured_ids) + k, movie_id) for k, movie_id in enumerate(target_ids)]
    with conn:
        conn.execute("UPDATE movies SET genome_row = NULL WHERE genome_source = 'measured'")
        conn.executemany("UPDATE movies SET genome_row = ? WHERE movie_id = ?", updates)
        conn.executemany(
            "UPDATE movies SET genome_source = 'predicted' WHERE movie_id = ?",
            [(movie_id,) for movie_id in target_ids],
        )
    return PredictionResult(len(measured_ids), len(target_ids), space.dimension)
