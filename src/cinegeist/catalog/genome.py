"""The tag genome as a numpy memmap.

The genome is an ``n_movies × n_tags`` matrix of relevance scores in ``[0, 1]`` — the taste
space the whole recommender works in. It is stored as a float32 ``.npy`` at
``data/genome.npy`` (a real numpy file with a header, so ``np.load`` reads it back), roughly
59 MB at full size. A recommendation is cosine similarity against this matrix, which is a
single matmul in a few milliseconds — hence no FAISS, no index, no extra dependency.

This module owns only the numeric artifact: turning streamed ``(movie, tag, score)`` triples
into the matrix, and loading it back. The mapping between matrix rows and ``movies.genome_row``
is applied by ``build.py``; the mapping between matrix columns and tags is ``genome_tags.position``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

# float32 is plenty for relevance scores and halves the file versus float64.
DTYPE = np.float32

GENOME_FILENAME = "genome.npy"

# (movie_id, column_position, relevance) — scores already mapped onto memmap columns.
PositionedScore = tuple[int, int, float]


def default_genome_path(data_dir: Path) -> Path:
    """Where the genome memmap lives, given the catalog's data directory."""
    return data_dir / GENOME_FILENAME


def build_memmap(
    out_path: Path,
    n_tags: int,
    positioned_scores: Iterable[PositionedScore],
    *,
    on_row: Callable[[], None] | None = None,
) -> dict[int, int]:
    """Write the genome matrix and return the ``{movie_id: row_index}`` mapping.

    ``positioned_scores`` yields one triple per (movie, tag) relevance, with the tag already
    resolved to its memmap column. Scores are accumulated per film into dense rows, then
    written in ascending ``movie_id`` order so the row layout is deterministic regardless of
    input ordering. ``on_row`` is pinged once per input triple, for progress reporting.

    The matrix is written to a temporary file and atomically renamed into place, so an
    interrupted build never leaves a half-written ``genome.npy`` that looks complete.
    """
    vectors: dict[int, np.ndarray] = {}
    for movie_id, position, relevance in positioned_scores:
        row = vectors.get(movie_id)
        if row is None:
            row = np.zeros(n_tags, dtype=DTYPE)
            vectors[movie_id] = row
        row[position] = relevance
        if on_row is not None:
            on_row()

    movie_ids = sorted(vectors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    matrix = open_memmap(tmp_path, mode="w+", dtype=DTYPE, shape=(len(movie_ids), n_tags))
    row_of: dict[int, int] = {}
    try:
        for index, movie_id in enumerate(movie_ids):
            matrix[index] = vectors[movie_id]
            row_of[movie_id] = index
        matrix.flush()
    finally:
        del matrix  # release the memmap handle before the rename (Windows needs this)
    os.replace(tmp_path, out_path)
    return row_of


def load_genome(path: Path) -> np.memmap:
    """Open the genome read-only as a memmap. Values are not copied into RAM."""
    return np.load(path, mmap_mode="r")


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of ``matrix`` against ``query``.

    The whole point of keeping the genome as one dense matrix: this is a single matmul plus a
    norm, milliseconds over the full catalog, no index required. Rows with zero magnitude (and
    a zero query) score 0 rather than NaN.
    """
    q = np.asarray(query, dtype=DTYPE)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=DTYPE)
    row_norms = np.linalg.norm(matrix, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (matrix @ q) / (row_norms * q_norm)
    scores[~np.isfinite(scores)] = 0.0
    return scores
