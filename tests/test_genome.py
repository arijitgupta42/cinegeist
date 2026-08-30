"""Unit tests for the genome memmap: building it, reading it back, and its guarantees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cinegeist.catalog import genome


def test_build_memmap_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "genome.npy"
    scores = [(10, 0, 0.5), (10, 1, 0.25), (20, 0, 0.1), (20, 2, 0.9)]
    row_of = genome.build_memmap(out, 3, iter(scores))

    assert row_of == {10: 0, 20: 1}
    matrix = genome.load_genome(out)
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix[0], [0.5, 0.25, 0.0])
    assert np.allclose(matrix[1], [0.1, 0.0, 0.9])  # unfilled tag stays zero


def test_build_memmap_orders_rows_by_movie_id(tmp_path: Path) -> None:
    out = tmp_path / "genome.npy"
    # Deliberately out of order; the row layout must not depend on input order.
    row_of = genome.build_memmap(out, 1, iter([(30, 0, 0.3), (10, 0, 0.1), (20, 0, 0.2)]))
    assert row_of == {10: 0, 20: 1, 30: 2}
    assert np.allclose(genome.load_genome(out)[:, 0], [0.1, 0.2, 0.3])


def test_build_memmap_leaves_no_temp_file(tmp_path: Path) -> None:
    out = tmp_path / "genome.npy"
    genome.build_memmap(out, 2, iter([(1, 0, 1.0), (1, 1, 0.0)]))
    assert out.exists()
    assert not out.with_name(out.name + ".tmp").exists()


def test_build_memmap_reports_each_row(tmp_path: Path) -> None:
    seen = 0

    def tick() -> None:
        nonlocal seen
        seen += 1

    genome.build_memmap(
        tmp_path / "g.npy", 2, iter([(1, 0, 1.0), (1, 1, 0.5), (2, 0, 0.2)]), on_row=tick
    )
    assert seen == 3


def test_default_genome_path(tmp_path: Path) -> None:
    assert genome.default_genome_path(tmp_path) == tmp_path / "genome.npy"


def test_loaded_genome_is_not_writable(tmp_path: Path) -> None:
    out = tmp_path / "genome.npy"
    genome.build_memmap(out, 1, iter([(1, 0, 0.7)]))
    matrix = genome.load_genome(out)
    # Opened read-only, so the on-disk artifact can't be corrupted by a stray write.
    assert matrix.flags.writeable is False


def test_cosine_scores_ranks_by_direction() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    scores = genome.cosine_scores(matrix, np.array([1.0, 0.0], dtype=np.float32))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-5)
    assert int(np.argmax(scores)) == 0


def test_cosine_scores_handles_zero_query() -> None:
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    assert list(genome.cosine_scores(matrix, np.zeros(2, dtype=np.float32))) == [0.0]


def test_cosine_scores_zero_row_is_zero_not_nan() -> None:
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    scores = genome.cosine_scores(matrix, np.array([1.0, 0.0], dtype=np.float32))
    assert scores[0] == 0.0
    assert np.all(np.isfinite(scores))
