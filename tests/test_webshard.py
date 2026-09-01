"""Unit tests for the browser-shard build (plan.md §8.3).

The pure helpers — sampling, SVD projection, int8 quantisation, PCA, top tags, coverage, and probe
grounding — are tested against tiny synthetic matrices so the whole pipeline runs in CI without the
59 MB genome. A second group validates the *committed* shard and probe artifacts (structure, size,
and referential integrity) rather than rebuilding them, since CI has no catalog to rebuild from.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.convo.probes import phrase_pair
from cinegeist.webshard import build

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = REPO_ROOT / "web" / "public" / "shard"


# -- sampling ------------------------------------------------------------------------


def test_stratified_sample_hits_target_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    years = rng.integers(1950, 2024, size=500)
    pops = rng.random(500) * 100
    a = build.stratified_sample(years, pops, 120, seed=3)
    b = build.stratified_sample(years, pops, 120, seed=3)
    assert len(a) == 120
    assert np.array_equal(a, b)  # same seed → identical sample
    assert len(np.unique(a)) == 120  # no duplicates


def test_stratified_sample_spreads_across_decades_and_keeps_obscure() -> None:
    years = np.array([1975] * 200 + [2015] * 800)  # a small old decade, a huge modern one
    pops = np.concatenate([np.linspace(0, 1, 200), np.linspace(0, 100, 800)])
    sample = build.stratified_sample(years, pops, 200, seed=1)
    decades = {(years[i] // 10) * 10 for i in sample}
    assert decades == {1970, 2010}  # the tiny decade is not erased by the huge one
    # The low-popularity tail survives rather than only the hits being kept.
    assert pops[sample].min() < 5.0


def test_sample_returns_everything_when_target_exceeds_pool() -> None:
    years = np.array([2000, 2001, 2002])
    pops = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(build.stratified_sample(years, pops, 10), np.arange(3))


# -- compression ---------------------------------------------------------------------


def test_svd_project_shapes_and_reconstruction() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.random((40, 12)).astype(np.float32)
    projected, components = build.svd_project(matrix, 6)
    assert projected.shape == (40, 6)
    assert components.shape == (12, 6)
    # Projecting then lifting back through the orthonormal basis approximates the original rows.
    reconstructed = projected @ components.T
    assert np.linalg.norm(reconstructed - matrix) < np.linalg.norm(matrix)


def test_svd_projection_is_sign_stable() -> None:
    matrix = np.random.default_rng(4).random((30, 10)).astype(np.float32)
    a, _ = build.svd_project(matrix, 5)
    b, _ = build.svd_project(matrix, 5)
    assert np.allclose(a, b)  # canonicalised signs → reproducible bytes


def test_quantise_int8_round_trips_within_a_step() -> None:
    rng = np.random.default_rng(5)
    projected = (rng.random((50, 8)) - 0.5) * 20
    q, scales = build.quantise_int8(projected)
    assert q.dtype == np.int8
    assert q.min() >= -127 and q.max() <= 127
    assert np.all(scales > 0)
    error = np.abs(q.astype(np.float32) * scales - projected)
    assert np.all(error <= scales + 1e-6)  # within one quantisation step per component


def test_pca_3d_is_three_dimensional_and_centred() -> None:
    matrix = np.random.default_rng(6).random((25, 9)).astype(np.float32)
    xyz = build.pca_3d(matrix)
    assert xyz.shape == (25, 3)
    assert np.allclose(xyz.mean(axis=0), 0.0, atol=1e-4)  # PCA is mean-centred


def test_top_tags_takes_the_strongest_and_drops_zeros() -> None:
    vector = np.array([0.0, 0.8, 0.1, 0.0, 0.4], dtype=np.float32)
    tags = build.top_tags(vector, k=12)
    assert [pos for pos, _ in tags] == [1, 4, 2]  # descending relevance, zeros dropped
    assert tags[0][1] == 204  # 0.8 rescaled to a 0–255 byte


# -- coverage ------------------------------------------------------------------------


def test_per_film_coverage_measures_the_kept_neighbourhood() -> None:
    # Two tight clusters, A (rows 0-2) and B (rows 3-5); cross-cluster cosine is ~0.
    full = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.10, 0.0, 0.0],
            [0.98, 0.15, 0.0, 0.0],  # A
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.99, 0.10],
            [0.0, 0.0, 0.98, 0.15],  # B
        ],
        dtype=np.float32,
    )
    # The shard keeps one A film and all three B films.
    coverage = build.per_film_coverage(full, np.array([0, 3, 4, 5]))
    assert coverage.dtype == np.uint8
    # A film 0's neighbourhood is the 3 A films but only itself survived → 1/3 ≈ 85/255.
    assert coverage[0] == round(255 / 3)
    # Every B film's whole neighbourhood survived → fully covered.
    assert list(coverage[1:]) == [255, 255, 255]


def test_per_film_coverage_is_never_zero_for_an_isolated_film() -> None:
    full = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # orthogonal, no neighbours
    coverage = build.per_film_coverage(full, np.array([0, 1]))
    assert list(coverage) == [255, 255]  # each counts only itself, in both → coverage 1


# -- the whole build on a synthetic catalog ------------------------------------------


def _synthetic_catalog() -> tuple[sqlite3.Connection, np.ndarray]:
    conn = db.connect(":memory:")
    db.migrate(conn)
    n_tags = 8
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(i + 1, i, f"tag{i}") for i in range(n_tags)],
    )
    rng = np.random.default_rng(11)
    matrix = rng.random((24, n_tags)).astype(np.float32)
    conn.executemany(
        "INSERT INTO movies (movie_id, tmdb_id, title, clean_title, year, runtime, poster_path, "
        "popularity, genome_row, genome_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'measured')",
        [
            (
                100 + i,
                900 + i,
                f"Film {i} ({1980 + i})",
                f"Film {i}",
                1980 + i,
                90 + i,
                f"/p{i}.jpg",
                float(i),
                i,
            )
            for i in range(24)
        ],
    )
    conn.commit()
    return conn, matrix


def test_build_shard_produces_a_decodable_bundle() -> None:
    conn, matrix = _synthetic_catalog()
    result = build.build_shard(conn, matrix, target=10, components=4, full_catalog_size=24)
    m = result.manifest

    assert m["version"] == build.SHARD_VERSION
    assert m["n_films"] == 10
    assert m["n_components"] == 4
    assert m["full_catalog_size"] == 24
    assert len(m["scales"]) == 4
    assert len(m["films"]) == 10
    assert m["tag_names"], "tag_names should cover the positions used by top tags"

    # The binary decodes back to the shapes the manifest promises.
    sections = {s["name"]: s for s in m["binary"]["sections"]}
    assert set(sections) == {"vectors", "xyz", "tag_pos", "tag_score", "coverage"}
    assert len(result.binary) == sum(s["length"] for s in sections.values())
    vec = sections["vectors"]
    block = result.binary[vec["offset"] : vec["offset"] + vec["length"]]
    decoded = np.frombuffer(block, dtype=np.int8).reshape(vec["shape"])
    assert decoded.shape == (10, 4)

    cov = sections["coverage"]
    cov_block = result.binary[cov["offset"] : cov["offset"] + cov["length"]]
    cov_bytes = np.frombuffer(cov_block, np.uint8)
    assert cov_bytes.shape == (10,) and cov_bytes.min() >= 1  # a film always covers itself


def test_build_shard_rejects_an_empty_catalog() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn)
    with pytest.raises(ValueError):
        build.build_shard(conn, np.zeros((0, 8), dtype=np.float32))


# -- precomputed probes --------------------------------------------------------------


def test_build_probes_grounds_real_pairs_ordered_by_spread() -> None:
    conn, matrix = _synthetic_catalog()
    result = build.build_probes(conn, matrix, target=10, n_axes=5)

    assert result["question_template"] == build.PROBE_QUESTION_TEMPLATE
    probes = result["probes"]
    assert 0 < len(probes) <= 5
    spreads = [p["spread"] for p in probes]
    assert spreads == sorted(spreads, reverse=True)  # most discriminative first
    for p in probes:
        assert p["high"]["id"] != p["low"]["id"]  # a real contrast, not a film against itself
        assert p["question"] == phrase_pair(p["high"]["title"], p["low"]["title"])


def test_build_probes_is_deterministic() -> None:
    conn, matrix = _synthetic_catalog()
    a = build.build_probes(conn, matrix, target=10, n_axes=5)["probes"]
    conn2, matrix2 = _synthetic_catalog()
    b = build.build_probes(conn2, matrix2, target=10, n_axes=5)["probes"]
    assert [p["axis"] for p in a] == [p["axis"] for p in b]


# -- the committed artifact ----------------------------------------------------------


@pytest.mark.skipif(
    not (SHARD_DIR / "shard.json").exists(), reason="run `make web-shard` to build the shard"
)
def test_committed_shard_is_well_formed_and_within_budget() -> None:
    manifest = json.loads((SHARD_DIR / "shard.json").read_text(encoding="utf-8"))
    binary = (SHARD_DIR / "shard.bin").read_bytes()

    assert manifest["n_films"] == len(manifest["films"]) == build.TARGET_FILMS
    assert manifest["n_components"] == build.SVD_COMPONENTS
    assert len(manifest["scales"]) == build.SVD_COMPONENTS
    assert manifest["full_catalog_size"] > manifest["n_films"]
    for film in manifest["films"]:
        assert film["id"] and film["title"]

    sections = {s["name"]: s for s in manifest["binary"]["sections"]}
    assert len(binary) == sum(s["length"] for s in sections.values())
    assert sections["vectors"]["shape"] == [build.TARGET_FILMS, build.SVD_COMPONENTS]

    # Coverage bytes are present and a real measurement — a genuine spread, not padded to
    # fully-covered, and every film covers at least itself (the honesty signal, §8.4, hard rule 9).
    cov = sections["coverage"]
    assert cov["shape"] == [build.TARGET_FILMS]
    cov_bytes = np.frombuffer(binary[cov["offset"] : cov["offset"] + cov["length"]], np.uint8)
    assert cov_bytes.min() >= 1
    assert cov_bytes.min() < cov_bytes.max(), "coverage should vary across the shard"
    assert (cov_bytes < 64).any(), "some films should land in thin regions (< 0.25 coverage)"

    gz = len(gzip.compress(binary, 9)) + len(
        gzip.compress((SHARD_DIR / "shard.json").read_bytes(), 9)
    )
    assert gz < 400 * 1024, f"shard is {gz / 1024:.0f} KB gzipped, over the 400 KB budget"


@pytest.mark.skipif(
    not (SHARD_DIR / "probes.json").exists(), reason="run `make web-shard` to build the shard"
)
def test_committed_probes_are_well_formed_and_reference_shard_films() -> None:
    probes_doc = json.loads((SHARD_DIR / "probes.json").read_text(encoding="utf-8"))
    manifest = json.loads((SHARD_DIR / "shard.json").read_text(encoding="utf-8"))
    shard_ids = {f["id"] for f in manifest["films"]}

    probes = probes_doc["probes"]
    assert probes and len(probes) <= build.PROBE_AXES
    assert probes_doc["question_template"] == build.PROBE_QUESTION_TEMPLATE
    seen_axes = set()
    for p in probes:
        assert p["axis"] not in seen_axes, "each axis appears at most once"
        seen_axes.add(p["axis"])
        assert p["high"]["id"] != p["low"]["id"]
        # Every probe film is in the shard, so the demo can look it up (referential integrity).
        assert p["high"]["id"] in shard_ids and p["low"]["id"] in shard_ids
        assert p["question"] == phrase_pair(p["high"]["title"], p["low"]["title"])
