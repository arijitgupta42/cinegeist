"""The honesty check: say so when the demo shard can't serve the user (plan.md §8.4).

The browser demo searches a 2,000-film shard, not the full ~16,000-film catalog. A visitor whose
taste points into a thinly-sampled corner has to be *told*, in numbers, rather than quietly handed
padding — that dishonesty is the exact failure the plan forbids (hard rule 9). This module is the
deterministic reference for that decision: given the profile centroid, the shard vectors near it,
and each film's per-film coverage byte (computed at shard-build time, §8.4), it measures how much
of the region survived and whether to take the honesty path.

It is pure numpy in, plain values out — no catalog, no shard format — so the browser's
``coverage.ts`` mirrors it exactly against the shared ``spec/`` fixtures. The thresholds are the
canonical values in ``spec/constants.json``; a test asserts these module constants still match.

The two triggers (either one fires the honesty path):

* ``region_coverage`` below :data:`REGION_COVERAGE_MIN` — averaged over the nearest shard films,
  weighted by closeness, the shard kept under a quarter of this neighbourhood.
* ``nearest_cosine`` below :data:`NEAREST_COSINE_MIN` — the centroid has no close neighbour in the
  shard at all, so even the top pick is a stretch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..catalog import genome

# A film counts as inside another's neighbourhood at or above this cosine. Used at build time to
# size each film's full-catalog vs in-shard neighbourhood (the coverage byte), and named here so
# the definition lives in one place. (plan.md §8.4)
COVERAGE_SIMILARITY = 0.85

# How many of the nearest shard films the region-coverage average is taken over.
COVERAGE_TOP_K = 25

# Take the honesty path when the weighted region coverage is below this...
REGION_COVERAGE_MIN = 0.25
# ...or when the single closest shard film is further than this cosine from the centroid.
NEAREST_COSINE_MIN = 0.45

# Reason slugs, in the fixed order they are reported, so both implementations agree byte-for-byte.
REASON_THIN_REGION = "region_coverage_below_min"
REASON_NO_CLOSE_NEIGHBOUR = "no_close_neighbour"


@dataclass(frozen=True)
class CoverageVerdict:
    """Whether to take the honesty path, the two measurements behind it, and why.

    ``honest`` is ``True`` when the shard is judged unable to serve this taste well. ``reasons``
    lists which trigger(s) fired, always region-coverage before nearest-neighbour, and is empty
    when ``honest`` is ``False``. The raw ``region_coverage`` and ``nearest_cosine`` are carried
    through so the caller can state the actual numbers to the user.
    """

    honest: bool
    region_coverage: float
    nearest_cosine: float
    reasons: tuple[str, ...]


def nearest_cosine(centroid: np.ndarray, vectors: np.ndarray) -> float:
    """The highest cosine from the centroid to any shard film (0.0 for an empty shard)."""
    if vectors.shape[0] == 0:
        return 0.0
    return float(genome.cosine_scores(vectors, centroid).max())


def region_coverage(
    centroid: np.ndarray,
    vectors: np.ndarray,
    coverage: np.ndarray,
    *,
    top_k: int = COVERAGE_TOP_K,
) -> float:
    """Closeness-weighted mean coverage over the ``top_k`` shard films nearest ``centroid``.

    ``coverage[i]`` is film ``i``'s coverage fraction in ``[0, 1]`` (the byte, rescaled). The weight
    of a film is its cosine to the centroid, floored at zero so a film pointing *away* from the
    taste (possible once vectors are SVD-compressed and can go negative) neither helps nor hurts.
    Returns 0.0 — the honest, pessimistic reading — when nothing near the centroid has positive
    weight, or the shard is empty.
    """
    n = vectors.shape[0]
    if n == 0:
        return 0.0
    cosines = genome.cosine_scores(vectors, centroid)
    k = min(top_k, n)
    # The k nearest by cosine; argpartition is enough since we only need the set, not its order.
    nearest = np.argpartition(cosines, n - k)[n - k :] if k < n else np.arange(n)
    weights = np.clip(cosines[nearest], 0.0, None)
    total = float(weights.sum())
    if total == 0.0:
        return 0.0
    return float((weights * np.asarray(coverage, dtype=np.float64)[nearest]).sum() / total)


def honesty_reasons(region_cov: float, top1_cosine: float) -> tuple[str, ...]:
    """Which honesty triggers fire for these two measurements, in the canonical report order."""
    reasons: list[str] = []
    if region_cov < REGION_COVERAGE_MIN:
        reasons.append(REASON_THIN_REGION)
    if top1_cosine < NEAREST_COSINE_MIN:
        reasons.append(REASON_NO_CLOSE_NEIGHBOUR)
    return tuple(reasons)


def assess(
    centroid: np.ndarray,
    vectors: np.ndarray,
    coverage: np.ndarray,
    *,
    top_k: int = COVERAGE_TOP_K,
) -> CoverageVerdict:
    """Measure the region and decide whether the demo should take the honesty path (§8.4).

    This is the whole check in one call, and the shape the browser consumes: compute the weighted
    region coverage and the nearest-neighbour cosine, then report honesty when either trigger fires.
    """
    region_cov = region_coverage(centroid, vectors, coverage, top_k=top_k)
    top1 = nearest_cosine(centroid, vectors)
    reasons = honesty_reasons(region_cov, top1)
    return CoverageVerdict(
        honest=bool(reasons),
        region_coverage=region_cov,
        nearest_cosine=top1,
        reasons=reasons,
    )
