// The honesty check: say so when the demo shard can't serve the user (plan.md §8.4) — a port of
// `cinegeist.recommend.coverage`, checked against spec/scoring/coverage.json. The demo searches a
// 2,000-film shard, not the full ~16,000-film catalog; a visitor whose taste points into a thinly
// sampled corner has to be *told*, in numbers, rather than quietly handed padding (hard rule 9).
//
// Two triggers, either one fires the honesty path: the weighted coverage of the region near the
// profile falls below a floor, or the centroid has no close neighbour in the shard at all. Pure
// numbers in, plain values out — no shard format, no catalog — so it mirrors the Python exactly.

import { COVERAGE } from "./constants.ts";
import { cosineScores } from "./math.ts";

// Reason slugs, in the fixed order they are reported, so both implementations agree byte-for-byte.
export const REASON_THIN_REGION = "region_coverage_below_min";
export const REASON_NO_CLOSE_NEIGHBOUR = "no_close_neighbour";

export interface CoverageVerdict {
  honest: boolean;
  regionCoverage: number;
  nearestCosine: number;
  reasons: string[];
}

/** The highest cosine from the centroid to any shard film (0 for an empty shard). */
export function nearestCosine(centroid: ArrayLike<number>, vectors: ArrayLike<number>, nCols: number): number {
  const rows = nCols === 0 ? 0 : vectors.length / nCols;
  if (rows === 0) return 0;
  const cos = cosineScores(vectors, nCols, centroid);
  let max = cos[0];
  for (let i = 1; i < cos.length; i++) if (cos[i] > max) max = cos[i];
  return max;
}

/** Closeness-weighted mean coverage over the `topK` shard films nearest the centroid (§8.4). */
export function regionCoverage(
  centroid: ArrayLike<number>,
  vectors: ArrayLike<number>,
  nCols: number,
  coverage: ArrayLike<number>,
  topK: number = COVERAGE.COVERAGE_TOP_K,
): number {
  const n = nCols === 0 ? 0 : vectors.length / nCols;
  if (n === 0) return 0;
  const cos = cosineScores(vectors, nCols, centroid);
  const k = Math.min(topK, n);

  // The k nearest by cosine — just the set, order doesn't matter for the weighted mean.
  const nearest = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => cos[b] - cos[a] || a - b)
    .slice(0, k);

  let total = 0;
  let weighted = 0;
  for (const i of nearest) {
    const w = Math.max(0, cos[i]); // a film pointing away from the taste neither helps nor hurts
    total += w;
    weighted += w * coverage[i];
  }
  return total === 0 ? 0 : weighted / total;
}

/** Which honesty triggers fire for these two measurements, in the canonical report order. */
export function honestyReasons(regionCov: number, top1Cosine: number): string[] {
  const reasons: string[] = [];
  if (regionCov < COVERAGE.REGION_COVERAGE_MIN) reasons.push(REASON_THIN_REGION);
  if (top1Cosine < COVERAGE.NEAREST_COSINE_MIN) reasons.push(REASON_NO_CLOSE_NEIGHBOUR);
  return reasons;
}

/** Measure the region and decide whether the demo should take the honesty path (§8.4). */
export function assess(
  centroid: ArrayLike<number>,
  vectors: ArrayLike<number>,
  nCols: number,
  coverage: ArrayLike<number>,
  topK: number = COVERAGE.COVERAGE_TOP_K,
): CoverageVerdict {
  const region = regionCoverage(centroid, vectors, nCols, coverage, topK);
  const top1 = nearestCosine(centroid, vectors, nCols);
  const reasons = honestyReasons(region, top1);
  return { honest: reasons.length > 0, regionCoverage: region, nearestCosine: top1, reasons };
}
