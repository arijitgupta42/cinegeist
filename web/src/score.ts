// Score, diversify, and shape the candidate pool into picks — a TypeScript port of
// `cinegeist.recommend.score` (plan.md §6, §8.6). Every weight and threshold comes from
// spec/constants.json via constants.ts, and every function here is checked against the same
// scoring/cases.json fixtures the Python suite asserts, so the two scorers cannot drift.
//
// The combined score, per film:
//
//   score = ( W_COSINE   × cosine(profile, film)
//           + W_QUALITY  × quality               (Bayesian-shrunk rating)
//           + W_SESSION  × cosine(session, film)
//           + W_FACET    × facet                 (0 in the demo — no credits in the shard)
//           − W_POP      × popularity_penalty )
//           × confidence                         (discount predicted vectors)
//
// then MMR spreads the top for diversity and the wildcard reaches deliberately outside the centroid.

import { SCORING } from "./constants.ts";
import { cosineScores, rowSlice } from "./math.ts";

export interface Candidate {
  movieId: number;
  title?: string;
  year?: number | null;
  voteAverage?: number | null;
  voteCount?: number | null;
  popularity?: number | null;
  genomeSource?: string;
}

export interface ScoredFilm {
  movieId: number;
  poolIndex: number;
  title: string;
  year: number | null;
  score: number;
  cosine: number;
  quality: number;
  sessionFit: number;
  facetMatch: number;
  popularityPenalty: number;
  confidence: number;
}

export interface Recommendations {
  picks: ScoredFilm[];
  wildcard: ScoredFilm | null;
  shortlist: ScoredFilm[];
}

/** A rating shrunk toward the global mean by a pseudo-count, normalised to [0, 1]. */
export function bayesianQuality(voteAverage: number | null | undefined, voteCount: number | null | undefined): number {
  if (voteAverage === null || voteAverage === undefined) {
    return SCORING.QUALITY_PRIOR_MEAN / SCORING.RATING_SCALE; // an unrated film reads as the neutral prior
  }
  const votes = voteCount ?? 0;
  const shrunk =
    (votes * voteAverage + SCORING.QUALITY_PRIOR_COUNT * SCORING.QUALITY_PRIOR_MEAN) /
    (votes + SCORING.QUALITY_PRIOR_COUNT);
  return Math.min(1, Math.max(0, shrunk / SCORING.RATING_SCALE));
}

/** The discovery nudge: log1p(popularity) scaled to [0, 1] against a reference; 0 when missing. */
export function popularityPenalty(popularity: number | null | undefined): number {
  if (popularity === null || popularity === undefined || popularity <= 0) return 0;
  return Math.min(1, Math.log1p(popularity) / Math.log1p(SCORING.POPULARITY_REFERENCE));
}

/** Trust a measured genome vector fully; discount a predicted one. */
export function confidenceFor(genomeSource: string | undefined): number {
  return genomeSource === "predicted" ? SCORING.PREDICTED_CONFIDENCE : 1;
}

interface ScoreOptions {
  sessionVector?: ArrayLike<number> | null;
  facetScores?: ArrayLike<number> | null;
}

function anyNonZero(v: ArrayLike<number>): boolean {
  for (let i = 0; i < v.length; i++) if (v[i] !== 0) return true;
  return false;
}

/** Score every candidate, returned in input order (`poolIndex` === list index). */
export function scorePool(
  candidates: Candidate[],
  vectors: ArrayLike<number>,
  nCols: number,
  profile: ArrayLike<number>,
  opts: ScoreOptions = {},
): ScoredFilm[] {
  const n = candidates.length;
  if (n === 0) return [];
  const cosine = cosineScores(vectors, nCols, profile);
  const session =
    opts.sessionVector && anyNonZero(opts.sessionVector)
      ? cosineScores(vectors, nCols, opts.sessionVector)
      : new Float64Array(n);

  const scored: ScoredFilm[] = [];
  for (let i = 0; i < n; i++) {
    const film = candidates[i];
    const quality = bayesianQuality(film.voteAverage, film.voteCount);
    const popPenalty = popularityPenalty(film.popularity);
    const facet = opts.facetScores ? opts.facetScores[i] : 0;
    const confidence = confidenceFor(film.genomeSource);
    const combined =
      (SCORING.W_COSINE * cosine[i] +
        SCORING.W_QUALITY * quality +
        SCORING.W_SESSION * session[i] +
        SCORING.W_FACET * facet -
        SCORING.W_POPULARITY * popPenalty) *
      confidence;
    scored.push({
      movieId: film.movieId,
      poolIndex: i,
      title: film.title ?? `Film ${film.movieId}`,
      year: film.year ?? null,
      score: combined,
      cosine: cosine[i],
      quality,
      sessionFit: session[i],
      facetMatch: facet,
      popularityPenalty: popPenalty,
      confidence,
    });
  }
  return scored;
}

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** Reorder for relevance *and* diversity (MMR), returning the top `k` (plan.md §6.3). */
export function mmrRank(
  scored: ScoredFilm[],
  vectors: ArrayLike<number>,
  nCols: number,
  lam: number = SCORING.MMR_LAMBDA,
  k?: number,
): ScoredFilm[] {
  if (scored.length === 0) return [];
  const limit = k === undefined ? scored.length : Math.min(k, scored.length);

  // The candidate vectors reordered to `scored` order, flat and row-major.
  const svecs = new Float64Array(scored.length * nCols);
  for (let r = 0; r < scored.length; r++) {
    const src = scored[r].poolIndex * nCols;
    const dst = r * nCols;
    for (let j = 0; j < nCols; j++) svecs[dst + j] = vectors[src + j];
  }

  const raw = scored.map((f) => f.score);
  const min = Math.min(...raw);
  const max = Math.max(...raw);
  const span = max - min;
  const relevance = raw.map((v) => (span > 0 ? (v - min) / span : 1));

  const maxSim = new Float64Array(scored.length); // closeness of each film to the picked set
  const chosen = new Array<boolean>(scored.length).fill(false);
  const order: number[] = [];
  for (let step = 0; step < limit; step++) {
    const value = new Float64Array(scored.length);
    for (let i = 0; i < scored.length; i++) {
      value[i] = chosen[i] ? -Infinity : lam * relevance[i] - (1 - lam) * maxSim[i];
    }
    const pick = argmax(value);
    order.push(pick);
    chosen[pick] = true;
    const sims = cosineScores(svecs, nCols, rowSlice(svecs, nCols, pick));
    for (let i = 0; i < scored.length; i++) maxSim[i] = Math.max(maxSim[i], sims[i]);
  }
  return order.map((i) => scored[i]);
}

/**
 * The exploration slot: the best film far from taste that still shares real tags (§6.5).
 *
 * `relevanceAt(poolIndex, position)` is how strongly a candidate loads on a genome tag — the film's
 * own vector column in the spec fixtures, and its top-tag table in the demo (whose scoring vectors
 * are SVD-compressed and can't be indexed by genome position). Everything else is the film's
 * already-computed cosine and score, so both callers share one implementation.
 */
export function selectWildcard(
  scored: ScoredFilm[],
  relevanceAt: (poolIndex: number, position: number) => number,
  strongTagPositions: Iterable<number>,
  excludeMovieIds: Set<number> = new Set(),
): ScoredFilm | null {
  const positions = [...strongTagPositions];
  if (scored.length === 0 || positions.length === 0) return null;

  let best: ScoredFilm | null = null;
  for (const film of scored) {
    if (excludeMovieIds.has(film.movieId)) continue;
    if (film.cosine > SCORING.WILDCARD_MAX_COSINE) continue;
    let shared = 0;
    for (const pos of positions) {
      if (relevanceAt(film.poolIndex, pos) >= SCORING.WILDCARD_TAG_RELEVANCE) shared++;
    }
    if (shared < SCORING.WILDCARD_MIN_SHARED_TAGS) continue;
    if (best === null || film.score > best.score) best = film;
  }
  return best;
}

export interface RecommendOptions {
  strongTagPositions?: Iterable<number>;
  sessionVector?: ArrayLike<number> | null;
  facetScores?: ArrayLike<number> | null;
  nConfident?: number;
  shortlistSize?: number;
  lam?: number;
  withWildcard?: boolean;
  // How the wildcard reads a candidate's tag relevance; defaults to the film's own vector column,
  // which is what the spec fixtures expect. The demo overrides it with its top-tag table.
  wildcardRelevanceAt?: (poolIndex: number, position: number) => number;
}

/** Score → diversify → shape into `nConfident` picks plus one wildcard, no LLM involved. */
export function recommend(
  candidates: Candidate[],
  vectors: ArrayLike<number>,
  nCols: number,
  profile: ArrayLike<number>,
  opts: RecommendOptions = {},
): Recommendations {
  const scored = scorePool(candidates, vectors, nCols, profile, {
    sessionVector: opts.sessionVector,
    facetScores: opts.facetScores,
  });
  if (scored.length === 0) return { picks: [], wildcard: null, shortlist: [] };

  const shortlist = mmrRank(scored, vectors, nCols, opts.lam ?? SCORING.MMR_LAMBDA, opts.shortlistSize ?? 40);
  const picks = shortlist.slice(0, opts.nConfident ?? 3);

  let wildcard: ScoredFilm | null = null;
  if (opts.withWildcard ?? true) {
    const relevanceAt = opts.wildcardRelevanceAt ?? ((poolIndex: number, position: number) => vectors[poolIndex * nCols + position]);
    const pickedIds = new Set(picks.map((f) => f.movieId));
    wildcard = selectWildcard(scored, relevanceAt, opts.strongTagPositions ?? [], pickedIds);
  }
  return { picks, wildcard, shortlist };
}
