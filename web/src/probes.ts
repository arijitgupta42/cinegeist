// Choose the next question by how much it teaches us, and decide when to stop — a TypeScript port
// of `cinegeist.convo.probes` (plan.md §5.2, §5.3, §8.6). Two layers live here:
//
//   1. The spec-faithful algorithm — `chooseProbe`, `groundPair`, `shouldStop`, `wantsToStop` —
//      checked against scoring/probes.json, the same fixtures the Python suite asserts. These work
//      on full genome vectors, exactly like the CLI.
//
//   2. The demo's runtime selector — `selectDemoProbe` — which picks among the precomputed pairs in
//      probes.json by information gain over the contested shard pool. The shard ships SVD-compressed
//      taste vectors that can't be indexed by genome position, so the demo measures spread through
//      each film's top-tag table instead. It never sees coverage data (hard rule 9): selection is
//      by information gain alone, so the demo can't steer visitors into the shard's dense regions.

import { PROBES } from "./constants.ts";
import { cosineScores, rowSlice } from "./math.ts";
import { filmTopTags, type DecodedShard, type PrecomputedProbe } from "./shard.ts";

// -- the spec-faithful algorithm (mirrors convo/probes.py) ---------------------------

export interface ProbeMovie {
  movieId: number;
  genomeRow: number;
  title: string;
  year: number | null;
  vector: ArrayLike<number>;
}

export interface Probe {
  axisPosition: number;
  axisName: string;
  spread: number;
  filmHighId: number;
  filmLowId: number;
  question: string;
}

/** The deterministic phrasing of a this-or-that probe (PROBE_QUESTION_TEMPLATE, filled in). */
export function phrasePair(highTitle: string, lowTitle: string): string {
  return `Which would you rather put on tonight — ${highTitle} or ${lowTitle}?`;
}

function anyNonZero(v: ArrayLike<number>): boolean {
  for (let i = 0; i < v.length; i++) if (v[i] !== 0) return true;
  return false;
}

function argmax(v: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

function argmin(v: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] < v[best]) best = i;
  return best;
}

/** Indices of `scores` by descending value, ties by descending index (np.argsort[::-1]). */
function argsortDescending(scores: Float64Array): number[] {
  const idx = Array.from(scores, (_, i) => i);
  idx.sort((a, b) => scores[a] - scores[b] || a - b); // ascending, stable
  idx.reverse();
  return idx;
}

/** Population variance of each column of a flat, row-major `nCols`-wide matrix (np.var, ddof=0). */
function columnVariance(matrix: ArrayLike<number>, nCols: number): Float64Array {
  const rows = matrix.length / nCols;
  const mean = new Float64Array(nCols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < nCols; j++) mean[j] += matrix[i * nCols + j];
  for (let j = 0; j < nCols; j++) mean[j] /= rows;
  const varr = new Float64Array(nCols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < nCols; j++) {
      const d = matrix[i * nCols + j] - mean[j];
      varr[j] += d * d;
    }
  for (let j = 0; j < nCols; j++) varr[j] /= rows;
  return varr;
}

function defaultUncertainty(profile: ArrayLike<number>, nTags: number): Float64Array {
  const u = new Float64Array(nTags);
  if (!anyNonZero(profile)) {
    u.fill(1);
    return u;
  }
  for (let j = 0; j < nTags; j++) u[j] = 1 - Math.min(1, Math.abs(profile[j]));
  return u;
}

/** Pick (high-pole, low-pole) indices within a contested set for one axis; null if degenerate. */
export function groundPair(contested: ArrayLike<number>, nCols: number, position: number): [number, number] | null {
  const rows = contested.length / nCols;
  const relevance = new Float64Array(rows);
  for (let i = 0; i < rows; i++) relevance[i] = contested[i * nCols + position];
  const high = argmax(relevance);
  const mean = relevance.reduce((a, b) => a + b, 0) / rows;

  const below: number[] = [];
  for (let i = 0; i < rows; i++) if (relevance[i] < mean) below.push(i);

  let low: number;
  if (below.length === 0) {
    low = argmin(relevance);
  } else {
    const sub = new Float64Array(below.length * nCols);
    below.forEach((r, k) => {
      for (let j = 0; j < nCols; j++) sub[k * nCols + j] = contested[r * nCols + j];
    });
    const sims = cosineScores(sub, nCols, rowSlice(contested, nCols, high));
    low = below[argmax(sims)];
  }
  return high === low ? null : [high, low];
}

export interface ChooseProbeOptions {
  excluded?: Set<number>;
  askedPositions?: Set<number>;
  uncertainty?: ArrayLike<number> | null;
  poolTop?: number;
}

/** Choose the most informative next probe over full genome vectors, or null (plan.md §5.2). */
export function chooseProbe(
  movies: ProbeMovie[],
  nCols: number,
  profile: ArrayLike<number>,
  names: Map<number, string>,
  opts: ChooseProbeOptions = {},
): Probe | null {
  const excluded = opts.excluded ?? new Set<number>();
  const poolTop = opts.poolTop ?? PROBES.POOL_TOP;

  const pool = movies.filter((m) => !excluded.has(m.movieId)).sort((a, b) => a.genomeRow - b.genomeRow);
  if (pool.length === 0) return null;

  const poolVecs = new Float64Array(pool.length * nCols);
  pool.forEach((m, i) => {
    for (let j = 0; j < nCols; j++) poolVecs[i * nCols + j] = m.vector[j];
  });

  const coldStart = !anyNonZero(profile);
  let contestedIdx: number[];
  if (coldStart || pool.length <= poolTop) {
    contestedIdx = pool.map((_, i) => i);
  } else {
    const scores = cosineScores(poolVecs, nCols, profile);
    contestedIdx = argsortDescending(scores).slice(0, poolTop);
  }

  const contested = new Float64Array(contestedIdx.length * nCols);
  contestedIdx.forEach((p, i) => {
    for (let j = 0; j < nCols; j++) contested[i * nCols + j] = poolVecs[p * nCols + j];
  });

  const weight = opts.uncertainty ?? defaultUncertainty(profile, nCols);
  const spread = columnVariance(contested, nCols);
  for (let j = 0; j < nCols; j++) spread[j] *= weight[j];
  if (opts.askedPositions) for (const p of opts.askedPositions) spread[p] = -1;

  const position = argmax(spread);
  if (spread[position] < PROBES.MIN_SPREAD) return null;

  const grounded = groundPair(contested, nCols, position);
  if (grounded === null) return null;
  const [highLocal, lowLocal] = grounded;
  const filmHigh = pool[contestedIdx[highLocal]];
  const filmLow = pool[contestedIdx[lowLocal]];

  return {
    axisPosition: position,
    axisName: names.get(position) ?? `tag#${position}`,
    spread: spread[position],
    filmHighId: filmHigh.movieId,
    filmLowId: filmLow.movieId,
    question: phrasePair(filmHigh.title, filmLow.title),
  };
}

// -- stopping rules (mirror convo/probes.py) -----------------------------------------

export interface StopDecision {
  stop: boolean;
  reason: string;
}

// Phrases that mean "stop quizzing me and show me something" — the escape hatch, honoured always.
const STOP_PATTERNS = [
  "just show me",
  "show me something",
  "just tell me",
  "just pick",
  "just recommend",
  "stop asking",
  "enough questions",
  "get on with it",
];

/** True when the user's words ask for the escape hatch ("just show me something"). */
export function wantsToStop(text: string): boolean {
  const lowered = text.trim().toLowerCase().replace(/\s+/g, " ");
  return STOP_PATTERNS.some((p) => lowered.includes(p));
}

/** Decide whether to stop asking, at the first rule that fires (plan.md §5.3). */
export function shouldStop(args: {
  turn: number;
  top5History: number[][];
  topScores?: number[] | null;
  userRequested?: boolean;
  maxTurns?: number;
  minTurns?: number;
  stableTurns?: number;
  marginThreshold?: number;
}): StopDecision {
  const maxTurns = args.maxTurns ?? PROBES.MAX_TURNS;
  const minTurns = args.minTurns ?? PROBES.MIN_TURNS;
  const stableTurns = args.stableTurns ?? PROBES.STABLE_TURNS;
  const marginThreshold = args.marginThreshold ?? PROBES.MARGIN_THRESHOLD;

  if (args.userRequested) return { stop: true, reason: "user_request" };
  if (args.turn >= maxTurns) return { stop: true, reason: "max_turns" };
  if (args.turn < minTurns) return { stop: false, reason: "continue" };

  const recent = args.top5History.slice(-stableTurns);
  if (recent.length === stableTurns && recent.every((s) => arraysEqual(s, recent[0]))) {
    return { stop: true, reason: "top5_stable" };
  }
  const scores = args.topScores;
  if (scores && scores.length >= 10 && scores[0] - scores[9] >= marginThreshold) {
    return { stop: true, reason: "margin" };
  }
  return { stop: false, reason: "continue" };
}

function arraysEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

// -- the demo's runtime selector over precomputed probes (plan.md §8.2) --------------

/**
 * Choose the next precomputed probe to ask by information gain over the contested shard pool.
 *
 * Cold start (no reactions yet) scores every film, so the axis of greatest spread across the whole
 * shard wins — which is the highest-spread precomputed probe. As reactions sharpen the profile the
 * contested set narrows to the high-scoring films and the spread is recomputed over *those*, so the
 * question that best divides the current contenders is asked next. Relevance is read from each
 * film's top-tag table (the SVD vectors can't be indexed by genome position). Coverage is never
 * consulted. Returns null when nothing left divides the pool.
 */
export function selectDemoProbe(
  profile: Float64Array,
  shard: DecodedShard,
  probes: PrecomputedProbe[],
  opts: { askedAxes?: Set<number>; seenFilmIds?: Set<number>; poolTop?: number } = {},
): PrecomputedProbe | null {
  const askedAxes = opts.askedAxes ?? new Set<number>();
  const seen = opts.seenFilmIds ?? new Set<number>();
  const poolTop = opts.poolTop ?? PROBES.POOL_TOP;

  // The contested set: every film at cold start, else the top-scoring films by taste cosine.
  let contested: number[];
  if (!anyNonZero(profile)) {
    contested = shard.films.map((_, i) => i);
  } else {
    const cos = cosineScores(shard.vectors, shard.nComponents, profile);
    contested = argsortDescending(cos).slice(0, poolTop);
  }

  // Per contested film, its genome-tag relevance map (from the top-tag table).
  const relevanceMaps = contested.map((i) => {
    const m = new Map<number, number>();
    for (const t of filmTopTags(shard, i)) m.set(t.position, t.relevance);
    return m;
  });

  let best: PrecomputedProbe | null = null;
  let bestSpread = PROBES.MIN_SPREAD;
  for (const probe of probes) {
    if (askedAxes.has(probe.axis)) continue;
    if (seen.has(probe.high.id) || seen.has(probe.low.id)) continue;
    let mean = 0;
    for (const map of relevanceMaps) mean += map.get(probe.axis) ?? 0;
    mean /= relevanceMaps.length;
    let variance = 0;
    for (const map of relevanceMaps) {
      const d = (map.get(probe.axis) ?? 0) - mean;
      variance += d * d;
    }
    variance /= relevanceMaps.length;
    if (variance > bestSpread) {
      bestSpread = variance;
      best = probe;
    }
  }
  return best;
}
