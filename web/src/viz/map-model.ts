// The pure maths behind the taste-space map (View A, plan.md §9.2). Everything here is a plain
// function over typed arrays — no three.js, no DOM — so it is unit-tested directly. The WebGL
// rendering lives in space3d.ts and the static fallback in map2d.ts; both consume what this produces.
//
// The map shows the 2,000 shard films at their precomputed UMAP coordinates, coloured by which
// cluster of taste-space they fall in, and a marker at the *barycentre* of the films the visitor has
// reacted to — the same signed, decayed weights the taste profile uses (plan.md §4.2, §9.2). We use
// the barycentre rather than projecting the profile vector because UMAP is non-linear and has no
// honest inverse; "you are here, between the films you liked" is both exact and one line of maths.

export interface WeightedPoint {
  w: number; // signed decayed weight (value × weight × decay), as in the profile
  coord: number[]; // the film's position in whatever space we're averaging (the map's xyz)
}

/** Σ wᵢ·coordᵢ / Σ|wᵢ| — the profile's own centroid formula, in coordinate space. */
export function barycenter(points: WeightedPoint[]): number[] | null {
  if (points.length === 0) return null;
  const dims = points[0].coord.length;
  const acc = new Array(dims).fill(0);
  let wabs = 0;
  for (const p of points) {
    for (let d = 0; d < dims; d++) acc[d] += p.w * p.coord[d];
    wabs += Math.abs(p.w);
  }
  if (wabs === 0) return null;
  return acc.map((x) => x / wabs);
}

/**
 * The barycentre after each successive reaction, for the map's trail. Reactions arrive as fixed-size
 * groups of points (a pair choice contributes two: the chosen film and the one passed over), so we
 * snapshot the cumulative barycentre after every `groupSize` points.
 */
export function cumulativeBarycenters(points: WeightedPoint[], groupSize = 2): number[][] {
  const trail: number[][] = [];
  for (let i = groupSize; i <= points.length; i += groupSize) {
    const b = barycenter(points.slice(0, i));
    if (b) trail.push(b);
  }
  return trail;
}

export interface FitTransform {
  center: number[]; // subtract this
  scale: number; // then multiply by this
}

/**
 * Centre a point cloud on its mean and scale it so its largest extent fits roughly in [-1, 1]. The
 * barycentre, trail and pick markers apply the same transform so they share the film cloud's frame.
 */
export function fitTransform(coords: Float32Array, n: number, dims = 3): FitTransform {
  const center = new Array(dims).fill(0);
  for (let i = 0; i < n; i++) for (let d = 0; d < dims; d++) center[d] += coords[i * dims + d];
  for (let d = 0; d < dims; d++) center[d] /= Math.max(1, n);

  let maxExtent = 0;
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dims; d++) maxExtent = Math.max(maxExtent, Math.abs(coords[i * dims + d] - center[d]));
  }
  return { center, scale: maxExtent > 0 ? 1 / maxExtent : 1 };
}

/** Apply a FitTransform to a single coordinate. */
export function applyTransform(coord: number[], t: FitTransform): number[] {
  return coord.map((x, d) => (x - t.center[d]) * t.scale);
}

// -- clustering (colours the film cloud by region of taste-space) --------------------

// A tiny deterministic RNG so the same shard always clusters and colours the same way.
function lcg(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

export interface KMeansResult {
  labels: Int32Array; // cluster index per point
  centroids: number[][];
}

/** Lloyd's algorithm with a seeded initialisation — deterministic given the seed. */
export function kmeans(data: Float32Array, n: number, dims: number, k: number, seed = 1, iters = 16): KMeansResult {
  const rnd = lcg(seed);
  const kk = Math.max(1, Math.min(k, n));
  const centroids: number[][] = [];
  const picked = new Set<number>();
  while (centroids.length < kk && picked.size < n) {
    const idx = Math.floor(rnd() * n);
    if (picked.has(idx)) continue;
    picked.add(idx);
    centroids.push(Array.from({ length: dims }, (_, d) => data[idx * dims + d]));
  }

  const labels = new Int32Array(n);
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        let d = 0;
        for (let j = 0; j < dims; j++) {
          const df = data[i * dims + j] - centroids[c][j];
          d += df * df;
        }
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      labels[i] = best;
    }
    const sums = centroids.map(() => new Array(dims).fill(0));
    const counts = new Array(centroids.length).fill(0);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c]++;
      for (let j = 0; j < dims; j++) sums[c][j] += data[i * dims + j];
    }
    for (let c = 0; c < centroids.length; c++) {
      if (counts[c] > 0) for (let j = 0; j < dims; j++) centroids[c][j] = sums[c][j] / counts[c];
    }
  }
  return { labels, centroids };
}

function hslToRgb(h: number, s: number, l: number): number[] {
  const f = (n: number): number => {
    const a = s * Math.min(l, 1 - l);
    const kk = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(kk - 3, Math.min(9 - kk, 1)));
  };
  return [f(0), f(8), f(4)];
}

/** k evenly-spaced hues, tuned to read on a black background — one colour per cluster. */
export function clusterPalette(k: number): number[][] {
  return Array.from({ length: Math.max(1, k) }, (_, i) => hslToRgb(i / Math.max(1, k), 0.62, 0.62));
}

/**
 * For each cluster, its tag positions ranked by how *characteristic* they are of that cluster — a tag
 * concentrated here relative to the whole shard outranks a globally common one. This lets the legend
 * label clusters with distinctive tags ("noir", "space travel") rather than ubiquitous ones
 * ("original", "criterion") that would otherwise win every cluster by raw count. `topTagOf(i)` gives
 * a film's strongest tag position (or -1). Ranked best-first; empty clusters return [].
 */
export function characteristicTags(labels: Int32Array, k: number, topTagOf: (i: number) => number): number[][] {
  const global = new Map<number, number>();
  const perCluster: Map<number, number>[] = Array.from({ length: k }, () => new Map());
  const sizes = new Array(k).fill(0);
  for (let i = 0; i < labels.length; i++) {
    const c = labels[i];
    if (c < 0 || c >= k) continue;
    sizes[c]++;
    const tag = topTagOf(i);
    if (tag < 0) continue;
    global.set(tag, (global.get(tag) ?? 0) + 1);
    perCluster[c].set(tag, (perCluster[c].get(tag) ?? 0) + 1);
  }
  const total = Math.max(1, labels.length);
  return perCluster.map((m, c) => {
    const scored: Array<{ tag: number; score: number }> = [];
    for (const [tag, count] of m) {
      // lift: how over-represented the tag is here versus shard-wide, weighted by support so a
      // one-off doesn't outrank a genuinely characteristic tag.
      const lift = count / Math.max(1, sizes[c]) / ((global.get(tag) ?? 1) / total);
      scored.push({ tag, score: lift * count });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.map((s) => s.tag);
  });
}

// -- 2D projection (for the static fallback) -----------------------------------------

/** The indices of the two highest-variance axes — the most informative plane to flatten onto. */
export function bestTwoAxes(coords: Float32Array, n: number, dims = 3): [number, number] {
  const mean = new Array(dims).fill(0);
  for (let i = 0; i < n; i++) for (let d = 0; d < dims; d++) mean[d] += coords[i * dims + d];
  for (let d = 0; d < dims; d++) mean[d] /= Math.max(1, n);
  const varc = new Array(dims).fill(0);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dims; d++) {
      const df = coords[i * dims + d] - mean[d];
      varc[d] += df * df;
    }
  }
  const order = Array.from({ length: dims }, (_, d) => d).sort((a, b) => varc[b] - varc[a]);
  return [order[0], order[1] ?? order[0]];
}
