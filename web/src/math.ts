// Cosine similarity, the one numeric primitive the whole demo rests on — a faithful port of
// `cinegeist.catalog.genome.cosine_scores` (plan.md §3). The Python computes it in float32; here it
// is float64, and the shared spec/ fixtures are authored so that gap never decides a discrete
// outcome (a ranking, a chosen axis, a verdict) within the documented 1e-5 tolerance.
//
// Matrices are flat, row-major `ArrayLike<number>` with an explicit column count, so the same
// function serves the demo's dequantised shard vectors (a Float32Array) and the fixtures' small
// hand-built pools (a flattened Float64Array) without copying.

/** Cosine of every row of `matrix` (flat, row-major, `nCols` wide) against `query`. */
export function cosineScores(matrix: ArrayLike<number>, nCols: number, query: ArrayLike<number>): Float64Array {
  const rows = nCols === 0 ? 0 : matrix.length / nCols;
  const out = new Float64Array(rows);

  let qNorm = 0;
  for (let j = 0; j < nCols; j++) qNorm += query[j] * query[j];
  qNorm = Math.sqrt(qNorm);
  if (qNorm === 0) return out; // a zero query scores everything 0, never NaN

  for (let i = 0; i < rows; i++) {
    const base = i * nCols;
    let dot = 0;
    let rowNorm = 0;
    for (let j = 0; j < nCols; j++) {
      const x = matrix[base + j];
      dot += x * query[j];
      rowNorm += x * x;
    }
    const denom = Math.sqrt(rowNorm) * qNorm;
    const s = denom === 0 ? 0 : dot / denom;
    out[i] = Number.isFinite(s) ? s : 0;
  }
  return out;
}

/** Cosine of a single flat row against `query` (both `nCols` wide). */
export function cosine(row: ArrayLike<number>, query: ArrayLike<number>, nCols: number): number {
  return cosineScores(row, nCols, query)[0] ?? 0;
}

/** A copy of row `i` from a flat, row-major matrix `nCols` wide. */
export function rowSlice(matrix: ArrayLike<number>, nCols: number, i: number): Float64Array {
  const out = new Float64Array(nCols);
  const base = i * nCols;
  for (let j = 0; j < nCols; j++) out[j] = matrix[base + j];
  return out;
}
