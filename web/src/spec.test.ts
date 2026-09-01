// The TypeScript half of the drift guard (plan.md §8.6). It loads the exact same spec/ fixtures the
// Python suite asserts (tests/test_spec.py) and re-runs them through the browser ports, checking the
// output matches within the documented float tolerance. Change a scoring or decay rule without
// regenerating the fixtures and this goes red — the same red build the Python side gets.
//
// The `run*Case` functions here mirror tests/spec_runner.py exactly: a JSON-shaped case in, a
// JSON-shaped result out, compared field by field.

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { computeProfile, type DecayEvent, type MovieRow, type TagRow } from "./profile.ts";
import { recommend, scorePool, type Candidate } from "./score.ts";
import { chooseProbe, shouldStop, wantsToStop, type ProbeMovie } from "./probes.ts";
import { honestyReasons, nearestCosine, regionCoverage } from "./coverage.ts";
import { SCORING } from "./constants.ts";

const TOLERANCE = 1e-5;

function load(relpath: string): any {
  return JSON.parse(readFileSync(new URL(`../../spec/${relpath}`, import.meta.url), "utf-8"));
}

// Assert two JSON-shaped values match: numbers within `tol`, everything else exactly — the port of
// spec_runner.assert_close, so a mismatch points at the first field that moved.
function assertClose(fresh: any, committed: any, path = ""): void {
  const where = path || "<root>";
  if (typeof committed === "boolean" || typeof fresh === "boolean") {
    expect(fresh, where).toBe(committed);
  } else if (typeof committed === "number" && typeof fresh === "number") {
    expect(Math.abs(fresh - committed), `${where}: ${fresh} != ${committed}`).toBeLessThanOrEqual(TOLERANCE);
  } else if (Array.isArray(committed)) {
    expect(Array.isArray(fresh), where).toBe(true);
    expect(fresh.length, `${where}: length`).toBe(committed.length);
    for (let i = 0; i < committed.length; i++) assertClose(fresh[i], committed[i], `${path}[${i}]`);
  } else if (committed !== null && typeof committed === "object") {
    expect(fresh !== null && typeof fresh === "object", where).toBe(true);
    expect(Object.keys(fresh).sort(), `${where}: keys`).toEqual(Object.keys(committed).sort());
    for (const key of Object.keys(committed)) assertClose(fresh[key], committed[key], path ? `${path}.${key}` : key);
  } else {
    expect(fresh, where).toBe(committed); // strings, null
  }
}

// -- scoring -------------------------------------------------------------------------

function flatVectors(films: any[], nTags: number): Float64Array {
  const out = new Float64Array(films.length * nTags);
  films.forEach((f, i) => f.vector.forEach((x: number, j: number) => (out[i * nTags + j] = x)));
  return out;
}

function runScoringCase(c: any): any {
  const nTags = c.n_tags;
  const vectors = flatVectors(c.films, nTags);
  const candidates: Candidate[] = c.films.map((f: any) => ({
    movieId: f.movie_id,
    title: f.title,
    year: f.year ?? null,
    voteAverage: f.vote_average ?? null,
    voteCount: f.vote_count ?? null,
    popularity: f.popularity ?? null,
    genomeSource: f.genome_source ?? "measured",
  }));
  const profile: number[] = c.profile;
  const session = c.session ?? null;
  const facetScores = c.facet_scores ?? null;

  const scored = scorePool(candidates, vectors, nTags, profile, { sessionVector: session, facetScores });
  const result = recommend(candidates, vectors, nTags, profile, {
    strongTagPositions: c.strong_tag_positions ?? [],
    sessionVector: session,
    facetScores,
    nConfident: c.n_confident,
    shortlistSize: c.shortlist_size,
    lam: c.lam,
    withWildcard: c.with_wildcard,
  });

  return {
    scored: scored.map((s) => ({
      movie_id: s.movieId,
      score: s.score,
      cosine: s.cosine,
      quality: s.quality,
      session_fit: s.sessionFit,
      facet_match: s.facetMatch,
      popularity_penalty: s.popularityPenalty,
      confidence: s.confidence,
    })),
    shortlist_ids: result.shortlist.map((s) => s.movieId),
    picks_ids: result.picks.map((s) => s.movieId),
    wildcard_id: result.wildcard ? result.wildcard.movieId : null,
  };
}

describe("scoring/cases.json", () => {
  const cases = load("scoring/cases.json").cases as any[];
  for (const c of cases) {
    it(c.name, () => assertClose(runScoringCase(c), c.expected));
  }
});

// -- decay ---------------------------------------------------------------------------

function runDecayCase(c: any): any {
  const nTags = c.n_tags;
  const movies = new Map<number, MovieRow>();
  for (const m of c.movies ?? []) {
    movies.set(m.movie_id, { movieId: m.movie_id, title: m.clean_title || m.title || `M${m.movie_id}`, vector: m.vector });
  }
  const tags = new Map<number, TagRow>();
  const names = new Map<number, string>();
  for (const t of c.tags ?? []) {
    tags.set(t.tag_id, { tagId: t.tag_id, position: t.position, name: t.name });
    names.set(t.position, t.name);
  }
  const events: DecayEvent[] = c.events.map((e: any) => ({
    subjectKind: e.subject_kind,
    subject: Number(e.subject),
    value: e.value,
    weight: e.weight ?? 1.0,
    ageDays: e.age_days,
    evidence: e.evidence ?? null,
  }));
  const halfLife = c.half_life_days ?? undefined;
  const profile = computeProfile(events, movies, tags, nTags, names, halfLife);

  return {
    centroid: Array.from(profile.centroid),
    total_weight: profile.totalWeight,
    axes: profile.axes.map((a) => ({ position: a.position, name: a.name, weight: a.weight, source: a.source, evidence: a.evidence })),
  };
}

describe("scoring/decay.json", () => {
  const cases = load("scoring/decay.json").cases as any[];
  for (const c of cases) {
    it(c.name, () => assertClose(runDecayCase(c), c.expected));
  }
});

// -- probes and stopping -------------------------------------------------------------

function runProbeCase(c: any): any {
  const movies: ProbeMovie[] = c.movies.map((m: any) => ({
    movieId: m.movie_id,
    genomeRow: m.genome_row,
    title: m.clean_title || m.title || `M${m.movie_id}`,
    year: m.year ?? null,
    vector: m.vector,
  }));
  const names = new Map<number, string>();
  for (const t of c.tags) names.set(t.position, t.name);
  const probe = chooseProbe(movies, c.n_tags, c.profile, names, {
    excluded: new Set<number>(c.excluded_movie_ids ?? []),
    askedPositions: new Set<number>(c.asked_positions ?? []),
    uncertainty: c.uncertainty ?? null,
    poolTop: c.pool_top,
  });
  if (probe === null) return { probe: null };
  return {
    probe: {
      axis_position: probe.axisPosition,
      axis_name: probe.axisName,
      spread: probe.spread,
      film_high_id: probe.filmHighId,
      film_low_id: probe.filmLowId,
      question: probe.question,
    },
  };
}

describe("scoring/probes.json — probe selection", () => {
  const cases = load("scoring/probes.json").probe_selection as any[];
  for (const c of cases) {
    it(c.name, () => assertClose(runProbeCase(c), c.expected));
  }
});

describe("scoring/probes.json — stopping", () => {
  const cases = load("scoring/probes.json").stopping as any[];
  for (const c of cases) {
    it(c.name, () => {
      const d = shouldStop({
        turn: c.turn,
        top5History: c.top5_history,
        topScores: c.top_scores ?? null,
        userRequested: c.user_requested ?? false,
      });
      assertClose({ stop: d.stop, reason: d.reason }, c.expected);
    });
  }
});

describe("scoring/probes.json — escape hatch", () => {
  const cases = load("scoring/probes.json").escape_hatch as any[];
  for (const c of cases) {
    it(c.text, () => assertClose({ wants_to_stop: wantsToStop(c.text) }, c.expected));
  }
});

// -- coverage ------------------------------------------------------------------------

function flat(rows: number[][]): Float64Array {
  const nCols = rows[0]?.length ?? 0;
  const out = new Float64Array(rows.length * nCols);
  rows.forEach((r, i) => r.forEach((x, j) => (out[i * nCols + j] = x)));
  return out;
}

describe("scoring/coverage.json — region", () => {
  const cases = load("scoring/coverage.json").region as any[];
  for (const c of cases) {
    it(c.name, () => {
      const nCols = c.centroid.length;
      const vectors = flat(c.vectors);
      const result = {
        region_coverage: regionCoverage(c.centroid, vectors, nCols, c.coverage, c.top_k),
        nearest_cosine: nearestCosine(c.centroid, vectors, nCols),
      };
      assertClose(result, c.expected);
    });
  }
});

describe("scoring/coverage.json — verdict", () => {
  const cases = load("scoring/coverage.json").verdict as any[];
  for (const c of cases) {
    it(c.name, () => {
      const reasons = honestyReasons(c.region_coverage, c.nearest_cosine);
      assertClose({ honest: reasons.length > 0, reasons }, c.expected);
    });
  }
});

// -- constants -----------------------------------------------------------------------

describe("constants.json", () => {
  it("the demo reads the same scoring weights the fixtures were generated with", () => {
    // A cheap tripwire: if the committed constants and what the demo imports ever disagree, the
    // fixtures above would already fail, but this names the cause directly.
    const committed = load("constants.json").scoring;
    assertClose(SCORING, committed);
  });
});
