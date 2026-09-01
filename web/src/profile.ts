// Turn the event log into a decayed taste profile — a TypeScript port of
// `cinegeist.profile.update` (plan.md §4.2, §8.6). The profile is a weighted centroid in the film
// space:
//
//   w_i    = value_i × weight_i × 0.5 ** (age_days_i / HALF_LIFE)
//   vector = Σ (w_i × v_i) / Σ |w_i|
//
// where v_i is a movie's genome row (or a one-hot on an answered axis for a tag event). Old
// evidence fades rather than being deleted, which gives taste drift for free. Ranked axes carry the
// strongest signed affinities with the evidence that most produced each, for display and
// explanation. Checked against scoring/decay.json, the same fixtures the Python suite asserts.
//
// In the demo the "genome" is the shard's SVD-compressed space and events are all recent (a session
// lives in one tab), but the maths is identical and dimension-agnostic — the same function serves
// the fixtures' full-genome cases and the demo's 96-component vectors.

import { DECAY } from "./constants.ts";

export interface DecayEvent {
  subjectKind: "movie" | "tag" | "facet";
  subject: number; // a movie id or a genome tag id, read per subjectKind
  value: number;
  weight: number;
  ageDays: number;
  evidence: string | null;
}

export interface MovieRow {
  movieId: number;
  title: string; // the display title (clean title preferred), used as an axis's source label
  vector: ArrayLike<number>;
}

export interface TagRow {
  tagId: number;
  position: number;
  name: string;
}

export interface Axis {
  position: number;
  name: string;
  weight: number;
  source: string;
  evidence: string | null;
}

export interface Profile {
  centroid: Float64Array;
  totalWeight: number;
  axes: Axis[];
}

/** The multiplier 0.5 ** (ageDays / halfLife), clamped to ≤ 1 (a future timestamp can't amplify). */
export function decayFactor(ageDays: number, halfLife: number = DECAY.HALF_LIFE_DAYS): number {
  return Math.pow(0.5, Math.max(0, ageDays) / halfLife);
}

// One event's contribution: its signed decayed weight, where it came from, its words, and a lookup
// of the vector value it acts on at a given axis (a genome column for a movie, a one-hot for a tag).
interface Contribution {
  w: number;
  source: string;
  evidence: string | null;
  valueAt: (position: number) => number;
}

function bestEvidence(contributions: Contribution[], position: number): { evidence: string | null; source: string } {
  let best: { evidence: string | null; source: string } | null = null;
  let bestMagnitude = 0;
  for (const c of contributions) {
    const magnitude = Math.abs(c.w * c.valueAt(position));
    if (magnitude > bestMagnitude) {
      bestMagnitude = magnitude;
      best = { evidence: c.evidence, source: c.source };
    }
  }
  return best ?? { evidence: null, source: "" };
}

/** Indices of `values` by descending |value|, ties broken by ascending index (np.argsort[::-1]). */
function argsortAbsDescending(values: Float64Array): number[] {
  const idx = Array.from(values, (_, i) => i);
  idx.sort((a, b) => Math.abs(values[a]) - Math.abs(values[b]) || a - b); // ascending, stable
  idx.reverse(); // → descending |value|, ties by descending index, matching numpy
  return idx;
}

function rankAxes(centroid: Float64Array, contributions: Contribution[], names: Map<number, string>): Axis[] {
  if (contributions.length === 0) return [];
  let positives = 0;
  let negatives = 0;
  const chosen: Axis[] = [];
  for (const position of argsortAbsDescending(centroid)) {
    const weight = centroid[position];
    if (Math.abs(weight) < DECAY.AXIS_EPSILON) break; // everything past here is noise
    if (weight > 0 && positives >= DECAY.AXES_PER_SIGN) continue;
    if (weight < 0 && negatives >= DECAY.AXES_PER_SIGN) continue;
    const { evidence, source } = bestEvidence(contributions, position);
    chosen.push({ position, name: names.get(position) ?? `tag#${position}`, weight, source, evidence });
    if (weight > 0) positives++;
    else negatives++;
    if (positives >= DECAY.AXES_PER_SIGN && negatives >= DECAY.AXES_PER_SIGN) break;
  }
  chosen.sort((a, b) => b.weight - a.weight); // stable: equal weights keep selection order
  return chosen;
}

/** Fold an event log into the decayed centroid, total weight, and ranked axes with evidence. */
export function computeProfile(
  events: DecayEvent[],
  movies: Map<number, MovieRow>,
  tags: Map<number, TagRow>,
  nTags: number,
  names: Map<number, string>,
  halfLife: number = DECAY.HALF_LIFE_DAYS,
): Profile {
  const numerator = new Float64Array(nTags);
  let totalWeight = 0;
  const contributions: Contribution[] = [];

  for (const event of events) {
    let valueAt: (position: number) => number;
    let source: string;
    let addToNumerator: (w: number) => void;

    if (event.subjectKind === "movie") {
      const movie = movies.get(event.subject);
      if (!movie) continue; // unknown or un-vectored film: real evidence, but no direction to add
      const vector = movie.vector;
      valueAt = (pos) => vector[pos];
      source = movie.title;
      addToNumerator = (w) => {
        for (let j = 0; j < nTags; j++) numerator[j] += w * vector[j];
      };
    } else if (event.subjectKind === "tag") {
      const tag = tags.get(event.subject);
      if (!tag) continue;
      const pos = tag.position;
      valueAt = (p) => (p === pos ? 1 : 0);
      source = `'${tag.name}'`;
      addToNumerator = (w) => {
        numerator[pos] += w;
      };
    } else {
      continue; // a facet/constraint is a filter, not a taste-space direction
    }

    const w = event.value * event.weight * decayFactor(event.ageDays, halfLife);
    if (w === 0) continue;
    addToNumerator(w);
    totalWeight += Math.abs(w);
    contributions.push({ w, source, evidence: event.evidence, valueAt });
  }

  const centroid = new Float64Array(nTags);
  for (let j = 0; j < nTags; j++) centroid[j] = totalWeight > 0 ? numerator[j] / totalWeight : numerator[j];

  return { centroid, totalWeight, axes: rankAxes(centroid, contributions, names) };
}
