// The demo's per-tab session: the reaction log, the taste profile derived from it, and the
// bookkeeping the conversation needs (which axes we've asked, which films we've shown, the top-5
// history the stopping rule reads). It lives in sessionStorage — per tab, gone when the tab closes,
// which is exactly the session-bound lifetime the plan wants (§8.5). The profile is a derived view
// recomputed from the log, never stored mutated (plan.md §4.2), so the log is the whole state.

import { rowSlice } from "./math.ts";
import { computeProfile, decayFactor, type DecayEvent, type MovieRow } from "./profile.ts";
import { filmTopTags, type DecodedShard } from "./shard.ts";

const STORAGE_KEY = "cinegeist.session.v1";

// Picking a film pulls the profile toward it; the film passed over pushes it gently away, so a pair
// choice sharpens the contested axis from both sides.
const CHOSEN_VALUE = 1.0;
const REJECTED_VALUE = -0.5;

// How many of the profile's strongest genome tags feed the wildcard's "shares real tags" test.
const STRONG_TAGS = 15;

// How many taste axes the bar chart shows per side, positive and negative (plan.md §9.2, View C).
const TASTE_AXES_PER_SIGN = 6;

// One reacted film's push on one taste axis: which film, how it was judged, and the signed weight it
// contributed. Carries the film so the evidence graph (View B) can trace a tag back to its sources.
export interface TasteContribution {
  filmId: number;
  title: string;
  value: number; // the event value: +1 for a chosen film, −0.5 for one passed over
  relevance: number; // the film's genome relevance for this tag, 0..1
  contribution: number; // signed value × weight × decay × relevance — this film's push on this axis
}

// A learned taste axis over a real genome tag: signed strength, a confidence, and the films behind
// it. The shared view-model the visualizations read (plan.md §9.3) so they can't disagree.
export interface TasteAxis {
  position: number; // genome tag position
  name: string;
  weight: number; // signed aggregate affinity across all reacted films
  confidence: number; // 0..1: how much evidence backs it, and how consistently it points one way
  contributors: TasteContribution[]; // the films that produced it, strongest push first
}

// The running per-tag accumulation behind both strongTagPositions and tasteAxes.
interface TagAggregate {
  weight: number; // signed Σ contribution
  absTotal: number; // Σ |contribution|, for the sign-consistency half of confidence
  contributors: TasteContribution[];
}

interface Persisted {
  events: DecayEvent[];
  askedAxes: number[];
  seenFilmIds: number[];
  top5History: number[][];
  turn: number;
}

export class DemoSession {
  events: DecayEvent[] = [];
  askedAxes = new Set<number>();
  seenFilmIds = new Set<number>();
  top5History: number[][] = [];
  turn = 0;
  private idToIndex = new Map<number, number>();

  constructor(private shard: DecodedShard) {
    shard.films.forEach((f, i) => this.idToIndex.set(f.id, i));
    this.restore();
  }

  indexOf(movieId: number): number | undefined {
    return this.idToIndex.get(movieId);
  }

  private filmVector(movieId: number): Float64Array | null {
    const idx = this.idToIndex.get(movieId);
    if (idx === undefined) return null;
    return rowSlice(this.shard.vectors, this.shard.nComponents, idx);
  }

  /** Record a pair choice: pull toward the chosen film, push gently off the one passed over. */
  recordChoice(chosenId: number, rejectedId: number, axis: number): void {
    this.events.push({ subjectKind: "movie", subject: chosenId, value: CHOSEN_VALUE, weight: 1, ageDays: 0, evidence: null });
    this.events.push({ subjectKind: "movie", subject: rejectedId, value: REJECTED_VALUE, weight: 1, ageDays: 0, evidence: null });
    this.markAsked(chosenId, rejectedId, axis);
  }

  /** Record "seen neither / skip": still signal (obscurity tolerance), no taste direction. */
  recordSkip(highId: number, lowId: number, axis: number): void {
    this.markAsked(highId, lowId, axis);
  }

  private markAsked(aId: number, bId: number, axis: number): void {
    this.seenFilmIds.add(aId);
    this.seenFilmIds.add(bId);
    this.askedAxes.add(axis);
    this.turn++;
    this.persist();
  }

  recordTop5(ids: number[]): void {
    this.top5History.push(ids.slice(0, 5));
    this.persist();
  }

  /** The decayed taste centroid in the shard's SVD space, recomputed from the reaction log. */
  centroid(): Float64Array {
    const movies = new Map<number, MovieRow>();
    for (const e of this.events) {
      if (movies.has(e.subject)) continue;
      const v = this.filmVector(e.subject);
      if (v) movies.set(e.subject, { movieId: e.subject, title: "", vector: v });
    }
    return computeProfile(this.events, movies, new Map(), this.shard.nComponents, new Map()).centroid;
  }

  /** The profile's strongest genome-tag positions, from the reacted films' top-tag tables. */
  strongTagPositions(): number[] {
    return [...this.aggregateTags().entries()]
      .filter(([, a]) => a.weight > 0)
      .sort((a, b) => b[1].weight - a[1].weight)
      .slice(0, STRONG_TAGS)
      .map(([pos]) => pos);
  }

  /**
   * The learned taste as ranked, signed axes over real genome tags — the shared view-model the bar
   * chart (View C) and the other visualizations read (plan.md §9.2, §9.3). Positive weight = drawn
   * toward that tag, negative = pushed away. Confidence blends how much evidence backs the axis (more
   * distinct films → surer) with how consistently that evidence points one way (mixed signals → less
   * sure). Up to `perSign` axes on each side, ordered most-positive to most-negative for a diverging
   * chart.
   */
  tasteAxes(perSign: number = TASTE_AXES_PER_SIGN): TasteAxis[] {
    const axes: TasteAxis[] = [];
    for (const [position, a] of this.aggregateTags()) {
      if (a.absTotal === 0) continue;
      const films = new Set(a.contributors.map((c) => c.filmId)).size;
      const volume = 1 - Math.pow(0.5, films); // 1 film → 0.5, 2 → 0.75, saturating toward 1
      const consistency = Math.abs(a.weight) / a.absTotal; // 1 when every contributor agrees in sign
      a.contributors.sort((p, q) => Math.abs(q.contribution) - Math.abs(p.contribution));
      axes.push({
        position,
        name: this.shard.tagNames.get(position) ?? `tag#${position}`,
        weight: a.weight,
        confidence: volume * consistency,
        contributors: a.contributors,
      });
    }
    const positives = axes
      .filter((x) => x.weight > 0)
      .sort((p, q) => q.weight - p.weight)
      .slice(0, perSign);
    const negatives = axes
      .filter((x) => x.weight < 0)
      .sort((p, q) => p.weight - q.weight)
      .slice(0, perSign);
    return [...positives, ...negatives].sort((p, q) => q.weight - p.weight);
  }

  // Fold the reaction log into per-tag signed weight and the films that produced it. Each reacted
  // film contributes its top genome tags, each scaled by the event's signed decayed weight and the
  // film's relevance for that tag. Shared by strongTagPositions and tasteAxes so the two can't drift.
  private aggregateTags(): Map<number, TagAggregate> {
    const agg = new Map<number, TagAggregate>();
    for (const e of this.events) {
      const idx = this.idToIndex.get(e.subject);
      if (idx === undefined) continue;
      const w = e.value * e.weight * decayFactor(e.ageDays); // ageDays 0 in a session, so decay is 1
      const title = this.shard.films[idx].title;
      for (const t of filmTopTags(this.shard, idx)) {
        const contribution = w * t.relevance;
        let a = agg.get(t.position);
        if (!a) {
          a = { weight: 0, absTotal: 0, contributors: [] };
          agg.set(t.position, a);
        }
        a.weight += contribution;
        a.absTotal += Math.abs(contribution);
        a.contributors.push({ filmId: e.subject, title, value: e.value, relevance: t.relevance, contribution });
      }
    }
    return agg;
  }

  get hasReactions(): boolean {
    return this.events.length > 0;
  }

  /** Wipe the session (the demo's "Start over" / part of "Clear everything", §8.5). */
  clear(): void {
    this.events = [];
    this.askedAxes = new Set();
    this.seenFilmIds = new Set();
    this.top5History = [];
    this.turn = 0;
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // storage unavailable — nothing to clear
    }
  }

  /** The session as JSON — the "Export my session" download and the migration path to the CLI. */
  exportJson(): string {
    return JSON.stringify(this.snapshot(), null, 2);
  }

  private snapshot(): Persisted {
    return {
      events: this.events,
      askedAxes: [...this.askedAxes],
      seenFilmIds: [...this.seenFilmIds],
      top5History: this.top5History,
      turn: this.turn,
    };
  }

  private persist(): void {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(this.snapshot()));
    } catch {
      // storage full or blocked — the demo still works, it just won't survive a reload
    }
  }

  private restore(): void {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const d = JSON.parse(raw) as Persisted;
      this.events = d.events ?? [];
      this.askedAxes = new Set(d.askedAxes ?? []);
      this.seenFilmIds = new Set(d.seenFilmIds ?? []);
      this.top5History = d.top5History ?? [];
      this.turn = d.turn ?? 0;
    } catch {
      // corrupt or unavailable — start fresh
    }
  }
}
