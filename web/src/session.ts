// The demo's per-tab session: the reaction log, the taste profile derived from it, and the
// bookkeeping the conversation needs (which axes we've asked, which films we've shown, the top-5
// history the stopping rule reads). It lives in sessionStorage — per tab, gone when the tab closes,
// which is exactly the session-bound lifetime the plan wants (§8.5). The profile is a derived view
// recomputed from the log, never stored mutated (plan.md §4.2), so the log is the whole state.

import { rowSlice } from "./math.ts";
import { computeProfile, type DecayEvent, type MovieRow } from "./profile.ts";
import { filmTopTags, type DecodedShard } from "./shard.ts";

const STORAGE_KEY = "cinegeist.session.v1";

// Picking a film pulls the profile toward it; the film passed over pushes it gently away, so a pair
// choice sharpens the contested axis from both sides.
const CHOSEN_VALUE = 1.0;
const REJECTED_VALUE = -0.5;

// How many of the profile's strongest genome tags feed the wildcard's "shares real tags" test.
const STRONG_TAGS = 15;

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
    const weights = new Map<number, number>();
    for (const e of this.events) {
      const idx = this.idToIndex.get(e.subject);
      if (idx === undefined) continue;
      const w = e.value * e.weight; // ageDays 0 in a session, so decay is 1
      for (const t of filmTopTags(this.shard, idx)) {
        weights.set(t.position, (weights.get(t.position) ?? 0) + w * t.relevance);
      }
    }
    return [...weights.entries()]
      .filter(([, w]) => w > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, STRONG_TAGS)
      .map(([pos]) => pos);
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
