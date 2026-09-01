// The click-based conversation (plan.md §5, §8.2). Each turn the demo picks the most informative
// precomputed pair by information gain, shows the two real films, and turns the click into a
// reaction. It stops when the top-5 settles, a confident margin opens, the turn cap is hit, or the
// visitor takes the escape hatch — then presents three confident picks and one wildcard, each with
// a templated reason. No LLM anywhere; the maths decides, the templates phrase.

import { recommend, type Candidate, type Recommendations, type ScoredFilm } from "./score.ts";
import { selectDemoProbe, shouldStop } from "./probes.ts";
import { assess, REASON_NO_CLOSE_NEIGHBOUR, type CoverageVerdict } from "./coverage.ts";
import { DemoSession } from "./session.ts";
import { tasteBarsHtml } from "./viz/bars.ts";
import {
  filmTopTags,
  TAG_SENTINEL,
  TOP_TAGS_PER_FILM,
  type DecodedShard,
  type PrecomputedProbe,
  type ProbesFile,
  type ShardFilm,
} from "./shard.ts";

const POSTER_BASE = "https://image.tmdb.org/t/p/w342";
const N_CONFIDENT = 3;

interface Pool {
  candidates: Candidate[];
  vectors: Float64Array;
  shardIndex: number[]; // pool index → shard film index, for posters, tags, and reasons
}

export class Conversation {
  private session: DemoSession;
  private probes: PrecomputedProbe[];

  constructor(
    private mount: HTMLElement,
    private shard: DecodedShard,
    probesFile: ProbesFile,
  ) {
    this.probes = probesFile.probes;
    this.session = new DemoSession(shard);
  }

  /** Begin (or resume) the conversation. */
  start(): void {
    this.runTurn();
  }

  /** Reset and begin a fresh conversation. */
  restart(): void {
    this.session.clear();
    this.runTurn();
  }

  // -- the turn loop -----------------------------------------------------------------

  private runTurn(recs?: Recommendations): void {
    const centroid = this.session.centroid();
    const result = recs ?? this.computeRecs(centroid);

    if (this.session.turn > 0) {
      const stop = shouldStop({
        turn: this.session.turn,
        top5History: this.session.top5History,
        topScores: result.shortlist.map((f) => f.score),
      });
      if (stop.stop) return this.presentPicks(result);
    }

    const probe = selectDemoProbe(centroid, this.shard, this.probes, {
      askedAxes: this.session.askedAxes,
      seenFilmIds: this.session.seenFilmIds,
    });
    if (probe === null) return this.presentPicks(result);
    this.renderProbe(probe);
  }

  private onChoice(chosenId: number, rejectedId: number, axis: number): void {
    this.session.recordChoice(chosenId, rejectedId, axis);
    this.afterReaction();
  }

  private onSkip(highId: number, lowId: number, axis: number): void {
    this.session.recordSkip(highId, lowId, axis);
    this.afterReaction();
  }

  private afterReaction(): void {
    const recs = this.computeRecs(this.session.centroid());
    this.session.recordTop5(recs.shortlist.map((f) => f.movieId));
    this.runTurn(recs);
  }

  // -- scoring -----------------------------------------------------------------------

  private buildPool(): Pool {
    const seen = this.session.seenFilmIds;
    const candidates: Candidate[] = [];
    const shardIndex: number[] = [];
    const k = this.shard.nComponents;
    const rows: number[] = [];
    this.shard.films.forEach((f, i) => {
      if (seen.has(f.id)) return;
      candidates.push({ movieId: f.id, title: f.title, year: f.year });
      shardIndex.push(i);
      rows.push(i);
    });
    const vectors = new Float64Array(rows.length * k);
    rows.forEach((src, dst) => {
      for (let j = 0; j < k; j++) vectors[dst * k + j] = this.shard.vectors[src * k + j];
    });
    return { candidates, vectors, shardIndex };
  }

  private computeRecs(centroid: Float64Array): Recommendations & { pool: Pool } {
    const pool = this.buildPool();
    const strong = new Set(this.session.strongTagPositions());
    const recs = recommend(pool.candidates, pool.vectors, this.shard.nComponents, centroid, {
      strongTagPositions: strong,
      nConfident: N_CONFIDENT,
      withWildcard: true,
      wildcardRelevanceAt: (poolIndex, pos) => tagRelevance(this.shard, pool.shardIndex[poolIndex], pos),
    });
    return { ...recs, pool };
  }

  // -- rendering ---------------------------------------------------------------------

  private renderProbe(probe: PrecomputedProbe): void {
    const high = this.filmById(probe.high.id);
    const low = this.filmById(probe.low.id);
    if (!high || !low) {
      // A precomputed pair referencing a film not in the shard — shouldn't happen, but degrade
      // gracefully by asking the next-best question instead of showing a broken card.
      this.session.recordSkip(probe.high.id, probe.low.id, probe.axis);
      return this.runTurn();
    }

    const n = this.session.turn + 1;
    this.mount.innerHTML = `
      <div class="panel-head">
        <span class="sq cyan"></span><span class="mono">Question ${n}</span>
        <span class="mono progress">of up to 9</span>
      </div>
      <p class="ask">Which would you rather put on tonight?</p>
      <div class="pair">
        ${filmButton(high, "choose")}
        <div class="pair-or"><span class="mono">or</span></div>
        ${filmButton(low, "choose")}
      </div>
      <div class="controls subtle">
        <button class="link" data-skip><span class="mono">Haven't seen either</span></button>
        <button class="link" data-escape><span class="mono">Just show me something ›</span></button>
      </div>`;

    const [a, b] = this.mount.querySelectorAll<HTMLButtonElement>("[data-film]");
    a.addEventListener("click", () => this.onChoice(high.id, low.id, probe.axis));
    b.addEventListener("click", () => this.onChoice(low.id, high.id, probe.axis));
    this.mount.querySelector("[data-skip]")?.addEventListener("click", () => this.onSkip(high.id, low.id, probe.axis));
    this.mount.querySelector("[data-escape]")?.addEventListener("click", () => {
      this.presentPicks(this.computeRecs(this.session.centroid()));
    });
  }

  private presentPicks(recs: Recommendations & { pool?: Pool }): void {
    const pool = recs.pool ?? this.buildPool();
    const picks = recs.picks;

    // The honesty check runs against the whole shard and its coverage bytes, not the unseen pool —
    // it measures the shard's coverage of this taste, regardless of what's been shown (§8.4). It
    // never touches probe selection, so it can't have steered the conversation here.
    const verdict = assess(this.session.centroid(), this.shard.vectors, this.shard.nComponents, this.shard.coverage);

    const header = verdict.honest ? "The closest the demo catalog gets" : "Your picks";
    const reason = (f: ScoredFilm) => (verdict.honest ? this.honestReason(f, pool) : this.reasonFor(f, pool));
    const pickCards = picks.map((f) => this.pickCard(f, pool, reason(f))).join("");

    // A deliberate exploration slot inside an already-sparse region is noise, so the honesty path
    // suppresses the wildcard entirely (plan.md §8.4). Ranking is otherwise unchanged — no backfill.
    const wildcard =
      !verdict.honest && recs.wildcard
        ? `<div class="pick-group">
             <div class="panel-head"><span class="sq magenta"></span><span class="mono">Wildcard</span></div>
             ${this.pickCard(recs.wildcard, pool, this.wildcardReason(recs.wildcard, pool), true)}
           </div>`
        : "";

    // The taste bars read the same axes the reasons above are built from (plan.md §9.2, §9.3), so
    // the chart and the picks always tell one story. Empty until there's signal, so it hides itself
    // on the escape-hatch-with-no-reactions path rather than drawing an empty frame.
    const bars = tasteBarsHtml(this.session.tasteAxes());

    this.mount.innerHTML = `
      <div class="panel-head"><span class="sq ${verdict.honest ? "orange" : "cyan"}"></span><span class="mono">${header}</span>
        <span class="mono progress">from ${this.session.turn} reaction${this.session.turn === 1 ? "" : "s"}</span></div>
      ${verdict.honest ? this.honestyBanner(verdict) : ""}
      <div class="picks">${pickCards || `<p class="note">Not enough signal yet — react to a few pairs first.</p>`}</div>
      ${wildcard}
      ${bars}
      <div class="controls">
        <button class="btn" data-restart><span class="mono">Start over</span></button>
        <button class="btn" data-export><span class="mono">Export session</span></button>
      </div>`;

    this.mount.querySelector("[data-restart]")?.addEventListener("click", () => this.restart());
    this.mount.querySelector("[data-export]")?.addEventListener("click", () => this.exportSession());
  }

  // The honesty path: state the numbers, name the thin direction, and show the nearest thing as
  // such — never pad with popular titles (plan.md §8.4, hard rule 9).
  private honestyBanner(verdict: CoverageVerdict): string {
    const pct = Math.max(1, Math.round(verdict.regionCoverage * 100));
    const leanNames = this.strongTagNames(2);
    const lean = leanNames.length ? ` You're leaning ${joinTags(leanNames)}.` : "";
    const noNeighbour = verdict.reasons.includes(REASON_NO_CLOSE_NEIGHBOUR)
      ? " Nothing in the demo is especially close to it."
      : "";
    return `
      <div class="honesty">
        <div class="panel-head"><span class="sq yellow"></span><span class="mono">The demo catalog runs thin here</span></div>
        <p>The demo catalog is ${this.shard.nFilms.toLocaleString()} films. Your taste is pointing somewhere it
        covers thinly — about ${pct}% of this neighbourhood survived the sample, and the full version searches
        about ${this.shard.fullCatalogSize.toLocaleString()}.${lean}${noNeighbour} Here's the closest it has, shown as such.</p>
      </div>`;
  }

  // States distance plainly rather than asserting a match: when the region is thin the top pick can
  // still be cosine-close, so this reports the cosine without claiming it's far (or near).
  private honestReason(f: ScoredFilm, pool: Pool): string {
    const names = this.topTagNames(pool.shardIndex[f.poolIndex], 2);
    const leans = names.length ? ` It leans ${joinTags(names)}.` : "";
    return `The closest the shard has to your taste — cosine ${f.cosine.toFixed(2)}.${leans}`;
  }

  private strongTagNames(k: number): string[] {
    return this.session
      .strongTagPositions()
      .slice(0, k)
      .map((pos) => this.shard.tagNames.get(pos) ?? `tag#${pos}`);
  }

  private pickCard(f: ScoredFilm, pool: Pool, reason: string, wild = false): string {
    const film = this.shard.films[pool.shardIndex[f.poolIndex]];
    return `
      <article class="pick${wild ? " wild" : ""}">
        ${poster(film)}
        <div class="pick-body">
          <h3>${escapeHtml(film.title)}${film.year ? ` <span class="year">${film.year}</span>` : ""}</h3>
          <p class="reason">${reason}</p>
        </div>
      </article>`;
  }

  // Reasons are HTML: the static copy is safe and the tag names are escaped inside joinTags.
  private reasonFor(f: ScoredFilm, pool: Pool): string {
    const names = this.topTagNames(pool.shardIndex[f.poolIndex], 3);
    if (names.length === 0) return "Close to the taste you've shown so far.";
    return `Matches your taste on ${joinTags(names)}.`;
  }

  private wildcardReason(f: ScoredFilm, pool: Pool): string {
    const names = this.topTagNames(pool.shardIndex[f.poolIndex], 2);
    const shared = names.length ? ` It still shares ${joinTags(names)}.` : "";
    return `Further from your center — a deliberate stretch.${shared}`;
  }

  private topTagNames(shardIndex: number, k: number): string[] {
    return filmTopTags(this.shard, shardIndex)
      .slice(0, k)
      .map((t) => this.shard.tagNames.get(t.position) ?? `tag#${t.position}`);
  }

  private filmById(id: number): ShardFilm | undefined {
    const idx = this.session.indexOf(id);
    return idx === undefined ? undefined : this.shard.films[idx];
  }

  private exportSession(): void {
    const blob = new Blob([this.session.exportJson()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cinegeist-session.json";
    a.click();
    URL.revokeObjectURL(url);
  }
}

// -- small pure helpers --------------------------------------------------------------

function tagRelevance(shard: DecodedShard, filmIndex: number, position: number): number {
  const base = filmIndex * TOP_TAGS_PER_FILM;
  for (let j = 0; j < TOP_TAGS_PER_FILM; j++) {
    const pos = shard.tagPos[base + j];
    if (pos === TAG_SENTINEL) break;
    if (pos === position) return shard.tagScore[base + j] / 255;
  }
  return 0;
}

function poster(film: ShardFilm): string {
  if (!film.poster) {
    return `<div class="poster ph"><span class="mono">${escapeHtml(film.title.slice(0, 2).toUpperCase())}</span></div>`;
  }
  return `<img class="poster" loading="lazy" src="${POSTER_BASE}${film.poster}" alt="${escapeHtml(film.title)} poster" />`;
}

function filmButton(film: ShardFilm, kind: string): string {
  return `
    <button class="film" data-film data-kind="${kind}">
      ${poster(film)}
      <span class="film-title">${escapeHtml(film.title)}${film.year ? ` <span class="year">${film.year}</span>` : ""}</span>
    </button>`;
}

function joinTags(names: string[]): string {
  const em = names.map((n) => `<em>${escapeHtml(n)}</em>`);
  if (em.length <= 1) return em.join("");
  if (em.length === 2) return `${em[0]} and ${em[1]}`;
  return `${em.slice(0, -1).join(", ")}, and ${em[em.length - 1]}`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
