// The click-based conversation, organised as the demo's four pipeline stages (plan.md §5, §8.2). The
// strip at the top — React · Learn · Recommend · Explain — is a tab bar: each stage renders into the
// panel below, all reading the one shared session so they can never disagree.
//
//   React      the this-or-that pairs; each click becomes a reaction (the only place taste is added)
//   Learn      what the reactions taught us, as the taste bars (View C; the 3D map lands here next)
//   Recommend  three confident picks and one wildcard, with the honesty path when the shard runs thin
//   Explain    each pick traced back to the reactions that produced it (the evidence graph lands here)
//
// React drives the flow: answering advances it, and when the top-5 settles, a confident margin opens,
// the turn cap is hit, or the visitor takes the escape hatch, it hands off to Recommend. The other
// three stages are views onto the current session and recompute whenever shown, so flipping to Learn
// mid-conversation shows the taste as it stands right now. No LLM anywhere; the maths decides.

import { recommend, type Candidate, type Recommendations, type ScoredFilm } from "./score.ts";
import { selectDemoProbe, shouldStop } from "./probes.ts";
import { assess, REASON_NO_CLOSE_NEIGHBOUR, type CoverageVerdict } from "./coverage.ts";
import { DemoSession, type TasteAxis } from "./session.ts";
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
const CHEV = `<span class="chev" aria-hidden="true">›</span>`;

export type Stage = "react" | "learn" | "recommend" | "explain";

interface Pool {
  candidates: Candidate[];
  vectors: Float64Array;
  shardIndex: number[]; // pool index → shard film index, for posters, tags, and reasons
}

export class Conversation {
  private session: DemoSession;
  private probes: PrecomputedProbe[];
  private pendingProbe: PrecomputedProbe | null = null;

  constructor(
    private mount: HTMLElement,
    private strip: HTMLElement,
    private shard: DecodedShard,
    probesFile: ProbesFile,
  ) {
    this.probes = probesFile.probes;
    this.session = new DemoSession(shard);
    this.strip.querySelectorAll<HTMLElement>("[data-stage]").forEach((btn) =>
      btn.addEventListener("click", () => this.goToStage(btn.dataset.stage as Stage)),
    );
  }

  /** Begin (or resume) the conversation at the React stage. */
  start(): void {
    this.goToStage("react");
  }

  /** Reset the session and begin a fresh conversation. */
  restart(): void {
    this.session.clear();
    this.pendingProbe = null;
    this.goToStage("react");
  }

  // -- stage routing -----------------------------------------------------------------

  /** Switch the active stage, sync the strip, and render that stage's view into the panel. */
  goToStage(stage: Stage): void {
    this.strip.querySelectorAll<HTMLElement>("[data-stage]").forEach((btn) => {
      const active = btn.dataset.stage === stage;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    switch (stage) {
      case "react":
        return this.renderReact();
      case "learn":
        return this.renderLearn();
      case "recommend":
        return this.renderRecommend();
      case "explain":
        return this.renderExplain();
    }
  }

  // -- React -------------------------------------------------------------------------

  private renderReact(): void {
    if (!this.pendingProbe) this.pendingProbe = this.nextProbe();
    if (this.pendingProbe) return this.renderProbe(this.pendingProbe);
    // Out of useful questions but the visitor is still on React: point them at their picks rather
    // than forcing a pair we don't have.
    this.mount.innerHTML = `
      <div class="panel-head"><span class="sq green"></span><span class="mono">React</span></div>
      <p class="note">That's every pair the demo catalog has to separate your taste. Your picks are ready.</p>
      <div class="controls"><button class="btn btn-accent" data-go="recommend"><span class="mono">See your picks</span>${CHEV}</button></div>`;
    this.wireNav();
  }

  private nextProbe(): PrecomputedProbe | null {
    return selectDemoProbe(this.session.centroid(), this.shard, this.probes, {
      askedAxes: this.session.askedAxes,
      seenFilmIds: this.session.seenFilmIds,
    });
  }

  private renderProbe(probe: PrecomputedProbe): void {
    const high = this.filmById(probe.high.id);
    const low = this.filmById(probe.low.id);
    if (!high || !low) {
      // A precomputed pair referencing a film not in the shard — shouldn't happen, but degrade
      // gracefully by moving on to the next-best question instead of showing a broken card.
      this.session.recordSkip(probe.high.id, probe.low.id, probe.axis);
      this.pendingProbe = null;
      return this.renderReact();
    }

    // "of up to 9" is the guided flow's soft cap; past it the visitor is answering extra pairs by
    // choice (via the React tab or an "answer more" link), so drop the cap phrase rather than lie.
    const n = this.session.turn + 1;
    const count = n <= 9 ? `question ${n} of up to 9` : `question ${n}`;
    this.mount.innerHTML = `
      <div class="panel-head">
        <span class="sq green"></span><span class="mono">React</span>
        <span class="mono progress">${count}</span>
      </div>
      <p class="ask">Which would you rather put on tonight?</p>
      <div class="pair">
        ${filmButton(high)}
        <div class="pair-or"><span class="mono">or</span></div>
        ${filmButton(low)}
      </div>
      <div class="controls subtle">
        <button class="link" data-skip><span class="mono">Haven't seen either</span></button>
        <button class="link" data-escape><span class="mono">Just show me something ›</span></button>
      </div>`;

    const [a, b] = this.mount.querySelectorAll<HTMLButtonElement>("[data-film]");
    a.addEventListener("click", () => this.onChoice(high.id, low.id, probe.axis));
    b.addEventListener("click", () => this.onChoice(low.id, high.id, probe.axis));
    this.mount.querySelector("[data-skip]")?.addEventListener("click", () => this.onSkip(high.id, low.id, probe.axis));
    this.mount.querySelector("[data-escape]")?.addEventListener("click", () => this.goToStage("recommend"));
  }

  private onChoice(chosenId: number, rejectedId: number, axis: number): void {
    this.session.recordChoice(chosenId, rejectedId, axis);
    this.afterReaction();
  }

  private onSkip(highId: number, lowId: number, axis: number): void {
    this.session.recordSkip(highId, lowId, axis);
    this.afterReaction();
  }

  // After a reaction, decide whether to keep asking or hand off to the picks. Recompute once here so
  // the stopping rule and the recorded top-5 see the same fresh ranking.
  private afterReaction(): void {
    const recs = this.computeRecs(this.session.centroid());
    this.session.recordTop5(recs.shortlist.map((f) => f.movieId));
    this.pendingProbe = null;

    const stop = shouldStop({
      turn: this.session.turn,
      top5History: this.session.top5History,
      topScores: recs.shortlist.map((f) => f.score),
    });
    if (stop.stop) return this.goToStage("recommend");

    this.pendingProbe = this.nextProbe();
    if (!this.pendingProbe) return this.goToStage("recommend");
    this.goToStage("react");
  }

  // -- Learn -------------------------------------------------------------------------

  private renderLearn(): void {
    if (!this.session.hasReactions) return this.renderEmpty("learn");
    this.mount.innerHTML = `
      <div class="panel-head"><span class="sq cyan"></span><span class="mono">Learn</span>
        <span class="mono progress">from ${this.reactionLabel()}</span></div>
      <p class="note">Every pair you react to nudges this. It's the same taste the picks are ranked against.</p>
      ${tasteBarsHtml(this.session.tasteAxes())}
      <div class="controls">
        <button class="btn" data-go="react"><span class="mono">Answer more pairs</span></button>
        <button class="btn btn-accent" data-go="recommend"><span class="mono">See your picks</span>${CHEV}</button>
      </div>`;
    this.wireNav();
  }

  // -- Recommend ---------------------------------------------------------------------

  private renderRecommend(): void {
    if (!this.session.hasReactions) return this.renderEmpty("recommend");
    const recs = this.computeRecs(this.session.centroid());
    const pool = recs.pool;
    const picks = recs.picks;

    // The honesty check runs against the whole shard and its coverage bytes, not the unseen pool — it
    // measures the shard's coverage of this taste, regardless of what's been shown (§8.4). It never
    // touches probe selection, so it can't have steered the conversation here.
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

    this.mount.innerHTML = `
      <div class="panel-head"><span class="sq ${verdict.honest ? "orange" : "magenta"}"></span><span class="mono">${header}</span>
        <span class="mono progress">from ${this.reactionLabel()}</span></div>
      ${verdict.honest ? this.honestyBanner(verdict) : ""}
      <div class="picks">${pickCards || `<p class="note">Not enough signal yet — react to a few pairs first.</p>`}</div>
      ${wildcard}
      <div class="controls">
        <button class="btn" data-go="explain"><span class="mono">Why these</span></button>
        <button class="btn" data-restart><span class="mono">Start over</span></button>
        <button class="btn" data-export><span class="mono">Export session</span></button>
      </div>`;
    this.wireNav();
  }

  // -- Explain -----------------------------------------------------------------------

  private renderExplain(): void {
    if (!this.session.hasReactions) return this.renderEmpty("explain");
    const recs = this.computeRecs(this.session.centroid());
    const pool = recs.pool;
    const byName = new Map(this.session.tasteAxes(1000).map((a) => [a.name, a] as const));
    const cards = recs.picks.map((f) => this.explainCard(f, pool, byName)).join("");

    this.mount.innerHTML = `
      <div class="panel-head"><span class="sq orange"></span><span class="mono">Explain</span>
        <span class="mono progress">traced to what you reacted to</span></div>
      <p class="note">Each pick, and the tags it matches — every tag links back to the films you actually chose.</p>
      <div class="why-list">${cards || `<p class="note">React to a few pairs first.</p>`}</div>
      <div class="controls">
        <button class="btn" data-go="recommend"><span class="mono">Back to picks</span></button>
        <button class="btn" data-go="react"><span class="mono">Answer more pairs</span></button>
      </div>`;
    this.wireNav();
  }

  // One pick's "why": its matched tags, each traced to the reacted films that carry them. This is the
  // evidence graph (View B) as text, and the same tasteAxes it will draw from (plan.md §9.2, §9.3).
  private explainCard(f: ScoredFilm, pool: Pool, byName: Map<string, TasteAxis>): string {
    const film = this.shard.films[pool.shardIndex[f.poolIndex]];
    const names = this.topTagNames(pool.shardIndex[f.poolIndex], 3);
    const rows = names
      .map((name) => {
        const axis = byName.get(name);
        const sources = axis
          ? [...new Set(axis.contributors.filter((c) => c.value > 0).map((c) => c.title))].slice(0, 3)
          : [];
        const from = sources.length
          ? `from ${sources.map((s) => `<span class="src">${escapeHtml(s)}</span>`).join(", ")}`
          : `from the overall pattern of your choices`;
        return `<li><span class="etag mono">${escapeHtml(name)}</span> <span class="efrom">${from}</span></li>`;
      })
      .join("");
    return `
      <article class="why">
        <div class="why-poster">${poster(film)}</div>
        <div class="why-body">
          <h3>${escapeHtml(film.title)}${film.year ? ` <span class="year">${film.year}</span>` : ""}</h3>
          <ul class="evidence">${rows || `<li class="efrom">Close to the taste you've shown so far.</li>`}</ul>
        </div>
      </article>`;
  }

  // -- empty state -------------------------------------------------------------------

  private renderEmpty(stage: Stage): void {
    const what =
      stage === "learn"
        ? "what it's learning about you"
        : stage === "recommend"
          ? "your recommendations"
          : "the reasons behind your picks";
    const sq = stage === "learn" ? "cyan" : stage === "recommend" ? "magenta" : "orange";
    const label = stage === "learn" ? "Learn" : stage === "recommend" ? "Recommend" : "Explain";
    this.mount.innerHTML = `
      <div class="panel-head"><span class="sq ${sq}"></span><span class="mono">${label}</span></div>
      <p class="note">React to a few pairs first, and ${what} will appear here.</p>
      <div class="controls"><button class="btn btn-accent" data-go="react"><span class="mono">Start reacting</span>${CHEV}</button></div>`;
    this.wireNav();
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

  // -- pick cards and reasons --------------------------------------------------------

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

  // -- small helpers -----------------------------------------------------------------

  private reactionLabel(): string {
    return `${this.session.turn} reaction${this.session.turn === 1 ? "" : "s"}`;
  }

  private filmById(id: number): ShardFilm | undefined {
    const idx = this.session.indexOf(id);
    return idx === undefined ? undefined : this.shard.films[idx];
  }

  // Shared wiring for the in-panel stage links and the results controls.
  private wireNav(): void {
    this.mount.querySelectorAll<HTMLElement>("[data-go]").forEach((b) =>
      b.addEventListener("click", () => this.goToStage(b.dataset.go as Stage)),
    );
    this.mount.querySelector("[data-restart]")?.addEventListener("click", () => this.restart());
    this.mount.querySelector("[data-export]")?.addEventListener("click", () => this.exportSession());
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

function filmButton(film: ShardFilm): string {
  return `
    <button class="film" data-film>
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
