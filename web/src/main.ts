// The demo's entry point. It renders the page chrome, loads the shard and the precomputed
// questions, then hands the stage to the click-based conversation. Everything happens in the
// browser — the only network traffic is the shard and probes fetched here, and posters that
// lazy-load from TMDB's CDN once picks are shown (plan.md §8.1, §8.5).

import "./style.css";
import { clearShardCache, loadProbes, loadShard, type DecodedShard, type ProbesFile } from "./shard.ts";
import { Conversation } from "./conversation.ts";

const app = document.querySelector<HTMLDivElement>("#app")!;

const chevron = `<span class="chev" aria-hidden="true">›</span>`;

function renderChrome(): void {
  const base = import.meta.env.BASE_URL; // so the self-hosted TMDB logo resolves under the Pages base path
  app.innerHTML = `
    <div class="notice">
      This demo runs entirely in your browser — no account, no server, no AI calls.
      <a href="#privacy">Why that matters</a>
    </div>

    <nav class="nav">
      <div class="nav-pill">
        <a class="brand" href="#top">
          <span class="brand-mark" aria-hidden="true">
            <span style="background:var(--green)"></span><span style="background:var(--cyan)"></span>
            <span style="background:var(--magenta)"></span><span style="background:var(--yellow)"></span>
          </span>
          <span class="brand-name">CineGeist</span>
        </a>
        <div class="nav-links">
          <a class="nav-link" href="#how"><span class="sq green"></span><span class="mono">How it works</span></a>
          <a class="nav-link" href="#privacy"><span class="sq magenta"></span><span class="mono">Privacy</span></a>
          <a class="nav-link" href="#full"><span class="sq orange"></span><span class="mono">Full version</span></a>
        </div>
        <button class="btn btn-accent" data-start><span class="mono">Start the demo</span>${chevron}</button>
      </div>
    </nav>

    <header class="hero" id="top">
      <h1>Taste, not genres.</h1>
      <p class="sub">CineGeist shows you pairs of real films and asks which you'd rather watch.
      About eight clicks in, it knows more about your taste than a genre filter ever will — and it
      shows you exactly what it learned, and why.</p>
      <div class="cta">
        <button class="btn btn-accent" data-start><span class="mono">Start the demo</span>${chevron}</button>
        <a class="btn" href="#how"><span class="mono">How it works</span>${chevron}</a>
      </div>
    </header>

    <div class="strip" id="how" role="tablist" aria-label="How it works — pick a stage to view">
      <button class="seg active" role="tab" data-stage="react" aria-selected="true"><span class="sq green"></span><span class="mono">React</span><span class="step mono">01</span></button>
      <button class="seg" role="tab" data-stage="learn" aria-selected="false"><span class="sq cyan"></span><span class="mono">Learn</span><span class="step mono">02</span></button>
      <button class="seg" role="tab" data-stage="recommend" aria-selected="false"><span class="sq magenta"></span><span class="mono">Recommend</span><span class="step mono">03</span></button>
      <button class="seg" role="tab" data-stage="explain" aria-selected="false"><span class="sq orange"></span><span class="mono">Explain</span><span class="step mono">04</span></button>
    </div>

    <main class="wrap" id="stage">
      <section class="panel" id="panel"></section>
    </main>

    <footer class="wrap foot">
      <div class="row" id="full">
        <span><span class="sq cyan"></span> The full version searches about 16,000 films and holds a real conversation in plain language.</span>
        <span>Install it: <code>pipx install cinegeist</code></span>
      </div>
      <p class="attrib" id="privacy">
        Nothing you do here leaves the page: no account, no server, no AI calls, and your session is gone
        when the tab closes. It searches a 2,000-film sample and tells you when your taste points somewhere
        that sample covers thinly. The only network traffic after load is posters, lazy-loaded from TMDB.
      </p>
      <div class="colophon">
        <a class="tmdb" href="https://www.themoviedb.org/" target="_blank" rel="noopener noreferrer"
          aria-label="The Movie Database (TMDB)"><img src="${base}tmdb.svg" alt="The Movie Database (TMDB)" /></a>
        <p class="attrib legal">
          This product uses the TMDB API but is not endorsed or certified by TMDB. Posters and film metadata
          from TMDB; movie tag data from MovieLens (GroupLens Research).
        </p>
      </div>
      <p class="credit">
        <a href="https://github.com/arijitgupta42/cinegeist" target="_blank" rel="noopener noreferrer">Source on GitHub</a>
        · Visual design inspired by
        <a href="https://www.plasticity.xyz/" target="_blank" rel="noopener noreferrer">plasticity.xyz</a>.
        <button class="link" data-clear><span class="mono">Clear everything</span></button>
      </p>
    </footer>
  `;
}

function panel(): HTMLElement {
  return app.querySelector<HTMLElement>("#panel")!;
}

function showLoading(): void {
  panel().innerHTML = `
    <div class="panel-head"><span class="sq cyan"></span><span class="mono">Catalog</span></div>
    <div class="loading"><span class="dot"></span> Loading the taste catalog…</div>`;
}

function showIntro(shard: DecodedShard, onBegin: () => void): void {
  const decades = new Set(
    shard.films.map((f) => (f.year ? Math.floor(f.year / 10) * 10 : null)).filter((d) => d !== null),
  );
  panel().innerHTML = `
    <div class="panel-head"><span class="sq cyan"></span><span class="mono">Catalog ready</span></div>
    <div class="stats">
      <div class="stat"><div class="num">${shard.nFilms.toLocaleString()}</div>
        <div class="cap"><span class="sq cyan"></span><span class="mono">Films in this demo</span></div></div>
      <div class="stat"><div class="num">${decades.size}</div>
        <div class="cap"><span class="sq magenta"></span><span class="mono">Decades covered</span></div></div>
      <div class="stat"><div class="num">${shard.fullCatalogSize.toLocaleString()}</div>
        <div class="cap"><span class="sq orange"></span><span class="mono">In the full version</span></div></div>
    </div>
    <p class="note">Answer a handful of this-or-that pairs and CineGeist will find something to watch —
    then show you why. No question ever repeats a film you've already judged.</p>
    <div class="controls"><button class="btn btn-accent" data-begin><span class="mono">Begin</span>${chevron}</button></div>`;
  panel().querySelector("[data-begin]")?.addEventListener("click", onBegin);
}

function showError(err: unknown): void {
  panel().innerHTML = `
    <div class="panel-head"><span class="sq orange"></span><span class="mono">Couldn't load</span></div>
    <p class="note">${String(err)}</p>`;
}

function scrollToStage(): void {
  document.querySelector("#stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function main(): Promise<void> {
  renderChrome();

  app.querySelector("[data-clear]")?.addEventListener("click", async () => {
    try {
      sessionStorage.clear();
    } catch {
      // ignore
    }
    await clearShardCache();
    location.reload();
  });

  showLoading();
  let shard: DecodedShard;
  let probes: ProbesFile;
  try {
    [shard, probes] = await Promise.all([loadShard(import.meta.env.BASE_URL), loadProbes(import.meta.env.BASE_URL)]);
  } catch (err) {
    showError(err);
    return;
  }

  const strip = app.querySelector<HTMLElement>(".strip")!;
  const conversation = new Conversation(panel(), strip, shard, probes);
  const begin = (): void => {
    conversation.start();
    scrollToStage();
  };

  app.querySelectorAll<HTMLElement>("[data-start]").forEach((el) =>
    el.addEventListener("click", (e) => {
      e.preventDefault();
      begin();
    }),
  );

  showIntro(shard, begin);
}

void main();
