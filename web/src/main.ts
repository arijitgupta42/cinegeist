// The demo's entry point. It renders the page chrome, then loads the shard and reports what it
// holds; the click-based conversation, scoring, and honesty path arrive in later PRs. Everything
// happens in the browser — the only network traffic is the shard fetched here and, later, posters.

import "./style.css";
import { loadShard, type DecodedShard } from "./shard.ts";

const app = document.querySelector<HTMLDivElement>("#app")!;

const chevron = `<span class="chev" aria-hidden="true">›</span>`;

function renderChrome(): void {
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
        <a class="btn btn-accent" href="#stage" data-start><span class="mono">Start the demo</span>${chevron}</a>
      </div>
    </nav>

    <header class="hero" id="top">
      <h1>Taste, not genres.</h1>
      <p class="sub">CineGeist shows you pairs of real films and asks which you'd rather watch.
      About eight clicks in, it knows more about your taste than a genre filter ever will — and it
      shows you exactly what it learned, and why.</p>
      <div class="cta">
        <a class="btn btn-accent" href="#stage" data-start><span class="mono">Start the demo</span>${chevron}</a>
        <a class="btn" href="#how"><span class="mono">How it works</span>${chevron}</a>
      </div>
    </header>

    <div class="strip" id="how" aria-label="How it works">
      <div class="seg active g"><span class="sq green"></span><span class="mono">React</span><span class="step mono">01</span></div>
      <div class="seg"><span class="sq cyan"></span><span class="mono">Learn</span><span class="step mono">02</span></div>
      <div class="seg"><span class="sq magenta"></span><span class="mono">Recommend</span><span class="step mono">03</span></div>
      <div class="seg"><span class="sq orange"></span><span class="mono">Explain</span><span class="step mono">04</span></div>
    </div>

    <main class="wrap" id="stage">
      <section class="panel" id="panel"></section>
    </main>

    <footer class="wrap foot">
      <div class="row" id="full">
        <span><span class="sq cyan"></span> The full version searches about 13,000 films and holds a real conversation.</span>
        <span>Install it: <code>pipx install cinegeist</code></span>
      </div>
      <p class="attrib" id="privacy">
        Nothing you do here leaves the page: no account, no server, no AI calls, and your session is
        gone when the tab closes. This product uses the TMDB API but is not endorsed or certified by
        TMDB. Movie tag data from MovieLens (GroupLens Research).
      </p>
      <p class="credit">
        Visual design inspired by
        <a href="https://www.plasticity.xyz/" target="_blank" rel="noopener noreferrer">plasticity.xyz</a>.
      </p>
    </footer>
  `;

  app.querySelectorAll<HTMLElement>("[data-start]").forEach((el) =>
    el.addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelector("#stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }),
  );
}

function panel(): HTMLElement {
  return app.querySelector<HTMLElement>("#panel")!;
}

function showLoading(): void {
  panel().innerHTML = `
    <div class="panel-head"><span class="sq cyan"></span><span class="mono">Catalog</span></div>
    <div class="loading"><span class="dot"></span> Loading the taste catalog…</div>`;
}

function showLoaded(shard: DecodedShard): void {
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
    <p class="note">The pick-by-pick conversation lands in the next update — this is the catalog it
    will search, loaded from a ${(shard.nFilms).toLocaleString()}-film shard cached in your browser.</p>`;
}

function showError(err: unknown): void {
  panel().innerHTML = `
    <div class="panel-head"><span class="sq orange"></span><span class="mono">Couldn't load</span></div>
    <p class="note">${String(err)}</p>`;
}

async function main(): Promise<void> {
  renderChrome();
  showLoading();
  try {
    showLoaded(await loadShard(import.meta.env.BASE_URL));
  } catch (err) {
    showError(err);
  }
}

void main();
