// The demo's entry point. It renders the page chrome, loads the shard and the precomputed
// questions, then hands the stage to the click-based conversation. Everything happens in the
// browser — the only network traffic is the shard and probes fetched here, and posters that
// lazy-load from TMDB's CDN once picks are shown (plan.md §8.1, §8.5).

import "./style.css";
import { clearShardCache, loadProbes, loadShard, type DecodedShard, type ProbesFile } from "./shard.ts";
import { Conversation } from "./conversation.ts";
import { CannedChat } from "./chat.ts";
import { TRANSCRIPT, TRANSCRIPT_MODEL } from "./transcript.ts";
import { detectBackend, LiveChat } from "./live.ts";

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
          <a class="nav-link" href="#how"><span class="sq amber"></span><span class="mono">How it works</span></a>
          <a class="nav-link" href="#conversation"><span class="sq blue"></span><span class="mono">Conversation</span></a>
          <a class="nav-link" href="#privacy"><span class="sq violet"></span><span class="mono">Privacy</span></a>
          <a class="nav-link" href="#full"><span class="sq rose"></span><span class="mono">Full version</span></a>
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

    <section class="wrap section" id="how">
      <div class="section-head"><span class="sq amber"></span><h2 class="mono">How it works</h2></div>
      <p class="section-lead">No sign-up and no questionnaire — you react to real films and the maths does
      the rest. It's the same recommender the full app runs, with the AI phrasing removed, so the demo
      stays a real recommender rather than a mock-up.</p>
      <p class="section-lead">It works in four stages — react, learn, recommend, explain — and they're the
      four tabs just below, live. Start the demo and step through them; each one explains itself as you go.</p>
    </section>

    <div class="strip" role="tablist" aria-label="Pipeline stages — pick one to view">
      <button class="seg active" role="tab" data-stage="react" aria-selected="true"><span class="sq green"></span><span class="mono">React</span><span class="step mono">01</span></button>
      <button class="seg" role="tab" data-stage="learn" aria-selected="false"><span class="sq cyan"></span><span class="mono">Learn</span><span class="step mono">02</span></button>
      <button class="seg" role="tab" data-stage="recommend" aria-selected="false"><span class="sq magenta"></span><span class="mono">Recommend</span><span class="step mono">03</span></button>
      <button class="seg" role="tab" data-stage="explain" aria-selected="false"><span class="sq orange"></span><span class="mono">Explain</span><span class="step mono">04</span></button>
    </div>

    <main class="wrap" id="stage">
      <p class="stage-blurb" id="stage-blurb" hidden></p>
      <section class="panel" id="panel"></section>
    </main>

    <section class="wrap section" id="conversation">
      <div class="section-head"><span class="sq blue"></span><h2 class="mono">A full conversation</h2></div>
      <p class="section-lead" id="conversation-lead">The demo above is the live recommender with the words
      stripped out. The full version keeps the words — it phrases every question, reads your answers in
      plain language, and explains the picks. Here's a recording of one; step through it.</p>
      <div class="chat" id="chat"></div>
    </section>

    <section class="wrap section" id="privacy">
      <div class="section-head"><span class="sq violet"></span><h2 class="mono">Privacy</h2></div>
      <p class="section-lead">Nothing you do here leaves the page: no account, no server, no AI calls, and
      your session is gone when the tab closes. It searches a 2,000-film sample and tells you when your
      taste points somewhere that sample covers thinly. The only network traffic after load is posters,
      lazy-loaded from TMDB.</p>
    </section>

    <section class="wrap section" id="full">
      <div class="section-head"><span class="sq rose"></span><h2 class="mono">The full version</h2></div>
      <p class="section-lead">The full version searches about 16,000 films and holds a real conversation
      in plain language, keeping a profile that follows your taste as it changes.</p>
      <p class="install">Install it: <code>pipx install cinegeist</code></p>
    </section>

    <footer class="wrap foot">
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

function showError(err: unknown): void {
  panel().innerHTML = `
    <div class="panel-head"><span class="sq orange"></span><span class="mono">Couldn't load</span></div>
    <p class="note">${String(err)}</p>`;
}

function scrollToStage(): void {
  document.querySelector("#stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Mark the nav link of whichever content section is currently near the top of the viewport, so a
// jump reads as intentional and you can see where you are. Nav links are hidden on mobile, and this
// no-ops without IntersectionObserver — the anchors still work either way.
function setupScrollSpy(): void {
  const links = new Map<string, HTMLElement>();
  app.querySelectorAll<HTMLElement>(".nav-link").forEach((link) => {
    const id = link.getAttribute("href")?.slice(1);
    if (id) links.set(id, link);
  });
  const sections = [...links.keys()]
    .map((id) => document.getElementById(id))
    .filter((el): el is HTMLElement => el !== null);
  if (!sections.length || !("IntersectionObserver" in window)) return;

  const setActive = (activeId: string | null): void => {
    links.forEach((link, id) => {
      const on = id === activeId;
      link.classList.toggle("active", on);
      if (on) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  };

  const visible = new Set<string>();
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target.id);
        else visible.delete(entry.target.id);
      }
      // The active section is the first one, in document order, still in the top band.
      let active: string | null = null;
      for (const id of links.keys()) {
        if (visible.has(id)) {
          active = id;
          break;
        }
      }
      setActive(active);
    },
    { rootMargin: "0px 0px -65% 0px", threshold: 0 },
  );
  sections.forEach((section) => observer.observe(section));
}

// The conversation section plays a recording by default. If a local backend is serving this page
// (`cinegeist serve`), it becomes the real, live conversation instead — the public Pages build has no
// backend, so detectBackend() returns null and the recording plays (plan.md §10, session 8).
async function mountChat(mount: HTMLElement): Promise<void> {
  mount.innerHTML = `<div class="chat-log"><div class="loading"><span class="dot"></span> Connecting…</div></div>`;
  const backend = await detectBackend();
  if (backend) {
    setConversationLive();
    await new LiveChat(mount, backend).start();
  } else {
    new CannedChat(mount, TRANSCRIPT, TRANSCRIPT_MODEL).render();
  }
}

function setConversationLive(): void {
  const lead = document.querySelector<HTMLElement>("#conversation-lead");
  if (lead) {
    lead.textContent =
      "You're running the full version, so this is the real conversation — you answer in plain " +
      "language, it phrases the questions and explains the picks, all served by the backend on this machine.";
  }
}

async function main(): Promise<void> {
  renderChrome();
  setupScrollSpy();

  const chatMount = app.querySelector<HTMLElement>("#chat");
  if (chatMount) void mountChat(chatMount);

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
  const blurb = app.querySelector<HTMLElement>("#stage-blurb")!;
  const conversation = new Conversation(panel(), strip, blurb, shard, probes);
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

  conversation.mountInitial();
}

void main();
