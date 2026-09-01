// The demo's entry point. For now it loads the shard and reports what it holds; the click-based
// conversation, scoring, and honesty path arrive in later PRs. Everything the demo does happens in
// the browser — the only network traffic is the shard fetched here and, later, lazy-loaded posters.

import "./style.css";
import { loadShard, type DecodedShard } from "./shard.ts";

const app = document.querySelector<HTMLDivElement>("#app")!;

function shell(bodyHtml: string): void {
  app.innerHTML = `
    <header class="masthead">
      <h1>CineGeist</h1>
      <p>A movie recommender for people who can't explain their own taste. It shows you pairs of
      real films and asks which you'd rather watch — no account, no server, no AI calls.</p>
    </header>
    ${bodyHtml}
  `;
}

function reportLoaded(shard: DecodedShard): void {
  const decades = new Set(shard.films.map((f) => (f.year ? Math.floor(f.year / 10) * 10 : null)).filter((d) => d !== null));
  shell(`
    <div class="card">
      <p class="status">Loaded <strong>${shard.nFilms.toLocaleString()}</strong> films
      (${shard.nComponents} taste components) spanning
      <strong>${decades.size}</strong> decades.</p>
      <p class="muted">The full version searches about ${shard.fullCatalogSize.toLocaleString()} films.
      The conversation lands in the next update.</p>
    </div>
  `);
}

async function main(): Promise<void> {
  shell(`<div class="card"><p class="status">Loading the catalog…</p></div>`);
  try {
    const shard = await loadShard(import.meta.env.BASE_URL);
    reportLoaded(shard);
  } catch (err) {
    shell(`<div class="card"><p class="status">Couldn't load the catalog: ${String(err)}</p></div>`);
  }
}

void main();
