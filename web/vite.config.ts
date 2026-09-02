import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// This config is ESM (package.json "type": "module"), so derive paths from import.meta.url rather
// than __dirname. The demo imports the shared scoring constants from spec/constants.json (the
// single source of truth both languages read, plan.md §8.6), which lives outside this Vite root,
// so the dev server has to be allowed to read the repo root.
const here = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

// A short content hash of the committed shard, injected so the demo fetches it as `?v=<hash>`.
// GitHub Pages could otherwise serve a stale shard after a redeploy; the hash changes with the
// bytes, busting the browser/CDN cache and the IndexedDB entry (which is keyed by the URL). Computed
// from the files on disk at config load; falls back to "dev" if they aren't built yet.
function shardHash(): string {
  try {
    const dir = fileURLToPath(new URL("./public/shard/", import.meta.url));
    const h = createHash("sha256");
    for (const f of ["shard.json", "shard.bin", "probes.json"]) h.update(readFileSync(dir + f));
    return h.digest("hex").slice(0, 12);
  } catch {
    return "dev";
  }
}

export default defineConfig(({ command }) => ({
  root: here,
  // A relative base makes one build work wherever it's mounted: a GitHub Pages project site under
  // /<repo>/, and equally `cinegeist serve` hosting it at the root for full mode (plan.md §10,
  // session 8). All asset and data URLs go through import.meta.env.BASE_URL, which becomes "./".
  // The dev server stays at the root so `make web-dev` opens at http://localhost:5173/.
  base: command === "build" ? "./" : "/",
  define: {
    __SHARD_HASH__: JSON.stringify(shardHash()),
  },
  server: {
    fs: {
      // Allow importing spec/constants.json (and, in tests, the spec/ fixtures) from the repo root.
      allow: [repoRoot],
    },
  },
  test: {
    // Pure functions and fixtures read from disk with node fs — no browser APIs needed here.
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
}));
