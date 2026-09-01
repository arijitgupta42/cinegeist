import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// This config is ESM (package.json "type": "module"), so derive paths from import.meta.url rather
// than __dirname. The demo imports the shared scoring constants from spec/constants.json (the
// single source of truth both languages read, plan.md §8.6), which lives outside this Vite root,
// so the dev server has to be allowed to read the repo root.
const here = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  root: here,
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
});
