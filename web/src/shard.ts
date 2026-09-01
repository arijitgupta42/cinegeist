// Fetch, decode, and cache the browser-demo shard (plan.md §8.3, §8.5).
//
// The shard is the demo's whole catalog: ~2,000 films reduced from the full ~16,000-film genome to
// a few hundred kilobytes. `scripts/build_web_shard.py` emits two files — `shard.json` (metadata,
// the int8 scales, the binary layout) and `shard.bin` (packed int8 taste vectors, float32 xyz, the
// per-film top tags, and the coverage byte). This module turns those back into typed arrays the
// scorer works on, and caches them in IndexedDB so a second tab doesn't re-download the same bytes.
//
// Decoding is pure (`decodeShard`) so it is unit-tested against the committed shard with no browser
// APIs; only `loadShard` touches the network and IndexedDB.

import { EXCLUDED_TAGS } from "./constants.ts";

export interface ShardFilm {
  id: number;
  tmdb: number | null;
  title: string;
  year: number | null;
  runtime: number | null;
  poster: string | null;
}

interface BinarySection {
  name: string;
  dtype: "int8" | "float32" | "uint16" | "uint8";
  shape: number[];
  offset: number;
  length: number;
}

export interface ShardManifest {
  version: number;
  generated_at: string;
  seed: number;
  n_films: number;
  n_components: number;
  full_catalog_size: number;
  coverage_similarity: number;
  scales: number[];
  binary: { file: string; sections: BinarySection[] };
  tag_names: Record<string, string>;
  films: ShardFilm[];
}

// The demo's precomputed questions (probes.json), baked offline because the browser has no LLM to
// phrase them (plan.md §8.2). The demo chooses which to ask at runtime by information gain.
export interface PrecomputedProbe {
  axis: number;
  name: string;
  spread: number;
  high: { id: number; title: string; year: number | null };
  low: { id: number; title: string; year: number | null };
  question: string;
}

export interface ProbesFile {
  version: number;
  generated_at: string;
  seed: number;
  n_films: number;
  question_template: string;
  probes: PrecomputedProbe[];
}

// The number of top-tag slots packed per film (build.py TOP_TAGS_PER_FILM). A film with fewer
// non-zero tags pads unused slots with TAG_SENTINEL, which no real genome position reaches.
export const TOP_TAGS_PER_FILM = 12;
export const TAG_SENTINEL = 65535;

export interface DecodedShard {
  version: number;
  generatedAt: string;
  seed: number;
  nFilms: number;
  nComponents: number;
  fullCatalogSize: number;
  coverageSimilarity: number;
  // Dequantised taste vectors, row-major `nFilms x nComponents` (int8 * per-component scale). This
  // is the space the demo scores in: cosine here tracks cosine in the full genome closely enough
  // for a demo, at a fraction of the size.
  vectors: Float32Array;
  // Precomputed 3D coordinates for the taste-space map (session 7), row-major `nFilms x 3`.
  xyz: Float32Array;
  // Top-tag genome positions and 0-255 relevance scores per film, row-major `nFilms x 12`.
  tagPos: Uint16Array;
  tagScore: Uint8Array;
  // Per-film coverage fraction in [0, 1] (the honesty byte, §8.4), rescaled from the packed byte.
  coverage: Float32Array;
  tagNames: Map<number, string>;
  // Genome positions of the non-content tags (reception/verdict, spec/excluded_tags.json), so
  // filmTopTags can drop them from every tag the demo shows and reasons about.
  excludedPositions: Set<number>;
  films: ShardFilm[];
  scales: Float32Array;
}

function section(manifest: ShardManifest, name: string): BinarySection {
  const found = manifest.binary.sections.find((s) => s.name === name);
  if (!found) throw new Error(`shard.bin is missing the '${name}' section`);
  return found;
}

// A typed-array view over one packed section. Offsets are aligned to their element size by the
// packer, and the bytes are little-endian — true on every platform a browser runs on.
function view(bin: ArrayBuffer, s: BinarySection): Int8Array | Uint8Array | Uint16Array | Float32Array {
  const count = s.shape.reduce((a, b) => a * b, 1);
  switch (s.dtype) {
    case "int8":
      return new Int8Array(bin, s.offset, count);
    case "uint8":
      return new Uint8Array(bin, s.offset, count);
    case "uint16":
      return new Uint16Array(bin, s.offset, count);
    case "float32":
      return new Float32Array(bin, s.offset, count);
  }
}

/** Turn the manifest and packed binary back into typed arrays, dequantising the taste vectors. */
export function decodeShard(manifest: ShardManifest, bin: ArrayBuffer): DecodedShard {
  const n = manifest.n_films;
  const k = manifest.n_components;
  const scales = Float32Array.from(manifest.scales);

  const qVectors = view(bin, section(manifest, "vectors")) as Int8Array;
  const vectors = new Float32Array(n * k);
  for (let i = 0; i < n; i++) {
    const base = i * k;
    for (let c = 0; c < k; c++) vectors[base + c] = qVectors[base + c] * scales[c];
  }

  const xyz = Float32Array.from(view(bin, section(manifest, "xyz")) as Float32Array);
  const tagPos = Uint16Array.from(view(bin, section(manifest, "tag_pos")) as Uint16Array);
  const tagScore = Uint8Array.from(view(bin, section(manifest, "tag_score")) as Uint8Array);

  const coverageBytes = view(bin, section(manifest, "coverage")) as Uint8Array;
  const coverage = new Float32Array(n);
  for (let i = 0; i < n; i++) coverage[i] = coverageBytes[i] / 255;

  const tagNames = new Map<number, string>();
  for (const [pos, name] of Object.entries(manifest.tag_names)) tagNames.set(Number(pos), name);

  // Reception/verdict tags to drop from what the demo surfaces (mirrors the columns build_web_shard
  // zeroes for a fresh shard; applied here too so the committed shard is clean without a rebuild).
  const excludedPositions = new Set<number>();
  for (const [pos, name] of tagNames) if (EXCLUDED_TAGS.has(name)) excludedPositions.add(pos);

  return {
    version: manifest.version,
    generatedAt: manifest.generated_at,
    seed: manifest.seed,
    nFilms: n,
    nComponents: k,
    fullCatalogSize: manifest.full_catalog_size,
    coverageSimilarity: manifest.coverage_similarity,
    vectors,
    xyz,
    tagPos,
    tagScore,
    coverage,
    tagNames,
    excludedPositions,
    films: manifest.films,
    scales,
  };
}

/** The (position, score/255) tags of one film, sentinel-padded slots dropped, strongest first. */
export function filmTopTags(shard: DecodedShard, filmIndex: number): Array<{ position: number; relevance: number }> {
  const out: Array<{ position: number; relevance: number }> = [];
  const base = filmIndex * TOP_TAGS_PER_FILM;
  for (let j = 0; j < TOP_TAGS_PER_FILM; j++) {
    const pos = shard.tagPos[base + j];
    if (pos === TAG_SENTINEL) break;
    if (shard.excludedPositions.has(pos)) continue; // drop reception/verdict tags (keep the rest)
    out.push({ position: pos, relevance: shard.tagScore[base + j] / 255 });
  }
  return out;
}

// -- IndexedDB cache (plan.md §8.5) --------------------------------------------------
//
// The shard is identical for every visitor and isn't personal data, so it lives in IndexedDB
// rather than being re-downloaded per tab. Keyed by URL: once the shard filename is content-hashed
// (PR5), a redeploy changes the URL and misses the cache automatically. Every access is guarded —
// a browser that blocks storage (private mode) simply falls back to fetching each load.

const DB_NAME = "cinegeist";
const STORE = "shard";
const DB_VERSION = 1;

interface CachedShard {
  manifest: ShardManifest;
  bin: ArrayBuffer;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet<T>(key: string): Promise<T | undefined> {
  try {
    const db = await openDb();
    return await new Promise<T | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(key);
      req.onsuccess = () => resolve(req.result as T | undefined);
      req.onerror = () => reject(req.error);
      tx.oncomplete = () => db.close();
    });
  } catch {
    return undefined;
  }
}

async function idbPut<T>(key: string, value: T): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(value, key);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // Storage unavailable (private mode, quota) — the demo works, it just re-fetches next load.
  }
}

/** Wipe the cached shard. Backs the demo's "Clear everything" control (plan.md §8.5). */
export async function clearShardCache(): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).clear();
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // Nothing to clear if storage is unavailable.
  }
}

/**
 * Load and decode the shard, preferring the IndexedDB cache over the network.
 *
 * `baseUrl` is Vite's `import.meta.env.BASE_URL`, so the demo fetches correctly under a project
 * page's base path. The only network traffic here is the two shard files; posters lazy-load later
 * from TMDB's CDN, and nothing else leaves the page (plan.md §8.1).
 */
// The shard's content hash, injected at build time (vite.config.ts). Appending it as `?v=<hash>`
// busts the browser/CDN cache — and, since the IndexedDB entry is keyed by the full URL, the local
// cache too — whenever the committed shard changes, so a redeploy can never serve stale bytes.
const CACHE_BUST = typeof __SHARD_HASH__ === "string" ? `?v=${__SHARD_HASH__}` : "";

export async function loadShard(baseUrl: string): Promise<DecodedShard> {
  const jsonUrl = `${baseUrl}shard/shard.json${CACHE_BUST}`;
  const cached = await idbGet<CachedShard>(jsonUrl);
  if (cached) return decodeShard(cached.manifest, cached.bin);

  const manifest = (await (await fetch(jsonUrl)).json()) as ShardManifest;
  const binUrl = `${baseUrl}shard/${manifest.binary.file}${CACHE_BUST}`;
  const bin = await (await fetch(binUrl)).arrayBuffer();
  await idbPut<CachedShard>(jsonUrl, { manifest, bin });
  return decodeShard(manifest, bin);
}

/** Load the precomputed probe questions, preferring the IndexedDB cache over the network. */
export async function loadProbes(baseUrl: string): Promise<ProbesFile> {
  const url = `${baseUrl}shard/probes.json${CACHE_BUST}`;
  const cached = await idbGet<ProbesFile>(url);
  if (cached) return cached;
  const probes = (await (await fetch(url)).json()) as ProbesFile;
  await idbPut<ProbesFile>(url, probes);
  return probes;
}
