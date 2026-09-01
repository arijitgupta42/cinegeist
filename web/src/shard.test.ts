import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeShard, filmTopTags, TAG_SENTINEL, TOP_TAGS_PER_FILM, type ShardManifest } from "./shard.ts";

// Decode the actual committed shard (the one Pages will serve) so this test guards the real bytes,
// not a synthetic fixture. Reading through node fs keeps it a pure decode with no browser APIs.
const jsonUrl = new URL("../public/shard/shard.json", import.meta.url);
const manifest = JSON.parse(readFileSync(jsonUrl, "utf-8")) as ShardManifest;
const binBuf = readFileSync(new URL(`../public/shard/${manifest.binary.file}`, import.meta.url));
const bin = binBuf.buffer.slice(binBuf.byteOffset, binBuf.byteOffset + binBuf.byteLength);
const shard = decodeShard(manifest, bin);

describe("decodeShard", () => {
  it("reports the manifest's film and component counts", () => {
    expect(shard.nFilms).toBe(manifest.n_films);
    expect(shard.nComponents).toBe(manifest.n_components);
    expect(shard.films).toHaveLength(manifest.n_films);
    expect(shard.fullCatalogSize).toBeGreaterThan(shard.nFilms);
  });

  it("dequantises one taste vector per film per component", () => {
    expect(shard.vectors).toHaveLength(shard.nFilms * shard.nComponents);
    for (let i = 0; i < shard.vectors.length; i += 7919) {
      expect(Number.isFinite(shard.vectors[i])).toBe(true);
    }
  });

  it("carries a coverage fraction in [0, 1] for every film", () => {
    expect(shard.coverage).toHaveLength(shard.nFilms);
    for (const c of shard.coverage) expect(c).toBeGreaterThanOrEqual(0), expect(c).toBeLessThanOrEqual(1);
  });

  it("keeps xyz coordinates for the taste-space map", () => {
    expect(shard.xyz).toHaveLength(shard.nFilms * 3);
    expect(Number.isFinite(shard.xyz[0])).toBe(true);
  });

  it("returns each film's top tags strongest-first, sentinels dropped", () => {
    const tags = filmTopTags(shard, 0);
    expect(tags.length).toBeGreaterThan(0);
    expect(tags.length).toBeLessThanOrEqual(TOP_TAGS_PER_FILM);
    for (let j = 1; j < tags.length; j++) {
      expect(tags[j].relevance).toBeLessThanOrEqual(tags[j - 1].relevance);
      expect(tags[j].position).not.toBe(TAG_SENTINEL);
    }
    // Every kept tag position has a name in the manifest's tag table.
    for (const t of tags) expect(shard.tagNames.has(t.position)).toBe(true);
  });

  it("exposes real film metadata", () => {
    const first = shard.films[0];
    expect(typeof first.id).toBe("number");
    expect(typeof first.title).toBe("string");
    expect(first.title.length).toBeGreaterThan(0);
  });
});
