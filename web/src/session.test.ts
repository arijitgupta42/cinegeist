import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { decodeShard, type ShardManifest } from "./shard.ts";
import { DemoSession } from "./session.ts";

// Drive the real committed shard, the same bytes Pages serves, so the taste-axis derivation is
// tested against genuine genome tags rather than a synthetic fixture. sessionStorage is absent in
// the node test environment; DemoSession guards every access, so it simply runs without persistence.
const manifest = JSON.parse(
  readFileSync(new URL("../public/shard/shard.json", import.meta.url), "utf-8"),
) as ShardManifest;
const binBuf = readFileSync(new URL(`../public/shard/${manifest.binary.file}`, import.meta.url));
const bin = binBuf.buffer.slice(binBuf.byteOffset, binBuf.byteOffset + binBuf.byteLength);
const shard = decodeShard(manifest, bin);

function freshSession(): DemoSession {
  const s = new DemoSession(shard);
  s.clear();
  return s;
}

describe("DemoSession.tasteAxes", () => {
  it("is empty before any reaction", () => {
    expect(freshSession().tasteAxes()).toEqual([]);
  });

  it("derives named, signed, ranked axes from the reacted films' genome tags", () => {
    const s = freshSession();
    s.recordChoice(shard.films[0].id, shard.films[1].id, 0);
    s.recordChoice(shard.films[2].id, shard.films[3].id, 1);
    const axes = s.tasteAxes();

    expect(axes.length).toBeGreaterThan(0);
    for (const a of axes) {
      expect(a.name.length).toBeGreaterThan(0);
      expect(Number.isFinite(a.weight)).toBe(true);
      expect(a.confidence).toBeGreaterThanOrEqual(0);
      expect(a.confidence).toBeLessThanOrEqual(1);
      expect(a.contributors.length).toBeGreaterThan(0);
    }
    // ordered most-positive to most-negative, the diverging chart's read order
    for (let i = 1; i < axes.length; i++) expect(axes[i].weight).toBeLessThanOrEqual(axes[i - 1].weight);
    // a chosen film pulls at least one axis positive
    expect(axes.some((a) => a.weight > 0)).toBe(true);
  });

  it("caps each side at the requested count", () => {
    const s = freshSession();
    for (let i = 0; i + 1 < 12; i += 2) s.recordChoice(shard.films[i].id, shard.films[i + 1].id, i);
    const axes = s.tasteAxes(3);
    expect(axes.filter((a) => a.weight > 0).length).toBeLessThanOrEqual(3);
    expect(axes.filter((a) => a.weight < 0).length).toBeLessThanOrEqual(3);
  });

  it("keeps strongTagPositions consistent with the positive taste axes after the refactor", () => {
    const s = freshSession();
    s.recordChoice(shard.films[0].id, shard.films[1].id, 0);
    const strong = new Set(s.strongTagPositions());
    expect(strong.size).toBeGreaterThan(0);
    // strongTagPositions is the positive slice of the same aggregate, so every strong position must
    // show up among the positive axes once the per-sign cap is lifted.
    const positive = new Set(s.tasteAxes(1000).filter((a) => a.weight > 0).map((a) => a.position));
    for (const p of strong) expect(positive.has(p)).toBe(true);
  });

  it("traces each axis back to the films that were reacted to", () => {
    const s = freshSession();
    const chosen = shard.films[0].id;
    const rejected = shard.films[1].id;
    s.recordChoice(chosen, rejected, 0);
    const ids = new Set(s.tasteAxes().flatMap((a) => a.contributors.map((c) => c.filmId)));
    for (const id of ids) expect([chosen, rejected]).toContain(id);
  });
});
