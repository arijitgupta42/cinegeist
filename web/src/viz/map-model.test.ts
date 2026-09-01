import { describe, expect, it } from "vitest";
import {
  applyTransform,
  barycenter,
  bestTwoAxes,
  characteristicTags,
  clusterPalette,
  cumulativeBarycenters,
  fitTransform,
  kmeans,
} from "./map-model.ts";

describe("barycenter", () => {
  it("is the weighted average of the coordinates", () => {
    expect(barycenter([{ w: 1, coord: [0, 0, 0] }, { w: 1, coord: [2, 4, 6] }])).toEqual([1, 2, 3]);
  });

  it("uses Σ|w| as the denominator, so a passed-over film pulls the marker the other way", () => {
    // chosen (+1) at origin, passed over (−0.5) at x=2: Σw·c = (−1, 0); Σ|w| = 1.5
    const b = barycenter([{ w: 1, coord: [0, 0] }, { w: -0.5, coord: [2, 0] }])!;
    expect(b[0]).toBeCloseTo(-2 / 3);
    expect(b[1]).toBeCloseTo(0);
  });

  it("returns null with no points or no weight", () => {
    expect(barycenter([])).toBeNull();
    expect(barycenter([{ w: 0, coord: [1, 1, 1] }])).toBeNull();
  });
});

describe("cumulativeBarycenters", () => {
  it("snapshots the barycentre after each group of points", () => {
    const pts = [
      { w: 1, coord: [0, 0] },
      { w: 1, coord: [2, 0] },
      { w: 1, coord: [4, 0] },
      { w: 1, coord: [6, 0] },
    ];
    const trail = cumulativeBarycenters(pts, 2);
    expect(trail).toHaveLength(2);
    expect(trail[0][0]).toBeCloseTo(1); // mean of first two x's
    expect(trail[1][0]).toBeCloseTo(3); // mean of all four
  });
});

describe("fitTransform / applyTransform", () => {
  it("centres on the mean and scales the largest extent to 1", () => {
    const coords = Float32Array.from([-2, 0, 0, 2, 0, 0, 0, 0, 0]);
    const t = fitTransform(coords, 3);
    expect(t.center).toEqual([0, 0, 0]);
    expect(t.scale).toBeCloseTo(0.5);
    expect(applyTransform([2, 0, 0], t)).toEqual([1, 0, 0]);
  });
});

describe("kmeans", () => {
  const data = Float32Array.from([0, 0, 0, 0.1, 0, 0, 10, 10, 10, 10.1, 10, 10]);

  it("groups well-separated points and separates the two clumps", () => {
    const { labels } = kmeans(data, 4, 3, 2, 1);
    expect(labels[0]).toBe(labels[1]);
    expect(labels[2]).toBe(labels[3]);
    expect(labels[0]).not.toBe(labels[2]);
  });

  it("is deterministic for a given seed", () => {
    const a = kmeans(data, 4, 3, 2, 7);
    const b = kmeans(data, 4, 3, 2, 7);
    expect([...a.labels]).toEqual([...b.labels]);
  });
});

describe("clusterPalette", () => {
  it("returns k rgb triples in [0, 1]", () => {
    const pal = clusterPalette(5);
    expect(pal).toHaveLength(5);
    for (const c of pal) {
      expect(c).toHaveLength(3);
      for (const v of c) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe("characteristicTags", () => {
  it("ranks each cluster's tags, surfacing the one concentrated there", () => {
    const labels = Int32Array.from([0, 0, 0, 0, 1, 1]);
    const topTag = (i: number) => [1, 1, 1, 1, 2, 2][i]; // cluster 0 is all tag 1, cluster 1 all tag 2
    const ranked = characteristicTags(labels, 2, topTag);
    expect(ranked[0][0]).toBe(1);
    expect(ranked[1][0]).toBe(2);
  });

  it("prefers a cluster-specific tag over a globally ubiquitous one", () => {
    // Tag 9 appears in every film (ubiquitous); tag 5 only in cluster 1 (characteristic of it).
    const labels = Int32Array.from([0, 0, 0, 1, 1, 1]);
    const dom = [9, 9, 9, 9, 5, 5];
    // Model "top tag" as the film's single dominant tag; cluster 1's films split 9,5,5.
    const ranked = characteristicTags(labels, 2, (i) => dom[i]);
    expect(ranked[1][0]).toBe(5); // 5 is characteristic of cluster 1, not the ubiquitous 9
  });
});

describe("bestTwoAxes", () => {
  it("picks the two highest-variance axes", () => {
    const coords = Float32Array.from([-5, 0, -2, 5, 0.01, 2, 0, 0, 0]);
    expect(bestTwoAxes(coords, 3, 3)).toEqual([0, 2]);
  });
});
