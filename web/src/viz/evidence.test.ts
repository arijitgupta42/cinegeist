import { describe, expect, it } from "vitest";
import { buildEvidenceGraph } from "./evidence.ts";

const axes = [
  {
    position: 10,
    name: "noir",
    weight: 0.8,
    contributors: [
      { filmId: 1, title: "Double Indemnity", value: 1, contribution: 0.6 },
      { filmId: 2, title: "The Maltese Falcon", value: 1, contribution: 0.4 },
    ],
  },
  {
    position: 20,
    name: "slow",
    weight: -0.5,
    contributors: [{ filmId: 3, title: "Solaris", value: -0.5, contribution: -0.3 }],
  },
];
const picks = [
  { filmId: 100, title: "Chinatown", tags: ["noir", "mystery"], wildcard: false },
  { filmId: 101, title: "Weird One", tags: ["unrelated"], wildcard: true },
];

describe("buildEvidenceGraph", () => {
  it("has a YOU node and one node per shown tag, contributing film, and pick", () => {
    const g = buildEvidenceGraph(axes, picks);
    expect(g.nodes.find((n) => n.kind === "you")).toBeTruthy();
    expect(g.nodes.filter((n) => n.kind === "tag").map((n) => n.label).sort()).toEqual(["noir", "slow"]);
    expect(g.nodes.filter((n) => n.kind === "film")).toHaveLength(3);
    expect(g.nodes.filter((n) => n.kind === "pick")).toHaveLength(2);
  });

  it("carries the sign of each tag and film", () => {
    const g = buildEvidenceGraph(axes, picks);
    expect(g.nodes.find((n) => n.label === "noir")!.sign).toBe(1);
    expect(g.nodes.find((n) => n.label === "slow")!.sign).toBe(-1);
    expect(g.nodes.find((n) => n.label === "Solaris")!.sign).toBe(-1);
  });

  it("links films to tags, tags to you, and picks to their matched tag or straight to you", () => {
    const g = buildEvidenceGraph(axes, picks);
    expect(g.links.some((l) => l.source === "film:1" && l.target === "tag:10")).toBe(true);
    expect(g.links.some((l) => l.source === "tag:10" && l.target === "you")).toBe(true);
    expect(g.links.some((l) => l.source === "pick:100" && l.target === "tag:10")).toBe(true); // matches noir
    expect(g.links.some((l) => l.source === "pick:101" && l.target === "you")).toBe(true); // matches nothing shown
  });

  it("caps the number of tags shown, strongest first", () => {
    const many = Array.from({ length: 20 }, (_, i) => ({ position: i, name: `t${i}`, weight: 1 - i * 0.01, contributors: [] }));
    const g = buildEvidenceGraph(many, [], { maxTags: 5 });
    expect(g.nodes.filter((n) => n.kind === "tag")).toHaveLength(5);
    expect(g.nodes.find((n) => n.label === "t0")).toBeTruthy(); // strongest kept
    expect(g.nodes.find((n) => n.label === "t19")).toBeFalsy(); // weakest dropped
  });
});
