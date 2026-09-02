import { describe, expect, it } from "vitest";

import { type ChatEntry, entryHtml, TranscriptPlayer } from "./chat.ts";
import { TRANSCRIPT } from "./transcript.ts";

const ENTRIES: ChatEntry[] = [
  { from: "them", text: "one" },
  { from: "you", text: "two" },
  { from: "them", text: "three" },
];

describe("TranscriptPlayer", () => {
  it("starts with only the first entry visible", () => {
    const player = new TranscriptPlayer(ENTRIES);
    expect(player.step).toBe(1);
    expect(player.visible()).toHaveLength(1);
    expect(player.atStart).toBe(true);
    expect(player.atEnd).toBe(false);
  });

  it("steps forward and clamps at the end", () => {
    const player = new TranscriptPlayer(ENTRIES);
    expect(player.next()).toBe(true);
    expect(player.next()).toBe(true);
    expect(player.atEnd).toBe(true);
    expect(player.next()).toBe(false); // no fourth entry to reveal
    expect(player.step).toBe(3);
    expect(player.visible()).toHaveLength(3);
  });

  it("steps back and clamps at the start", () => {
    const player = new TranscriptPlayer(ENTRIES);
    player.next();
    expect(player.back()).toBe(true);
    expect(player.atStart).toBe(true);
    expect(player.back()).toBe(false); // already at the first entry
    expect(player.step).toBe(1);
  });

  it("restarts to the first entry", () => {
    const player = new TranscriptPlayer(ENTRIES);
    player.next();
    player.next();
    player.restart();
    expect(player.step).toBe(1);
  });

  it("handles an empty transcript without stepping past zero", () => {
    const player = new TranscriptPlayer([]);
    expect(player.total).toBe(0);
    expect(player.step).toBe(0);
    expect(player.next()).toBe(false);
    expect(player.visible()).toHaveLength(0);
  });
});

describe("the committed transcript", () => {
  it("is a real back-and-forth that ends in picks", () => {
    expect(TRANSCRIPT.length).toBeGreaterThan(4);
    expect(TRANSCRIPT.some((e) => e.from === "you")).toBe(true);
    const last = TRANSCRIPT[TRANSCRIPT.length - 1];
    expect("picks" in last && last.picks.length).toBeTruthy();
  });

  it("has exactly one wildcard among the final picks", () => {
    const last = TRANSCRIPT[TRANSCRIPT.length - 1];
    if (!("picks" in last)) throw new Error("the last entry must be the picks");
    expect(last.picks.filter((p) => p.wildcard)).toHaveLength(1);
  });
});

describe("entryHtml", () => {
  it("puts your answers on the 'you' side and theirs on the 'them' side", () => {
    expect(entryHtml({ from: "you", text: "hi" })).toContain("chat-row you");
    expect(entryHtml({ from: "them", text: "hi" })).toContain("chat-row them");
  });

  it("escapes text so a title can't inject markup", () => {
    const html = entryHtml({ from: "you", text: "<script>alert(1)</script>" });
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
  });

  it("renders a picks block with a wildcard flag", () => {
    const html = entryHtml({
      from: "them",
      header: "Your picks",
      picks: [{ title: "A", reason: "because", wildcard: true }],
    });
    expect(html).toContain("chat-picks");
    expect(html).toContain("wildcard");
  });
});
