import { describe, expect, it } from "vitest";

import { apiMessagesToEntries } from "./live.ts";

describe("apiMessagesToEntries", () => {
  it("turns a say message into a 'them' bubble", () => {
    const [entry] = apiMessagesToEntries([{ type: "say", text: "hello" }]);
    expect(entry).toEqual({ from: "them", text: "hello" });
  });

  it("turns an error message into a 'them' bubble too", () => {
    const [entry] = apiMessagesToEntries([{ type: "error", text: "boom" }]);
    expect(entry).toEqual({ from: "them", text: "boom" });
  });

  it("folds the wildcard into the picks list with its flag set", () => {
    const [entry] = apiMessagesToEntries([
      {
        type: "picks",
        header: "Your picks",
        picks: [
          { id: 1, title: "A", year: 1999, explanation: "because A", wildcard: false },
          { id: 2, title: "B", year: null, explanation: "because B", wildcard: false },
        ],
        wildcard: { id: 3, title: "C", year: 2020, explanation: "a stretch", wildcard: false },
      },
    ]);
    if (!("picks" in entry)) throw new Error("expected a picks entry");
    expect(entry.header).toBe("Your picks");
    expect(entry.picks).toHaveLength(3);
    expect(entry.picks[0]).toEqual({ title: "A", year: 1999, reason: "because A", wildcard: false });
    expect(entry.picks[1].year).toBeUndefined(); // null year becomes undefined
    expect(entry.picks[2]).toEqual({ title: "C", year: 2020, reason: "a stretch", wildcard: true });
  });

  it("handles picks with no wildcard", () => {
    const [entry] = apiMessagesToEntries([
      {
        type: "picks",
        header: "The closest the catalog gets",
        picks: [{ id: 1, title: "A", year: 2001, explanation: "closest", wildcard: false }],
        wildcard: null,
      },
    ]);
    if (!("picks" in entry)) throw new Error("expected a picks entry");
    expect(entry.picks).toHaveLength(1);
    expect(entry.picks.some((p) => p.wildcard)).toBe(false);
  });

  it("maps a whole turn's messages in order", () => {
    const entries = apiMessagesToEntries([
      { type: "say", text: "one" },
      { type: "say", text: "two" },
    ]);
    expect(entries.map((e) => ("text" in e ? e.text : ""))).toEqual(["one", "two"]);
  });
});
