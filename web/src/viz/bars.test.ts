import { describe, expect, it } from "vitest";
import { barGeometry, tasteBarsHtml } from "./bars.ts";

describe("barGeometry", () => {
  it("returns nothing for no axes", () => {
    expect(barGeometry([])).toEqual([]);
  });

  it("normalises bar length against the strongest |weight| in the set", () => {
    const bars = barGeometry([
      { name: "a", weight: 1.0, confidence: 1 },
      { name: "b", weight: -0.5, confidence: 1 },
      { name: "c", weight: 0.25, confidence: 1 },
    ]);
    expect(bars[0].widthPct).toBeCloseTo(100);
    expect(bars[1].widthPct).toBeCloseTo(50);
    expect(bars[2].widthPct).toBeCloseTo(25);
  });

  it("reads the side from the weight's sign, counting zero as non-negative", () => {
    const [pos, neg, zero] = barGeometry([
      { name: "p", weight: 0.3, confidence: 1 },
      { name: "n", weight: -0.3, confidence: 1 },
      { name: "z", weight: 0, confidence: 1 },
    ]);
    expect(pos.sign).toBe(1);
    expect(neg.sign).toBe(-1);
    expect(zero.sign).toBe(1);
  });

  it("maps confidence into a visible opacity band and clamps out-of-range values", () => {
    const [lo, hi, over, under] = barGeometry([
      { name: "lo", weight: 1, confidence: 0 },
      { name: "hi", weight: 1, confidence: 1 },
      { name: "over", weight: 1, confidence: 2 },
      { name: "under", weight: 1, confidence: -1 },
    ]);
    expect(lo.opacity).toBeCloseTo(0.4);
    expect(hi.opacity).toBeCloseTo(1.0);
    expect(over.opacity).toBeCloseTo(1.0);
    expect(under.opacity).toBeCloseTo(0.4);
  });

  it("does not divide by zero when every weight is zero", () => {
    const [bar] = barGeometry([{ name: "z", weight: 0, confidence: 1 }]);
    expect(bar.widthPct).toBe(0);
    expect(Number.isFinite(bar.widthPct)).toBe(true);
  });
});

describe("tasteBarsHtml", () => {
  it("renders nothing when there is no signal yet", () => {
    expect(tasteBarsHtml([])).toBe("");
  });

  it("renders a row per axis carrying the side class and the width/opacity custom properties", () => {
    const html = tasteBarsHtml([
      { name: "atmospheric", weight: 0.8, confidence: 0.9 },
      { name: "gory", weight: -0.4, confidence: 0.5 },
    ]);
    expect(html).toContain("bar-row pos");
    expect(html).toContain("bar-row neg");
    expect(html).toContain("atmospheric");
    expect(html).toContain("gory");
    expect(html).toContain("--w:100.0%"); // the strongest bar fills its side
    expect(html).toContain("--w:50.0%");
  });

  it("escapes tag names so a hostile label can't inject markup", () => {
    const html = tasteBarsHtml([{ name: "<script>x</script>", weight: 1, confidence: 1 }]);
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
  });
});
