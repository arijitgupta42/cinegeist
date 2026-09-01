// View C — taste as a bar chart (plan.md §9.2). The plan calls this "the boring one you'll actually
// use... ugly, legible, and the one people screenshot" — a ranked, diverging bar chart of the
// strongest tag affinities the session has learned, positive to the right, negative to the left,
// with confidence carried by the fill's opacity (fainter = less sure of the direction).
//
// It reads the same taste axes the explanations and the other views read (DemoSession.tasteAxes) —
// "same data source as the CLI", plan.md §9.3 — so it can never disagree with the picks beside it.
// Geometry is a pure function so it is unit-tested without a DOM; the string builder just lays that
// geometry out in the demo's monospace chrome.

// The shape the chart needs from a taste axis. DemoSession.tasteAxes returns objects that satisfy
// this structurally, so the chart stays decoupled from how the axis was derived.
export interface BarAxis {
  name: string;
  weight: number; // signed affinity; sign picks the side and colour, magnitude the length
  confidence: number; // 0..1, how sure we are of the direction — drives the fill opacity
}

export interface Bar {
  name: string;
  weight: number;
  sign: 1 | -1;
  widthPct: number; // 0..100, |weight| as a fraction of the strongest bar shown
  opacity: number; // 0.4..1, from confidence — even the least certain bar stays visible
}

const MIN_OPACITY = 0.4;

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * Turn ranked taste axes into bar geometry. Lengths are normalised against the strongest bar in the
 * set, so the longest bar always fills its side and the rest are read relative to it. Opacity maps
 * confidence into [MIN_OPACITY, 1] so a low-confidence bar is faint but never invisible. Pure: no
 * DOM, no globals, same input → same output.
 */
export function barGeometry(axes: BarAxis[]): Bar[] {
  let max = 0;
  for (const a of axes) max = Math.max(max, Math.abs(a.weight));
  const denom = max > 0 ? max : 1; // all-zero (or empty) input → zero-length bars, no divide-by-zero
  return axes.map((a) => ({
    name: a.name,
    weight: a.weight,
    sign: a.weight < 0 ? -1 : 1,
    widthPct: (Math.abs(a.weight) / denom) * 100,
    opacity: MIN_OPACITY + (1 - MIN_OPACITY) * clamp01(a.confidence),
  }));
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

// A plain-text summary of one bar, for the row's aria-label — the visual state reachable as text
// (plan.md §9.3). "great cinematography — strong positive, high confidence".
function describe(bar: Bar): string {
  const dir = bar.sign > 0 ? "positive" : "negative";
  const strength = bar.widthPct >= 66 ? "strong" : bar.widthPct >= 33 ? "moderate" : "slight";
  const sure = bar.opacity >= 0.8 ? "high" : bar.opacity >= 0.6 ? "medium" : "low";
  return `${bar.name} — ${strength} ${dir}, ${sure} confidence`;
}

/**
 * Render the taste bars as an HTML string, in the demo's panel chrome. Bars grow from the centre on
 * insertion via a CSS keyframe reading the `--w` custom property; `prefers-reduced-motion` turns
 * that off in CSS. A small per-row delay staggers the cascade. Returns "" for an empty set so the
 * caller can decide what to show when there's no signal yet.
 */
export function tasteBarsHtml(axes: BarAxis[]): string {
  const bars = barGeometry(axes);
  if (bars.length === 0) return "";

  const rows = bars
    .map((bar, i) => {
      const delay = Math.min(i * 40, 400);
      return `
        <li class="bar-row ${bar.sign > 0 ? "pos" : "neg"}" aria-label="${escapeHtml(describe(bar))}">
          <span class="bar-label mono" title="${escapeHtml(bar.name)}">${escapeHtml(bar.name)}</span>
          <span class="bar-track" aria-hidden="true">
            <span class="bar-fill" style="--w:${bar.widthPct.toFixed(1)}%;--o:${bar.opacity.toFixed(2)};animation-delay:${delay}ms"></span>
          </span>
        </li>`;
    })
    .join("");

  return `
    <section class="viz-bars">
      <div class="panel-head">
        <span class="sq cyan"></span><span class="mono">What your taste looks like</span>
        <span class="mono progress">strongest tags, + and −</span>
      </div>
      <ul class="bars" role="list">${rows}</ul>
      <p class="bars-key mono" aria-hidden="true">
        <span class="sq cyan"></span> drawn toward &nbsp; <span class="sq magenta"></span> pushed away &nbsp;
        · fainter bars are less certain
      </p>
    </section>`;
}
