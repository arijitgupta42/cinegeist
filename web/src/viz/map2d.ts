// The static 2D fallback for the taste-space map (plan.md §9.3 — "3D needs a 2D fallback for
// low-power devices and prefers-reduced-motion"). It flattens the film cloud onto its two
// highest-variance axes and draws it once to a canvas: no animation, no WebGL, no three.js. Same
// data as the 3D map (the MapModel), so the picture agrees with it — coloured by cluster, faded
// where coverage is thin, with the taste marker, its trail, and the pulsing picks shown as static
// marks. The text summary beside it (built in the Learn wiring) is what makes it fully non-visual.

import { bestTwoAxes } from "./map-model.ts";
import type { MapModel } from "./space3d.ts";

const HEIGHT = 420;
const PAD = 28;

export function renderMap2D(container: HTMLElement, model: MapModel): void {
  const width = container.clientWidth || 640;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const canvas = document.createElement("canvas");
  canvas.width = width * dpr;
  canvas.height = HEIGHT * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${HEIGHT}px`;
  canvas.className = "map-2d";
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  const [ax, ay] = bestTwoAxes(model.xyz, model.nFilms);

  // Fit the two chosen axes into the padded canvas.
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < model.nFilms; i++) {
    const x = model.xyz[i * 3 + ax];
    const y = model.xyz[i * 3 + ay];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const toCanvas = (rawX: number, rawY: number): [number, number] => [
    PAD + ((rawX - minX) / spanX) * (width - 2 * PAD),
    HEIGHT - PAD - ((rawY - minY) / spanY) * (HEIGHT - 2 * PAD), // invert y for screen
  ];
  const projRaw = (raw: number[]): [number, number] => toCanvas(raw[ax], raw[ay]);

  // Film cloud: each point coloured by its cluster, faded where coverage is thin. Kept dim — it is
  // the backdrop, not the subject, so the marker, trail and region labels read on top of it.
  for (let i = 0; i < model.nFilms; i++) {
    const [cx, cy] = toCanvas(model.xyz[i * 3 + ax], model.xyz[i * 3 + ay]);
    const r = Math.round(model.colors[i * 3] * 255);
    const g = Math.round(model.colors[i * 3 + 1] * 255);
    const b = Math.round(model.colors[i * 3 + 2] * 255);
    ctx.fillStyle = `rgba(${r},${g},${b},${(0.1 + 0.45 * model.coverage[i]).toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(cx, cy, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }

  // Named taste regions, floated at each cluster centroid — the same labels the 3D map shows, so a
  // reader can place the marker without the legend.
  ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.textBaseline = "middle";
  for (const region of model.regions) {
    const [x, y] = projRaw(region.coord);
    const label = region.label.toUpperCase();
    const w = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(x - w / 2 - 5, y - 9, w + 10, 18);
    ctx.fillStyle = `rgb(${Math.round(region.color[0] * 255)},${Math.round(region.color[1] * 255)},${Math.round(region.color[2] * 255)})`;
    ctx.textAlign = "center";
    ctx.fillText(label, x, y + 1);
  }
  ctx.textAlign = "start";

  // Trail: a faint polyline through the barycentre's path, brightening toward the present.
  if (model.trail.length > 1) {
    for (let i = 1; i < model.trail.length; i++) {
      const [x0, y0] = projRaw(model.trail[i - 1]);
      const [x1, y1] = projRaw(model.trail[i]);
      ctx.strokeStyle = `rgba(255,255,255,${(0.08 + 0.4 * (i / (model.trail.length - 1))).toFixed(3)})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
  }

  // Recommendation marks (cyan) and the wildcard (magenta).
  const ring = (raw: number[], color: string, radius: number): void => {
    const [x, y] = projRaw(raw);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  };
  for (const i of model.pickIndices) ring(rawAt(model, i), "#3fd2fb", 6);
  if (model.wildcardIndex !== null) ring(rawAt(model, model.wildcardIndex), "#f365ff", 7);

  // The taste marker last, so it sits on top: a filled white dot with a halo.
  if (model.barycenter) {
    const [x, y] = projRaw(model.barycenter);
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  container.appendChild(canvas);
}

function rawAt(model: MapModel, i: number): number[] {
  return [model.xyz[i * 3], model.xyz[i * 3 + 1], model.xyz[i * 3 + 2]];
}
