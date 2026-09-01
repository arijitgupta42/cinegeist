// Wiring for the taste-space map inside the Learn tab: turn the live session and the shard into a
// MapModel, then mount it as 3D (space3d) or, for reduced-motion / no-WebGL, the static 2D fallback
// (map2d). It also produces the cluster legend and a plain-text summary so the map's state is
// reachable without the picture (plan.md §9.3). The film clustering is the same for every visitor,
// so it's computed once per shard and cached.

import { filmTopTags, type DecodedShard } from "../shard.ts";
import type { DemoSession } from "../session.ts";
import {
  barycenter,
  characteristicTags,
  clusterPalette,
  cumulativeBarycenters,
  kmeans,
  type WeightedPoint,
} from "./map-model.ts";
import { hasWebGL, TasteMap, type MapModel } from "./space3d.ts";
import { renderMap2D } from "./map2d.ts";

const N_CLUSTERS = 7;

export interface LegendEntry {
  color: string;
  label: string;
}

export interface MapBuild {
  model: MapModel;
  legend: LegendEntry[];
  summary: string;
}

interface Clustering {
  colors: Float32Array; // nFilms × 3
  labels: Int32Array;
  palette: number[][];
}

const clusterCache = new WeakMap<DecodedShard, Clustering>();

function clusterShard(shard: DecodedShard): Clustering {
  const cached = clusterCache.get(shard);
  if (cached) return cached;

  const { labels } = kmeans(shard.xyz, shard.nFilms, 3, N_CLUSTERS, shard.seed || 1);
  const palette = clusterPalette(N_CLUSTERS);
  const colors = new Float32Array(shard.nFilms * 3);
  for (let i = 0; i < shard.nFilms; i++) {
    const c = palette[labels[i]] ?? [1, 1, 1];
    colors[i * 3] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }
  const result = { colors, labels, palette };
  clusterCache.set(shard, result);
  return result;
}

function rgb(c: number[]): string {
  return `rgb(${Math.round(c[0] * 255)}, ${Math.round(c[1] * 255)}, ${Math.round(c[2] * 255)})`;
}

function topTagName(shard: DecodedShard, i: number): string {
  const tags = filmTopTags(shard, i);
  return tags.length ? (shard.tagNames.get(tags[0].position) ?? "") : "";
}

/** Build the map model, legend, and text summary from the current session. */
export function buildMap(
  session: DemoSession,
  shard: DecodedShard,
  picks: { pickIndices: number[]; wildcardIndex: number | null },
): MapBuild {
  const { colors, labels, palette } = clusterShard(shard);

  const points: WeightedPoint[] = session
    .reactionWeights()
    .map(({ index, w }) => ({ w, coord: [shard.xyz[index * 3], shard.xyz[index * 3 + 1], shard.xyz[index * 3 + 2]] }));
  const centroid = barycenter(points);
  const trail = cumulativeBarycenters(points, 2);

  const model: MapModel = {
    xyz: shard.xyz,
    nFilms: shard.nFilms,
    colors,
    coverage: shard.coverage,
    barycenter: centroid,
    trail,
    pickIndices: picks.pickIndices,
    wildcardIndex: picks.wildcardIndex,
    filmAt: (i) => {
      const f = shard.films[i];
      return {
        title: f.title,
        year: f.year,
        tags: filmTopTags(shard, i)
          .slice(0, 3)
          .map((t) => shard.tagNames.get(t.position) ?? "")
          .filter(Boolean),
      };
    },
    reducedMotion: prefersReducedMotion(),
  };

  // Legend: label each cluster with its most characteristic tag, largest clusters first, and dedupe
  // so two regions never carry the same word.
  const ranked = characteristicTags(labels, N_CLUSTERS, (i) => {
    const tags = filmTopTags(shard, i);
    return tags.length ? tags[0].position : -1;
  });
  const sizes = new Array(N_CLUSTERS).fill(0);
  for (let i = 0; i < labels.length; i++) sizes[labels[i]]++;
  const order = [...Array(N_CLUSTERS).keys()].sort((a, b) => sizes[b] - sizes[a]);
  const used = new Set<number>();
  const legend: LegendEntry[] = [];
  for (const c of order) {
    const pick = ranked[c].find((t) => !used.has(t));
    if (pick !== undefined) used.add(pick);
    legend.push({
      color: rgb(palette[c]),
      label: pick !== undefined ? (shard.tagNames.get(pick) ?? `region ${c + 1}`) : `region ${c + 1}`,
    });
  }

  return { model, legend, summary: summarise(shard, centroid) };
}

// Text alternative: name the region the marker sits in and the films nearest it (plan.md §9.3).
function summarise(shard: DecodedShard, centroid: number[] | null): string {
  if (!centroid) return "";
  let nearest = -1;
  let nearestD = Infinity;
  const near: Array<{ i: number; d: number }> = [];
  for (let i = 0; i < shard.nFilms; i++) {
    const dx = shard.xyz[i * 3] - centroid[0];
    const dy = shard.xyz[i * 3 + 1] - centroid[1];
    const dz = shard.xyz[i * 3 + 2] - centroid[2];
    const d = dx * dx + dy * dy + dz * dz;
    near.push({ i, d });
    if (d < nearestD) {
      nearestD = d;
      nearest = i;
    }
  }
  near.sort((a, b) => a.d - b.d);
  const titles = near.slice(0, 3).map((x) => shard.films[x.i].title);
  const clusterLabel = nearest >= 0 ? topTagName(shard, nearest) : "";
  const region = clusterLabel ? ` in the ${clusterLabel} region of taste-space` : "";
  return `Your taste marker sits${region}. The films nearest it are ${titles.join(", ")}.`;
}

export function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

/**
 * Mount the map into `container`. Returns the live 3D instance to dispose later, or null when the
 * static 2D fallback was drawn (reduced-motion, no WebGL, or a failed GL init).
 */
export async function mountMap(container: HTMLElement, model: MapModel): Promise<TasteMap | null> {
  if (model.reducedMotion || !hasWebGL()) {
    renderMap2D(container, model);
    return null;
  }
  const map = new TasteMap(container, model);
  try {
    await map.mount();
    return map;
  } catch {
    map.dispose();
    container.replaceChildren();
    renderMap2D(container, model);
    return null;
  }
}
