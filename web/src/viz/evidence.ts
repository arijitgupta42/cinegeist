// View B — the evidence graph (plan.md §9.2). "cinegeist profile show rendered as a picture, and the
// single most trust-building thing in the product. People believe recommendations they can trace."
//
// A node-link graph laid out with d3-force: the films you reacted to on the left, feeding the taste
// tags in the middle, which define YOU, from which the recommendations hang on the right —
// film → tag → you → pick. Edge thickness is contribution weight; edge colour is positive (drawn
// toward) or negative (pushed away). Click a tag to light up everything that produced it.
//
// buildEvidenceGraph is pure and unit-tested; the SVG rendering is verified in the browser. Under
// reduced motion the layout is solved synchronously and drawn once, with no settling animation.

import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";

export type NodeKind = "you" | "tag" | "film" | "pick";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  sign?: 1 | -1; // for tags and films: drawn-toward (+) or pushed-away (−)
  wildcard?: boolean;
}

export interface GraphLink {
  source: string;
  target: string;
  weight: number;
  sign: 1 | -1;
}

export interface EvidenceGraph {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface PickInput {
  filmId: number;
  title: string;
  tags: string[]; // the pick's top tag names, used to connect it to the taste tags
  wildcard?: boolean;
}

/**
 * Assemble the graph from the session's taste axes and the recommendations. The strongest axes
 * become tag nodes linked to YOU; their top contributing films become film nodes linked in; each
 * pick links to whichever shown tags it matches (or straight to YOU if none are shown). Pure.
 */
export function buildEvidenceGraph(
  axes: Array<{ position: number; name: string; weight: number; contributors: Array<{ filmId: number; title: string; value: number; contribution: number }> }>,
  picks: PickInput[],
  opts?: { maxTags?: number; maxFilmsPerTag?: number },
): EvidenceGraph {
  const maxTags = opts?.maxTags ?? 8;
  const maxFilmsPerTag = opts?.maxFilmsPerTag ?? 3;

  const shown = [...axes].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, maxTags);
  const shownByName = new Map(shown.map((a) => [a.name, a] as const));

  const nodes: GraphNode[] = [{ id: "you", kind: "you", label: "You" }];
  const links: GraphLink[] = [];
  const seenFilms = new Set<number>();

  for (const ax of shown) {
    const tagId = `tag:${ax.position}`;
    const sign: 1 | -1 = ax.weight >= 0 ? 1 : -1;
    nodes.push({ id: tagId, kind: "tag", label: ax.name, sign });
    links.push({ source: tagId, target: "you", weight: Math.abs(ax.weight), sign });
    for (const c of ax.contributors.slice(0, maxFilmsPerTag)) {
      const filmId = `film:${c.filmId}`;
      if (!seenFilms.has(c.filmId)) {
        seenFilms.add(c.filmId);
        nodes.push({ id: filmId, kind: "film", label: c.title, sign: c.value >= 0 ? 1 : -1 });
      }
      links.push({ source: filmId, target: tagId, weight: Math.abs(c.contribution), sign: c.contribution >= 0 ? 1 : -1 });
    }
  }

  for (const p of picks) {
    const pickId = `pick:${p.filmId}`;
    nodes.push({ id: pickId, kind: "pick", label: p.title, wildcard: p.wildcard });
    const matched = p.tags.map((t) => shownByName.get(t)).filter((a): a is (typeof shown)[number] => !!a);
    if (matched.length === 0) {
      links.push({ source: pickId, target: "you", weight: 0.3, sign: 1 });
    } else {
      for (const ax of matched) links.push({ source: pickId, target: `tag:${ax.position}`, weight: Math.abs(ax.weight), sign: 1 });
    }
  }

  return { nodes, links };
}

// -- rendering -----------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";
const HEIGHT = 460;
const COLORS = {
  you: "#ffffff",
  pos: "#3fd2fb",
  neg: "#f365ff",
  pickWild: "#f365ff",
  dim: "#5b5b62",
};

type SimNode = GraphNode & SimulationNodeDatum;
type SimLink = SimulationLinkDatum<SimNode> & { weight: number; sign: 1 | -1 };

function nodeColor(n: GraphNode): string {
  if (n.kind === "you") return COLORS.you;
  if (n.kind === "pick") return n.wildcard ? COLORS.pickWild : COLORS.pos;
  return n.sign === -1 ? COLORS.neg : COLORS.pos;
}

function radius(n: GraphNode): number {
  if (n.kind === "you") return 11;
  if (n.kind === "pick") return 8;
  if (n.kind === "tag") return 6;
  return 5;
}

export class EvidenceView {
  private sim?: Simulation<SimNode, SimLink>;
  private svg?: SVGSVGElement;
  private activeTag: string | null = null;

  constructor(
    private container: HTMLElement,
    private graph: EvidenceGraph,
    private reducedMotion: boolean,
  ) {}

  render(): void {
    const width = this.container.clientWidth || 640;
    const cx = width / 2;
    const cy = HEIGHT / 2;

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${HEIGHT}`);
    svg.setAttribute("class", "evidence-svg");
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(HEIGHT));
    this.svg = svg;
    this.container.appendChild(svg);

    const nodes: SimNode[] = this.graph.nodes.map((n) => ({ ...n }));
    const links: SimLink[] = this.graph.links.map((l) => ({ source: l.source, target: l.target, weight: l.weight, sign: l.sign }));

    const maxW = Math.max(1e-6, ...links.map((l) => l.weight));

    const linkEls = links.map((l) => {
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("stroke", l.sign === -1 ? COLORS.neg : COLORS.pos);
      line.setAttribute("stroke-width", String(0.6 + 2.6 * (l.weight / maxW)));
      line.setAttribute("stroke-opacity", "0.35");
      svg.appendChild(line);
      return line;
    });

    const nodeGroups = nodes.map((n) => {
      const g = document.createElementNS(SVG_NS, "g");
      g.setAttribute("class", `ev-node ev-${n.kind}`);
      const circle = document.createElementNS(SVG_NS, "circle");
      circle.setAttribute("r", String(radius(n)));
      circle.setAttribute("fill", nodeColor(n));
      circle.setAttribute("fill-opacity", n.kind === "film" ? "0.85" : "1");
      g.appendChild(circle);
      if (n.kind !== "film") {
        const text = document.createElementNS(SVG_NS, "text");
        text.setAttribute("x", String(radius(n) + 4));
        text.setAttribute("y", "3.5");
        text.setAttribute("class", "ev-label");
        text.textContent = n.label.length > 22 ? `${n.label.slice(0, 21)}…` : n.label;
        g.appendChild(text);
      } else {
        g.appendChild(titleEl(n.label)); // films get a hover <title> to stay uncluttered
      }
      if (n.kind === "tag") {
        g.style.cursor = "pointer";
        g.addEventListener("click", () => this.highlightTag(n.id, nodes, links, linkEls, nodeGroups));
      }
      svg.appendChild(g);
      return g;
    });

    const targetX = (n: SimNode): number =>
      n.kind === "film" ? cx - width * 0.34 : n.kind === "pick" ? cx + width * 0.34 : n.kind === "tag" ? cx - width * 0.04 : cx;

    this.sim = forceSimulation<SimNode>(nodes)
      .force("link", forceLink<SimNode, SimLink>(links).id((d) => d.id).distance(58).strength(0.25))
      .force("charge", forceManyBody().strength(-150))
      .force("x", forceX<SimNode>().x(targetX).strength(0.28))
      .force("y", forceY<SimNode>(cy).strength(0.08))
      .force("collide", forceCollide<SimNode>().radius((d) => radius(d) + 10));

    const draw = (): void => {
      for (let i = 0; i < links.length; i++) {
        const s = links[i].source as SimNode;
        const t = links[i].target as SimNode;
        linkEls[i].setAttribute("x1", String(s.x));
        linkEls[i].setAttribute("y1", String(s.y));
        linkEls[i].setAttribute("x2", String(t.x));
        linkEls[i].setAttribute("y2", String(t.y));
      }
      for (let i = 0; i < nodes.length; i++) {
        nodeGroups[i].setAttribute("transform", `translate(${nodes[i].x ?? cx}, ${nodes[i].y ?? cy})`);
      }
    };

    if (this.reducedMotion) {
      this.sim.stop();
      for (let i = 0; i < 320; i++) this.sim.tick();
      draw();
    } else {
      this.sim.on("tick", draw);
    }
  }

  // Clicking a tag dims everything not attached to it, so the films that produced it and the picks
  // that lean on it stand out — the graph's version of "click a tag to see the evidence" (§9.2).
  private highlightTag(
    tagId: string,
    nodes: SimNode[],
    links: SimLink[],
    linkEls: SVGLineElement[],
    nodeGroups: SVGGElement[],
  ): void {
    const connected = new Set<string>([tagId, "you"]);
    links.forEach((l) => {
      const s = (l.source as SimNode).id ?? (l.source as unknown as string);
      const t = (l.target as SimNode).id ?? (l.target as unknown as string);
      if (s === tagId || t === tagId) {
        connected.add(s);
        connected.add(t);
      }
    });
    const active = this.activeTag === tagId ? null : tagId;
    this.activeTag = active;

    nodes.forEach((n, i) => {
      const on = active === null || connected.has(n.id);
      nodeGroups[i].setAttribute("opacity", on ? "1" : "0.15");
    });
    links.forEach((l, i) => {
      const s = (l.source as SimNode).id;
      const t = (l.target as SimNode).id;
      const on = active === null || s === tagId || t === tagId;
      linkEls[i].setAttribute("stroke-opacity", on ? "0.7" : "0.05");
    });
  }

  dispose(): void {
    this.sim?.stop();
    this.svg?.remove();
  }
}

function titleEl(label: string): SVGTitleElement {
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = label;
  return title;
}
