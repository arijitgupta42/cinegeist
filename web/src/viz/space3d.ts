// View A — the 3D taste-space map (plan.md §9.1, §9.2, the hero). The 2,000 shard films sit at their
// precomputed UMAP coordinates as one glowing point cloud, coloured by taste-space cluster and faded
// where the shard's coverage runs thin (§8.4). A bright marker walks to the barycentre of the films
// the visitor reacted to, trailing the path it took; the recommendations pulse, the wildcard pulses
// in magenta further out. "Don't visualise the catalog — visualise the user moving through it."
//
// three.js is imported dynamically inside mount(), so it's a separate chunk that only downloads when
// the Learn tab first opens the map (plan.md §8.7). The pure maths (barycentre, clustering, the fit
// transform) lives in map-model.ts and is unit-tested; this file is the WebGL wiring, which is
// verified in the browser. A static 2D fallback (map2d.ts) covers reduced-motion and no-WebGL.

import type * as THREE from "three";
import { applyTransform, fitTransform, type FitTransform } from "./map-model.ts";

export interface MapFilm {
  title: string;
  year: number | null;
  tags: string[];
}

export interface MapModel {
  xyz: Float32Array; // nFilms × 3, raw UMAP coordinates
  nFilms: number;
  colors: Float32Array; // nFilms × 3, rgb per film from its cluster
  coverage: Float32Array; // nFilms, in [0, 1]
  barycenter: number[] | null; // raw coordinate of the taste marker
  trail: number[][]; // raw coordinates, oldest → newest
  pickIndices: number[]; // confident picks, as shard indices
  wildcardIndex: number | null;
  filmAt: (i: number) => MapFilm;
  reducedMotion: boolean;
}

/** Whether this browser can give us a WebGL context at all. */
export function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

const POINT_VERT = `
  attribute vec3 color;
  attribute float coverage;
  varying vec3 vColor;
  varying float vCoverage;
  uniform float uSize;
  void main() {
    vColor = color;
    vCoverage = coverage;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize / max(0.1, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

// A soft round point; thinly-covered films fade toward the background rather than being padded away
// (plan.md §8.4 — you watch the sparse regions instead of being told about them afterward).
const POINT_FRAG = `
  varying vec3 vColor;
  varying float vCoverage;
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    float glow = smoothstep(0.5, 0.0, d);
    float alpha = glow * mix(0.12, 0.85, vCoverage);
    gl_FragColor = vec4(vColor, alpha);
  }`;

const MARKER_VERT = `
  uniform float uSize;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize / max(0.1, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const MARKER_FRAG = `
  uniform vec3 uColor;
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    float core = smoothstep(0.5, 0.0, d);
    float ring = smoothstep(0.5, 0.42, d) * 0.6;
    gl_FragColor = vec4(uColor, max(core * core, ring));
  }`;

const TRAIL_VERT = `
  attribute float alpha;
  varying float vAlpha;
  void main() {
    vAlpha = alpha;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const TRAIL_FRAG = `
  uniform vec3 uColor;
  varying float vAlpha;
  void main() { gl_FragColor = vec4(uColor, vAlpha); }`;

export class TasteMap {
  private three!: typeof THREE;
  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private group!: THREE.Group; // holds the cloud + markers, rotated by drag / auto-rotate
  private cloud!: THREE.Points;
  private marker?: THREE.Points;
  private picks?: THREE.Points;
  private wildcard?: THREE.Points;
  private trailLine?: THREE.Line;
  private raycaster!: THREE.Raycaster;
  private pointer!: THREE.Vector2;
  private tip!: HTMLDivElement;
  private t!: FitTransform;

  private raf = 0;
  private disposed = false;
  private rotX = 0.35;
  private rotY = 0.2;
  private dist = 3.2;
  private autoRotate: boolean;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private walk = { active: false, start: 0, dur: 0 }; // the marker's walk along the trail
  private cleanups: Array<() => void> = [];

  constructor(
    private container: HTMLElement,
    private model: MapModel,
  ) {
    this.autoRotate = !model.reducedMotion;
  }

  /** Load three.js, build the scene, and start rendering. Throws if WebGL can't be initialised. */
  async mount(): Promise<void> {
    this.three = await import("three");
    if (this.disposed) return; // torn down while the chunk was loading

    const T = this.three;
    const width = this.container.clientWidth || 640;
    const height = this.container.clientHeight || 420;

    this.renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(width, height);
    this.container.appendChild(this.renderer.domElement);

    this.scene = new T.Scene();
    this.camera = new T.PerspectiveCamera(55, width / height, 0.01, 100);
    this.camera.position.set(0, 0, this.dist);

    this.group = new T.Group();
    this.scene.add(this.group);

    this.t = fitTransform(this.model.xyz, this.model.nFilms);
    this.buildCloud();
    this.buildTrail();
    this.buildMarkers();

    this.raycaster = new T.Raycaster();
    this.raycaster.params.Points = { threshold: 0.035 };
    this.pointer = new T.Vector2(-2, -2);
    this.buildTooltip();
    this.attachControls();

    if (!this.model.reducedMotion && this.model.trail.length > 1) {
      this.walk = { active: true, start: performance.now(), dur: Math.min(1800, 500 + 200 * this.model.trail.length) };
    }
    this.loop();
  }

  private buildCloud(): void {
    const T = this.three;
    const n = this.model.nFilms;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = applyTransform([this.model.xyz[i * 3], this.model.xyz[i * 3 + 1], this.model.xyz[i * 3 + 2]], this.t);
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.BufferAttribute(positions, 3));
    geo.setAttribute("color", new T.BufferAttribute(this.model.colors, 3));
    geo.setAttribute("coverage", new T.BufferAttribute(this.model.coverage, 1));
    const mat = new T.ShaderMaterial({
      uniforms: { uSize: { value: 26 * this.renderer.getPixelRatio() } },
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      transparent: true,
      depthWrite: false,
      blending: T.AdditiveBlending,
    });
    this.cloud = new T.Points(geo, mat);
    this.group.add(this.cloud);
  }

  private buildTrail(): void {
    const trail = this.model.trail;
    if (trail.length < 2) return;
    const T = this.three;
    const positions = new Float32Array(trail.length * 3);
    const alpha = new Float32Array(trail.length);
    trail.forEach((raw, i) => {
      const p = applyTransform(raw, this.t);
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
      alpha[i] = 0.05 + 0.55 * (i / (trail.length - 1)); // older = fainter
    });
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.BufferAttribute(positions, 3));
    geo.setAttribute("alpha", new T.BufferAttribute(alpha, 1));
    const mat = new T.ShaderMaterial({
      uniforms: { uColor: { value: new T.Color(0xffffff) } },
      vertexShader: TRAIL_VERT,
      fragmentShader: TRAIL_FRAG,
      transparent: true,
      depthWrite: false,
    });
    this.trailLine = new T.Line(geo, mat);
    if (this.walk.active) geo.setDrawRange(0, 1); // revealed as the marker walks
    this.group.add(this.trailLine);
  }

  private markerPoints(coords: number[][], color: number, size: number): THREE.Points {
    const T = this.three;
    const positions = new Float32Array(coords.length * 3);
    coords.forEach((raw, i) => {
      const p = applyTransform(raw, this.t);
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
    });
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.BufferAttribute(positions, 3));
    const mat = new T.ShaderMaterial({
      uniforms: { uSize: { value: size * this.renderer.getPixelRatio() }, uColor: { value: new T.Color(color) } },
      vertexShader: MARKER_VERT,
      fragmentShader: MARKER_FRAG,
      transparent: true,
      depthWrite: false,
      blending: T.AdditiveBlending,
    });
    return new T.Points(geo, mat);
  }

  private buildMarkers(): void {
    const m = this.model;
    if (m.pickIndices.length) {
      this.picks = this.markerPoints(m.pickIndices.map((i) => this.rawAt(i)), 0x3fd2fb, 60);
      this.group.add(this.picks);
    }
    if (m.wildcardIndex !== null) {
      this.wildcard = this.markerPoints([this.rawAt(m.wildcardIndex)], 0xf365ff, 66);
      this.group.add(this.wildcard);
    }
    if (m.barycenter) {
      const start = this.walk.active && m.trail.length ? m.trail[0] : m.barycenter;
      this.marker = this.markerPoints([start], 0xffffff, 90);
      this.group.add(this.marker);
    }
  }

  private rawAt(i: number): number[] {
    return [this.model.xyz[i * 3], this.model.xyz[i * 3 + 1], this.model.xyz[i * 3 + 2]];
  }

  // -- interaction -------------------------------------------------------------------

  private attachControls(): void {
    const el = this.renderer.domElement;
    el.style.touchAction = "none";
    el.style.cursor = "grab";

    const onDown = (e: PointerEvent): void => {
      this.dragging = true;
      this.autoRotate = false;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
      el.style.cursor = "grabbing";
    };
    const onMove = (e: PointerEvent): void => {
      const rect = el.getBoundingClientRect();
      this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      if (!this.dragging) return;
      this.rotY += (e.clientX - this.lastX) * 0.006;
      this.rotX += (e.clientY - this.lastY) * 0.006;
      this.rotX = Math.max(-1.4, Math.min(1.4, this.rotX));
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    };
    const onUp = (e: PointerEvent): void => {
      this.dragging = false;
      el.releasePointerCapture?.(e.pointerId);
      el.style.cursor = "grab";
    };
    const onLeave = (): void => {
      this.pointer.set(-2, -2);
    };
    const onWheel = (e: WheelEvent): void => {
      e.preventDefault();
      this.dist = Math.max(1.5, Math.min(7, this.dist + Math.sign(e.deltaY) * 0.3));
    };

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointerleave", onLeave);
    el.addEventListener("wheel", onWheel, { passive: false });
    const onResize = (): void => this.resize();
    window.addEventListener("resize", onResize);
    this.cleanups.push(() => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointerleave", onLeave);
      el.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
    });
  }

  private buildTooltip(): void {
    this.tip = document.createElement("div");
    this.tip.className = "map-tip";
    this.tip.hidden = true;
    this.container.appendChild(this.tip);
  }

  private updateHover(): void {
    if (this.dragging || this.pointer.x < -1.5) {
      this.tip.hidden = true;
      return;
    }
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObject(this.cloud);
    if (hit.length === 0 || hit[0].index === undefined) {
      this.tip.hidden = true;
      return;
    }
    const film = this.model.filmAt(hit[0].index);
    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = ((this.pointer.x + 1) / 2) * rect.width;
    const y = ((1 - this.pointer.y) / 2) * rect.height;
    this.tip.innerHTML = `<strong>${escapeHtml(film.title)}</strong>${
      film.year ? ` <span>${film.year}</span>` : ""
    }${film.tags.length ? `<em>${film.tags.map(escapeHtml).join(" · ")}</em>` : ""}`;
    this.tip.style.left = `${Math.min(x + 14, rect.width - 8)}px`;
    this.tip.style.top = `${Math.max(y - 8, 8)}px`;
    this.tip.hidden = false;
  }

  private resize(): void {
    if (this.disposed) return;
    const w = this.container.clientWidth || 640;
    const h = this.container.clientHeight || 420;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // -- render loop -------------------------------------------------------------------

  private loop = (): void => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    const now = performance.now();

    if (this.autoRotate) this.rotY += 0.0016;
    this.group.rotation.x = this.rotX;
    this.group.rotation.y = this.rotY;
    this.camera.position.set(0, 0, this.dist);

    // Pulse the recommendation and barycentre markers (unless the visitor asked for less motion).
    const pulse = this.model.reducedMotion ? 1 : 1 + 0.18 * Math.sin(now * 0.005);
    if (this.picks) (this.picks.material as THREE.ShaderMaterial).uniforms.uSize.value = 60 * this.renderer.getPixelRatio() * pulse;
    if (this.wildcard)
      (this.wildcard.material as THREE.ShaderMaterial).uniforms.uSize.value = 66 * this.renderer.getPixelRatio() * pulse;

    this.advanceWalk(now);
    this.updateHover();
    this.renderer.render(this.scene, this.camera);
  };

  // Walk the marker along the trail on first show, revealing the trail behind it (plan.md §9.1).
  private advanceWalk(now: number): void {
    if (!this.walk.active || !this.marker) return;
    const trail = this.model.trail;
    const p = Math.min(1, (now - this.walk.start) / this.walk.dur);
    const f = p * (trail.length - 1);
    const i = Math.min(trail.length - 2, Math.floor(f));
    const frac = f - i;
    const a = applyTransform(trail[i], this.t);
    const b = applyTransform(trail[i + 1], this.t);
    const attr = this.marker.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.setXYZ(0, a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac, a[2] + (b[2] - a[2]) * frac);
    attr.needsUpdate = true;
    this.trailLine?.geometry.setDrawRange(0, Math.max(2, Math.ceil(f) + 1));
    if (p >= 1) {
      this.walk.active = false;
      this.trailLine?.geometry.setDrawRange(0, trail.length);
    }
  }

  /** Tear down: stop the loop, drop GPU resources, remove the canvas and listeners. */
  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    for (const c of this.cleanups) c();
    this.cleanups = [];
    if (!this.renderer) return;
    this.scene.traverse((o) => {
      const any = o as unknown as { geometry?: THREE.BufferGeometry; material?: THREE.Material };
      any.geometry?.dispose();
      any.material?.dispose();
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.tip?.remove();
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
