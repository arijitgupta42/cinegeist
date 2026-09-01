"""Reduce the full catalog to the browser shard: sample, compress, pack (plan.md §8.3).

The full genome is ~16,000 × 1,128 float32 — far too big for a web page. This turns it into a
bundle of a few hundred KB that the demo scores in plain JavaScript, without giving up the thing
that makes the recommender honest: a real, diverse sample of the catalog rather than a flattering
shortlist of blockbusters.

The pipeline, all deterministic:

1. **Sample** ~2,000 genome-covered films, stratified across decades and popularity bands so
   obscure films survive on purpose (an all-blockbuster demo makes the recommender look stupid).
2. **Compress** their genome rows with a truncated SVD to 96 components, then quantise to int8 with
   a per-component scale. Cosine in the compressed space tracks cosine in the full space closely
   enough for a demo (rank correlation ≈ 0.96), at a fraction of the size.
3. **Embed** the same rows in 3D with PCA (linear, so the marker and trail in the visualization
   project correctly), sign-canonicalised so the bundle is byte-reproducible.
4. **Keep** each film's top tags for templated explanations, plus plain metadata.

The result is a ``shard.json`` (metadata, the int8 scales, the binary layout) and a packed
``shard.bin`` (int8 vectors, then float32 xyz). Per-film coverage — the honesty byte — is added by
a later PR; the binary layout leaves room for it at the end so vector and xyz offsets don't move.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

SHARD_VERSION = 1
TARGET_FILMS = 2000
SVD_COMPONENTS = 96
TOP_TAGS_PER_FILM = 12
_SAMPLE_SEED = 7

# Fills an unused top-tag slot for a film with fewer than TOP_TAGS_PER_FILM non-zero tags. No real
# genome position reaches it (there are ~1,128 tags), so the demo reads slots until it hits this.
TAG_SENTINEL = 65535


@dataclass(frozen=True)
class ShardBuild:
    """The two files that make up a shard: the JSON side-car and the packed binary."""

    manifest: dict
    binary: bytes


# -- sampling ------------------------------------------------------------------------


def stratified_sample(
    years: np.ndarray, popularity: np.ndarray, target: int, *, seed: int = _SAMPLE_SEED
) -> np.ndarray:
    """Choose ``target`` film indices spread across decades and popularity (plan.md §8.3).

    Decades are weighted by ``sqrt(size)`` so a huge modern decade doesn't crowd out older ones and
    a tiny early decade isn't erased; within each decade films are taken at an even stride along the
    popularity order, which deliberately keeps the obscure tail rather than only the hits. Fully
    deterministic given ``seed``. Returns sorted indices into the input arrays.
    """
    n = len(years)
    if target >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)

    decade = np.where(years > 0, (years // 10) * 10, -1)
    cells: dict[int, np.ndarray] = {}
    for d in np.unique(decade):
        cells[int(d)] = np.flatnonzero(decade == d)

    weights = {d: np.sqrt(len(idx)) for d, idx in cells.items()}
    total_w = sum(weights.values())
    picked: list[int] = []
    for d, idx in cells.items():
        quota = min(len(idx), max(1, round(target * weights[d] / total_w)))
        order = idx[np.argsort(popularity[idx], kind="stable")]
        stride = np.linspace(0, len(order) - 1, quota).round().astype(int)
        picked.extend(order[np.unique(stride)].tolist())

    picked_set = set(picked)
    if len(picked_set) > target:
        picked = rng.permutation(list(picked_set))[:target].tolist()
    elif len(picked_set) < target:
        rest = np.array([i for i in range(n) if i not in picked_set])
        picked.extend(rng.permutation(rest)[: target - len(picked_set)].tolist())
    return np.array(sorted(set(picked)))


# -- compression ---------------------------------------------------------------------


def _sign_canonicalise(components: np.ndarray) -> np.ndarray:
    """Flip each component so its largest-magnitude entry is positive (reproducible SVD/PCA).

    SVD and PCA are only defined up to a per-component sign, and the sign a library picks can vary.
    Fixing it makes the packed bytes identical on every machine that regenerates the shard.
    """
    flip = np.sign(
        components[np.argmax(np.abs(components), axis=0), np.arange(components.shape[1])]
    )
    flip[flip == 0] = 1.0
    return components * flip


def svd_project(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Project rows onto their top ``k`` right singular vectors (truncated SVD).

    Returns ``(projected, components)`` where ``projected`` is ``(n, k)`` and ``components`` is the
    ``(d, k)`` basis (sign-canonicalised). Fitting on the shard's own rows keeps the compression
    self-consistent: the demo builds profiles from these same films, in this same space.
    """
    k = min(k, *matrix.shape)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    components = _sign_canonicalise(vt[:k].T)
    return matrix @ components, components


def quantise_int8(projected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantise projected vectors to int8 with a per-component scale.

    Each of the ``k`` columns gets its own scale (its max absolute value over 127), so a component
    with a small dynamic range keeps its precision. Returns ``(int8 matrix, scales)`` such that
    ``int8 · scale`` reconstructs the projection; a zero column gets scale 1 to avoid dividing by 0.
    """
    scales = np.abs(projected).max(axis=0) / 127.0
    scales[scales == 0.0] = 1.0
    q = np.clip(np.round(projected / scales), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def pca_3d(matrix: np.ndarray) -> np.ndarray:
    """Embed rows in 3D by PCA (mean-centred, top-3 components), sign-canonicalised.

    Linear on purpose (plan.md §9.2): the profile marker is the barycenter of reacted films, and a
    linear map lets any new point project with a matmul. Returns an ``(n, 3)`` float32 array.
    """
    centred = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = _sign_canonicalise(vt[: min(3, matrix.shape[1])].T)
    coords = centred @ components
    if coords.shape[1] < 3:  # pad a degenerate low-rank space to a full xyz
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    return coords.astype(np.float32)


def top_tags(vector: np.ndarray, k: int = TOP_TAGS_PER_FILM) -> list[tuple[int, int]]:
    """The ``k`` highest-relevance tag positions of one genome row, as ``(position, score)``.

    ``score`` is the relevance rescaled to a 0–255 byte, so the demo can template explanations
    ("mostly *atmospheric*, *slow*") without shipping the full 1,128-dim row. Zero-relevance tags
    are dropped, so a sparse row returns fewer than ``k``.
    """
    order = np.argsort(vector)[::-1][:k]
    out: list[tuple[int, int]] = []
    for pos in order:
        rel = float(vector[pos])
        if rel <= 0.0:
            break
        out.append((int(pos), int(round(min(1.0, rel) * 255))))
    return out


# -- packing -------------------------------------------------------------------------


def _pack(
    q_vectors: np.ndarray, xyz: np.ndarray, tag_pos: np.ndarray, tag_score: np.ndarray
) -> tuple[bytes, list[dict]]:
    """Concatenate the numeric arrays into one blob and describe their layout for the decoder.

    Everything fixed-size and per-film lives here rather than in the JSON: int8 taste vectors,
    float32 xyz, and the top-tag positions/scores. Packed little-endian bytes gzip far better than
    the same integers spelled out in JSON, which is what keeps the bundle inside its budget. The
    order is fixed and coverage (a later PR) appends after ``tag_score`` so nothing above it moves.
    """
    n, k = q_vectors.shape
    parts = {
        "vectors": (np.ascontiguousarray(q_vectors, dtype=np.int8), "int8", [n, k]),
        "xyz": (np.ascontiguousarray(xyz, dtype="<f4"), "float32", [n, 3]),
        "tag_pos": (np.ascontiguousarray(tag_pos, dtype="<u2"), "uint16", [n, TOP_TAGS_PER_FILM]),
        "tag_score": (
            np.ascontiguousarray(tag_score, dtype=np.uint8),
            "uint8",
            [n, TOP_TAGS_PER_FILM],
        ),
    }
    blob = b""
    sections: list[dict] = []
    for name, (arr, dtype, shape) in parts.items():
        raw = arr.tobytes()
        sections.append(
            {"name": name, "dtype": dtype, "shape": shape, "offset": len(blob), "length": len(raw)}
        )
        blob += raw
    return blob, sections


# -- the whole build -----------------------------------------------------------------


@dataclass(frozen=True)
class _Film:
    movie_id: int
    genome_row: int
    tmdb_id: int | None
    title: str
    year: int | None
    runtime: int | None
    poster_path: str | None
    popularity: float


def _load_films(conn: sqlite3.Connection) -> list[_Film]:
    rows = conn.execute(
        "SELECT movie_id, genome_row, tmdb_id, clean_title, title, year, runtime, "
        "poster_path, popularity FROM movies WHERE genome_row IS NOT NULL ORDER BY genome_row"
    ).fetchall()
    return [
        _Film(
            movie_id=r["movie_id"],
            genome_row=r["genome_row"],
            tmdb_id=r["tmdb_id"],
            title=r["clean_title"] or r["title"],
            year=r["year"],
            runtime=r["runtime"],
            poster_path=r["poster_path"],
            popularity=float(r["popularity"] or 0.0),
        )
        for r in rows
    ]


def _tag_names(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute("SELECT position, name FROM genome_tags")
    return {r["position"]: r["name"] for r in rows}


def build_shard(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    *,
    target: int = TARGET_FILMS,
    components: int = SVD_COMPONENTS,
    seed: int = _SAMPLE_SEED,
    full_catalog_size: int | None = None,
) -> ShardBuild:
    """Build the demo shard from a catalog connection and its genome matrix.

    ``matrix`` is the full genome memmap; ``full_catalog_size`` defaults to its row count and is
    carried into the manifest so the demo can state "the full version searches about N films"
    honestly. The returned :class:`ShardBuild` is ready to write to ``shard.json`` + ``shard.bin``.
    """
    films = _load_films(conn)
    if not films:
        raise ValueError("no genome-covered films in the catalog")
    names = _tag_names(conn)
    full_size = full_catalog_size if full_catalog_size is not None else int(matrix.shape[0])

    years = np.array([f.year or 0 for f in films])
    popularity = np.array([f.popularity for f in films])
    sample = stratified_sample(years, popularity, target, seed=seed)
    chosen = [films[i] for i in sample]
    sub = np.asarray(matrix[[f.genome_row for f in chosen]], dtype=np.float32)

    projected, _ = svd_project(sub, components)
    q_vectors, scales = quantise_int8(projected)
    xyz = pca_3d(sub)

    n = len(chosen)
    tag_pos = np.full((n, TOP_TAGS_PER_FILM), TAG_SENTINEL, dtype="<u2")
    tag_score = np.zeros((n, TOP_TAGS_PER_FILM), dtype=np.uint8)
    used_positions: set[int] = set()
    film_entries: list[dict] = []
    for row, film in enumerate(chosen):
        for j, (pos, sc) in enumerate(top_tags(sub[row])):
            tag_pos[row, j] = pos
            tag_score[row, j] = sc
            used_positions.add(pos)
        film_entries.append(
            {
                "id": film.movie_id,
                "tmdb": film.tmdb_id,
                "title": film.title,
                "year": film.year,
                "runtime": film.runtime,
                "poster": film.poster_path,
            }
        )
    binary, sections = _pack(q_vectors, xyz, tag_pos, tag_score)

    manifest = {
        "version": SHARD_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "n_films": len(chosen),
        "n_components": int(q_vectors.shape[1]),
        "full_catalog_size": full_size,
        "scales": [float(s) for s in scales],
        "binary": {"file": "shard.bin", "sections": sections},
        "tag_names": {str(pos): names.get(pos, f"tag#{pos}") for pos in sorted(used_positions)},
        "films": film_entries,
    }
    return ShardBuild(manifest=manifest, binary=binary)
