"""A synthetic catalog with known ground truth, for the offline eval harness.

The real precision of the recommender can only be judged against films whose "right answer" we
already know. We don't have the full MovieLens catalog committed (it isn't redistributed, plan.md
§2.5), so instead we *generate* a catalog whose structure we control exactly.

Films live in a small **latent taste space** — a handful of named spectra (pace, tone, spectacle,
warmth, grit, strangeness) — and each **cluster** ("slow european drama", "loud superhero action",
…) is a point in it. A cluster's films are its point plus Gaussian noise. Genome tags are *not*
independent: each tag is a direction in the latent space (``slow`` loads on the slow end of pace,
``gritty`` on grit), so a film's tag vector is a noisy projection of its latent taste. That makes
neighbouring clusters genuinely overlap in tag space — a slow drama and a cerebral sci-fi share
their contemplative, intimate tags — so the recommender has to *discriminate*, not just pattern
match. That overlap is the whole point: it is what makes precision@3 land below a saturated 1.0 and
**move when the scoring weights change**, which is what the eval is for (plan.md session 8).

Everything is seeded, so the catalog is deterministic and builds in a few milliseconds in CI. This
is the genome side only — an in-memory ``cinegeist.db`` plus the float32 matrix, wired exactly like
a real build (``movies.genome_row`` indexes the matrix rows, ``genome_tags.position`` its columns) —
so the real ``retrieve`` → ``score`` → ``present`` pipeline runs against it unchanged. The TMDB
columns are left NULL, which the scorer treats as neutral priors, so taste cosine drives the
ranking, the thing under test.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from ..catalog import db

# The named latent spectra a film's taste is a point in. Each runs roughly -1 … +1; the name is the
# positive pole. Tags and clusters are both expressed as coordinates over these axes.
AXES: tuple[str, ...] = ("pace", "tone", "spectacle", "warmth", "grit", "strangeness")
_AX = {name: i for i, name in enumerate(AXES)}

# Each cluster is a point in latent space: {axis: value in [-1, 1]}, unspecified axes are 0. Placed
# so the six sit at genuinely different characters, but with near neighbours (slow drama ↔ cerebral
# sci-fi are both slow/intimate; crime ↔ superhero are both fast/high-spectacle) that overlap.
CLUSTERS: dict[str, dict[str, float]] = {
    "slow european drama": {"pace": -0.9, "tone": -0.5, "spectacle": -0.8, "strangeness": 0.1},
    "loud superhero action": {"pace": 0.9, "tone": 0.4, "spectacle": 0.95, "strangeness": -0.3},
    "quirky indie comedy": {"pace": 0.1, "tone": 0.5, "spectacle": -0.6, "strangeness": 0.9},
    "cerebral science fiction": {
        "pace": -0.2,
        "spectacle": 0.4,
        "warmth": -0.7,
        "strangeness": 0.6,
    },
    "feel good romance": {"pace": 0.2, "tone": 0.9, "spectacle": -0.4, "warmth": 0.9, "grit": -0.6},
    "gritty crime thriller": {"pace": 0.3, "tone": -0.7, "warmth": -0.4, "grit": 0.9},
}

# Each genome tag is a direction in the same latent space: {axis: loading}. A film's relevance for
# the tag is a squashed dot product of its latent taste with this direction, so tags are correlated
# exactly as the axes make them. Names are all content/craft descriptors — none is a reception or
# verdict tag (excluded_tags.py), so masking is a no-op here and the cosine stays honest.
TAG_LOADINGS: dict[str, dict[str, float]] = {
    "slow": {"pace": -1.0},
    "fast paced": {"pace": 1.0},
    "contemplative": {"pace": -0.8, "strangeness": 0.2},
    "melancholy": {"tone": -0.9, "warmth": -0.2},
    "atmospheric": {"tone": -0.4, "spectacle": -0.5},
    "character study": {"spectacle": -0.7, "warmth": 0.3, "pace": -0.4},
    "arthouse": {"strangeness": 0.6, "spectacle": -0.5, "tone": -0.3},
    "intimate": {"spectacle": -1.0},
    "spectacle": {"spectacle": 1.0},
    "cgi spectacle": {"spectacle": 0.9, "pace": 0.3},
    "superhero": {"spectacle": 0.8, "pace": 0.6},
    "comic book": {"spectacle": 0.7, "pace": 0.5, "strangeness": -0.2},
    "franchise": {"spectacle": 0.6, "strangeness": -0.5},
    "quirky": {"strangeness": 1.0},
    "offbeat": {"strangeness": 0.9, "tone": 0.2},
    "deadpan": {"strangeness": 0.6, "warmth": -0.1},
    "whimsical": {"strangeness": 0.7, "tone": 0.6, "warmth": 0.4},
    "indie": {"spectacle": -0.6, "strangeness": 0.5, "grit": 0.2},
    "dialogue driven": {"spectacle": -0.5, "pace": -0.2},
    "coming of age": {"warmth": 0.5, "strangeness": 0.3},
    "cerebral": {"warmth": -0.7, "strangeness": 0.4},
    "mind bending": {"strangeness": 0.8, "warmth": -0.5},
    "dystopian": {"tone": -0.7, "grit": 0.3, "spectacle": 0.3},
    "science fiction": {"spectacle": 0.4, "strangeness": 0.4},
    "twist ending": {"strangeness": 0.5, "tone": -0.2},
    "philosophical": {"warmth": -0.5, "pace": -0.5, "strangeness": 0.3},
    "romance": {"warmth": 0.9, "tone": 0.5},
    "heartwarming": {"warmth": 0.8, "tone": 0.6},
    "charming": {"warmth": 0.6, "tone": 0.5},
    "lighthearted": {"tone": 0.7, "grit": -0.4},
    "gritty": {"grit": 1.0},
    "crime": {"grit": 0.7, "tone": -0.4},
    "noir": {"grit": 0.7, "tone": -0.6, "spectacle": -0.2},
    "violent": {"grit": 0.7, "tone": -0.3},
    "tense": {"grit": 0.4, "pace": 0.3, "tone": -0.3},
    "brooding": {"tone": -0.6, "warmth": -0.3, "pace": -0.3},
}

FILMS_PER_CLUSTER = 60  # enough per cluster for near-miss distractors; still milliseconds to build
_LATENT_NOISE = 0.42  # per-axis Gaussian spread of a film around its cluster point; sets overlap
_TAG_GAIN = 0.9  # scales latent·loading before it is clipped into a relevance
_TAG_NOISE = 0.03  # small per-cell relevance noise on top


@dataclass(frozen=True)
class SyntheticCatalog:
    """An in-memory catalog with the cluster membership the eval measures against.

    ``matrix`` is the float32 genome (``films × tags``); ``cluster_of`` maps a movie id to the
    cluster it was drawn from (the ground truth); ``films_in_cluster`` is the inverse; and
    ``cluster_vector`` is each cluster's mean film vector, the direction a persona who loves that
    cluster points in. ``tag_position`` maps a tag name to its matrix column.
    """

    conn: sqlite3.Connection
    matrix: np.ndarray
    tag_position: dict[str, int]
    cluster_of: dict[int, str]
    films_in_cluster: dict[str, list[int]]
    cluster_vector: dict[str, np.ndarray]

    @property
    def cluster_names(self) -> list[str]:
        return list(CLUSTERS)


def _coords(spec: dict[str, float]) -> np.ndarray:
    """Turn an {axis: value} spec into a dense latent vector over :data:`AXES`."""
    vector = np.zeros(len(AXES), dtype=np.float64)
    for axis, value in spec.items():
        vector[_AX[axis]] = value
    return vector


def _loading_matrix(tags: list[str]) -> np.ndarray:
    """Stack the tags' latent directions into a ``tags × axes`` matrix, in ``tags`` order."""
    return np.vstack([_coords(TAG_LOADINGS[name]) for name in tags])


def build_synthetic_catalog(*, seed: int = 0) -> SyntheticCatalog:
    """Generate the clustered catalog and load it into an in-memory database.

    Deterministic in ``seed``: the same seed yields the same films, so a precision number is
    reproducible and a change to it means a change to the recommender, not the fixture.
    """
    rng = np.random.default_rng(seed)
    tags = list(TAG_LOADINGS)
    tag_position = {name: index for index, name in enumerate(tags)}
    loadings = _loading_matrix(tags)  # tags × axes
    cluster_points = {name: _coords(spec) for name, spec in CLUSTERS.items()}

    rows: list[np.ndarray] = []
    movie_rows: list[tuple] = []
    cluster_of: dict[int, str] = {}
    films_in_cluster: dict[str, list[int]] = {name: [] for name in CLUSTERS}

    movie_id = 0
    for cluster_name, point in cluster_points.items():
        for n in range(FILMS_PER_CLUSTER):
            latent = point + rng.normal(scale=_LATENT_NOISE, size=len(AXES))
            activation = loadings @ latent  # one value per tag; a tag opposed to the film goes < 0
            # Clip to [0, 1] rather than squash, so a tag only "lights up" when it aligns with the
            # film's taste and most tags stay near zero — the sparsity real genome vectors have, and
            # what lets cosine actually discriminate one cluster's films from another's.
            relevance = np.clip(_TAG_GAIN * activation, 0.0, 1.0)
            relevance = relevance + rng.normal(scale=_TAG_NOISE, size=len(tags))
            vector = np.clip(relevance, 0.0, 1.0).astype(np.float32)
            rows.append(vector)

            title = f"{cluster_name.title()} {n + 1}"
            year = 1970 + int(rng.integers(0, 55))
            runtime = int(rng.integers(85, 165))
            movie_rows.append((movie_id, f"{title} ({year})", title, year, runtime, "en", movie_id))
            cluster_of[movie_id] = cluster_name
            films_in_cluster[cluster_name].append(movie_id)
            movie_id += 1

    matrix = np.vstack(rows).astype(np.float32)

    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(pos, pos, name) for name, pos in tag_position.items()],
    )
    conn.executemany(
        "INSERT INTO movies "
        "(movie_id, title, clean_title, year, runtime, original_language, genome_row, "
        "genome_source) VALUES (?, ?, ?, ?, ?, ?, ?, 'measured')",
        movie_rows,
    )
    conn.commit()

    cluster_vector = {
        name: matrix[ids].mean(axis=0).astype(np.float32) for name, ids in films_in_cluster.items()
    }
    return SyntheticCatalog(
        conn=conn,
        matrix=matrix,
        tag_position=tag_position,
        cluster_of=cluster_of,
        films_in_cluster=films_in_cluster,
        cluster_vector=cluster_vector,
    )
