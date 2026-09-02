"""Synthetic personas with a known taste, for the eval harness.

Each persona loves one cluster of the synthetic catalog and (usually) bounces off another — the
sharp negative signal the plan prizes ("loves slow European drama, hates capes", plan.md §5.1,
session 8). The persona is the ground truth: after a simulated conversation, a recommendation is
"right" when it lands in the loved cluster, and precision@3 counts how many of the three confident
picks do. The hated cluster shapes both the seed dislike and the taste direction the persona uses
to answer pair questions, so aversion is measured, not just affinity.

The taste ``direction`` a persona reasons with is its loved cluster's centroid minus a fraction of
its hated cluster's — a plausible profile centroid, and what the oracle (oracle.py) uses to decide
which film in a pair it would rather watch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import SyntheticCatalog

# How much the hated cluster pulls the persona's taste direction away from centre. Kept small so
# the loved cluster dominates — an aversion narrows taste, it must not rotate it into a third
# cluster (disliking romance shouldn't read as loving crime).
_HATED_PULL = 0.35


@dataclass(frozen=True)
class Persona:
    """One synthetic viewer: what they love, what they bounce off, and how they seed the chat."""

    name: str
    description: str
    loved: str  # a cluster name in catalog.CLUSTERS
    hated: str | None = None  # a cluster name, or None for a purely positive persona
    n_seed_likes: int = 2  # loved-cluster films offered as opening "films I loved"
    n_seed_dislikes: int = 1  # hated-cluster films offered as "bounced off" (0 if no hated cluster)


PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="arthouse purist",
        description="slow, contemplative European drama; no interest in spectacle",
        loved="slow european drama",
        hated="loud superhero action",
    ),
    Persona(
        name="saturday blockbuster",
        description="loud superhero spectacle; finds slow cinema a chore",
        loved="loud superhero action",
        hated="slow european drama",
    ),
    Persona(
        name="indie darling",
        description="quirky offbeat indie comedy; allergic to franchise tentpoles",
        loved="quirky indie comedy",
        hated="loud superhero action",
    ),
    Persona(
        name="big ideas",
        description="cerebral, mind-bending science fiction; cool on sentiment",
        loved="cerebral science fiction",
        hated="feel good romance",
    ),
    Persona(
        name="hopeless romantic",
        description="warm feel-good romance; put off by grimness and violence",
        loved="feel good romance",
        hated="gritty crime thriller",
    ),
    Persona(
        name="neo noir",
        description="gritty crime and noir; finds rom-coms saccharine",
        loved="gritty crime thriller",
        hated="feel good romance",
    ),
    Persona(
        name="pure positive",
        description="loves quirky indie comedy, with no stated dislike (affinity only)",
        loved="quirky indie comedy",
        hated=None,
        n_seed_dislikes=0,
    ),
)


def taste_direction(persona: Persona, catalog: SyntheticCatalog) -> np.ndarray:
    """The vector the persona answers pair questions with: loved cluster, pulled off by hated."""
    direction = catalog.cluster_vector[persona.loved].astype(np.float32).copy()
    if persona.hated is not None:
        direction = direction - _HATED_PULL * catalog.cluster_vector[persona.hated]
    return direction.astype(np.float32)


def relevant_ids(persona: Persona, catalog: SyntheticCatalog) -> frozenset[int]:
    """The films that count as a good recommendation: everything in the loved cluster."""
    return frozenset(catalog.films_in_cluster[persona.loved])


def seed_film_ids(persona: Persona, catalog: SyntheticCatalog) -> tuple[list[int], list[int]]:
    """Deterministic opening evidence: (loved films the persona names, hated films they name).

    Drawn from the ends of each cluster's id range so the seed films and the held-out films used to
    score precision don't overlap — the recommender is judged on films it was *not* handed.
    """
    loved = catalog.films_in_cluster[persona.loved][: persona.n_seed_likes]
    hated: list[int] = []
    if persona.hated is not None and persona.n_seed_dislikes:
        hated = catalog.films_in_cluster[persona.hated][: persona.n_seed_dislikes]
    return loved, hated
