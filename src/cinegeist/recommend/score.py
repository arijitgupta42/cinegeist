"""Score, diversify, and shape the candidate pool into picks (plan.md §6, steps 2–3 and 5).

This is the deterministic heart of the recommender, and it is deliberately pure: numpy in, numpy
out, no database and no LLM. That is what lets the browser demo re-implement it in TypeScript
against the same ``spec/`` fixtures (plan.md §8.6) and stay a real recommender with the LLM
removed. Keep it that way — anything needing SQLite or a network belongs in the engine, not here.

The combined score, per film::

    score = ( W_COSINE   × cosine(profile, film)      # taste match, the dominant term
            + W_QUALITY  × quality_prior(film)        # Bayesian-shrunk rating
            + W_SESSION  × cosine(session, film)       # tonight's mood, thrown away after
            + W_FACET    × facet_match(film)           # director/era/country — CLI-only, 0 here
            − W_POP      × popularity_penalty(film) )   # a nudge toward discovery
            × genome_confidence(film)                  # discount predicted vectors

The facet term is passed in (the browser shard has no credits, so the demo leaves it 0); every
other term is computed here from the film's own numbers, each degrading to a neutral prior when
its TMDB column is missing. Then MMR (λ) spreads the top of the list so three near-identical
films don't take all three slots, and the wildcard reaches deliberately outside the centroid.

The weights and thresholds are provisional module constants; in session 5 they move to ``spec/``
and are asserted by both the Python and TypeScript test suites, so the two scorers can't drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..catalog import genome

# -- scoring weights (plan.md §6; provisional, they move to spec/ in session 5) -------
W_COSINE = 0.55
W_FACET = 0.20
W_QUALITY = 0.10
W_SESSION = 0.10
W_POPULARITY = 0.05

# Bayesian shrinkage for the quality prior: blend a film's own rating toward a global mean, by a
# pseudo-count of prior "votes", so a 9.0 from 12 people doesn't outrank a 7.8 from thousands.
QUALITY_PRIOR_MEAN = 6.2  # roughly the catalog-wide average TMDB vote, on the 0–10 scale
QUALITY_PRIOR_COUNT = 25.0  # how many prior votes the mean is worth; small, so real votes win fast
_RATING_SCALE = 10.0  # TMDB ratings are 0–10; we normalise the prior to [0, 1]

# Popularity is long-tailed, so we penalise its log, saturating at a "very popular" reference.
POPULARITY_REFERENCE = 50.0  # a TMDB popularity around here counts as fully mainstream (penalty 1)

# Predicted genome vectors (Phase 2) are less trustworthy than measured ones and get discounted.
PREDICTED_CONFIDENCE = 0.85

# MMR trade-off (plan.md §6, step 3): 1.0 is pure relevance, 0.0 is pure diversity.
MMR_LAMBDA = 0.7

# Wildcard (plan.md §6, step 5): far from the centroid, yet sharing enough strong tags to not be
# random. "Far" is a low cosine to the profile; "shares a strong tag" is high relevance on one of
# the profile's own top axes; two such tags qualify a film.
WILDCARD_MAX_COSINE = 0.35
WILDCARD_TAG_RELEVANCE = 0.5
WILDCARD_MIN_SHARED_TAGS = 2


@dataclass(frozen=True)
class ScoredFilm:
    """A film with its combined score and the component breakdown that produced it.

    The breakdown is not bookkeeping — explanations (a later PR) read it to say *why* a film
    scored, and the wildcard reads ``cosine`` to judge distance from the centroid. ``pool_index``
    is the film's row in the candidate-vector matrix passed to scoring, so MMR and the wildcard
    can look its genome vector back up after the list has been reordered.
    """

    movie_id: int
    genome_row: int
    pool_index: int
    title: str
    year: int | None
    score: float
    cosine: float
    quality: float
    session_fit: float
    facet_match: float
    popularity_penalty: float
    confidence: float


def bayesian_quality(vote_average: float | None, vote_count: int | None) -> float:
    """A rating shrunk toward the global mean by a pseudo-count, normalised to ``[0, 1]``.

    An un-rated film (no ``vote_average``) returns the prior mean itself — a neutral 0.62, not a
    zero — so a genome-only catalog isn't punished for lacking TMDB numbers.
    """
    if vote_average is None:
        return QUALITY_PRIOR_MEAN / _RATING_SCALE
    votes = float(vote_count or 0.0)
    shrunk = (votes * vote_average + QUALITY_PRIOR_COUNT * QUALITY_PRIOR_MEAN) / (
        votes + QUALITY_PRIOR_COUNT
    )
    return min(1.0, max(0.0, shrunk / _RATING_SCALE))


def popularity_penalty(popularity: float | None) -> float:
    """The discovery nudge: ``log1p(popularity)`` scaled to ``[0, 1]`` against a reference.

    Unknown or non-positive popularity means no penalty (0.0) — we never punish a film for a
    missing number. A film at the reference popularity or above is penalised fully.
    """
    if popularity is None or popularity <= 0.0:
        return 0.0
    return min(1.0, math.log1p(popularity) / math.log1p(POPULARITY_REFERENCE))


def confidence_for(genome_source: str) -> float:
    """The multiplier that trusts a measured genome vector fully and discounts a predicted one."""
    return PREDICTED_CONFIDENCE if genome_source == "predicted" else 1.0


def score_pool(
    films: list,
    vectors: np.ndarray,
    profile_vector: np.ndarray,
    *,
    session_vector: np.ndarray | None = None,
    facet_scores: np.ndarray | None = None,
) -> list[ScoredFilm]:
    """Score every candidate, returned in input order (``pool_index`` == list index).

    ``films`` are :class:`~cinegeist.recommend.retrieve.Candidate`-shaped (any object with the
    fields read below); ``vectors`` is their genome rows stacked in the same order, so
    ``vectors[i]`` is ``films[i]``'s vector. ``session_vector`` is tonight's mood as a genome
    direction (``None`` when there's no session intent); ``facet_scores[i]`` is the film's facet
    match in ``[-1, 1]`` (``None`` → 0, as in the browser demo). No sorting happens here — MMR
    decides the order next.
    """
    n = len(films)
    if n == 0:
        return []
    cosine = genome.cosine_scores(vectors, profile_vector)
    if session_vector is not None and np.any(session_vector):
        session = genome.cosine_scores(vectors, session_vector)
    else:
        session = np.zeros(n, dtype=np.float32)

    scored: list[ScoredFilm] = []
    for i, film in enumerate(films):
        quality = bayesian_quality(film.vote_average, film.vote_count)
        pop_penalty = popularity_penalty(film.popularity)
        facet = float(facet_scores[i]) if facet_scores is not None else 0.0
        confidence = confidence_for(film.genome_source)
        combined = (
            W_COSINE * float(cosine[i])
            + W_QUALITY * quality
            + W_SESSION * float(session[i])
            + W_FACET * facet
            - W_POPULARITY * pop_penalty
        ) * confidence
        scored.append(
            ScoredFilm(
                movie_id=film.movie_id,
                genome_row=film.genome_row,
                pool_index=i,
                title=film.title,
                year=film.year,
                score=combined,
                cosine=float(cosine[i]),
                quality=quality,
                session_fit=float(session[i]),
                facet_match=facet,
                popularity_penalty=pop_penalty,
                confidence=confidence,
            )
        )
    return scored


def _aligned_vectors(scored: list[ScoredFilm], vectors: np.ndarray) -> np.ndarray:
    """The genome rows of ``scored`` stacked in *its* order (``scored[i]`` ↔ result row ``i``)."""
    return np.asarray(vectors[[f.pool_index for f in scored]], dtype=np.float32)


def mmr_rank(
    scored: list[ScoredFilm],
    vectors: np.ndarray,
    *,
    lam: float = MMR_LAMBDA,
    k: int | None = None,
) -> list[ScoredFilm]:
    """Reorder ``scored`` for relevance *and* diversity, returning the top ``k`` (plan.md §6.3).

    Maximal Marginal Relevance: each pick maximises ``λ·relevance − (1−λ)·(nearest picked)``.
    Relevance is min-max normalised to ``[0, 1]`` so it trades off cleanly against genome cosine
    similarity (also ``[0, 1]`` for these non-negative vectors) and ``λ`` means what it says.
    Without this, three films that are effectively the same could take all three pick slots.
    """
    if not scored:
        return []
    limit = len(scored) if k is None else min(k, len(scored))
    svecs = _aligned_vectors(scored, vectors)

    raw = np.array([f.score for f in scored], dtype=np.float64)
    span = float(raw.max() - raw.min())
    relevance = (raw - raw.min()) / span if span > 0 else np.ones(len(scored))

    max_sim = np.zeros(len(scored), dtype=np.float64)  # closeness of each film to the picked set
    chosen_mask = np.zeros(len(scored), dtype=bool)
    order: list[int] = []
    for _ in range(limit):
        value = lam * relevance - (1.0 - lam) * max_sim
        value[chosen_mask] = -np.inf
        pick = int(np.argmax(value))
        order.append(pick)
        chosen_mask[pick] = True
        sims = genome.cosine_scores(svecs, svecs[pick])
        max_sim = np.maximum(max_sim, sims)
    return [scored[i] for i in order]


def select_wildcard(
    scored: list[ScoredFilm],
    vectors: np.ndarray,
    strong_tag_positions: frozenset[int],
    *,
    exclude_movie_ids: frozenset[int] = frozenset(),
) -> ScoredFilm | None:
    """The exploration slot: the best film that is far from taste yet shares real tags (§6.5).

    "Far" is a cosine to the profile at or below :data:`WILDCARD_MAX_COSINE`; "shares real tags"
    is high relevance on at least :data:`WILDCARD_MIN_SHARED_TAGS` of the profile's own strong
    axes, so the pick is adventurous, not random. Among the films that qualify we take the
    highest-scoring. Returns ``None`` when nothing is both far enough and grounded enough — an
    honest empty slot beats a forced one (and the honesty path suppresses it entirely, §8.4).
    """
    if not scored or not strong_tag_positions:
        return None
    positions = np.array(sorted(strong_tag_positions), dtype=int)
    svecs = _aligned_vectors(scored, vectors)

    best: ScoredFilm | None = None
    for row, film in enumerate(scored):
        if film.movie_id in exclude_movie_ids:
            continue
        if film.cosine > WILDCARD_MAX_COSINE:
            continue
        shared = int(np.count_nonzero(svecs[row, positions] >= WILDCARD_TAG_RELEVANCE))
        if shared < WILDCARD_MIN_SHARED_TAGS:
            continue
        if best is None or film.score > best.score:
            best = film
    return best


@dataclass(frozen=True)
class Recommendations:
    """A finished deterministic recommendation: the picks, the wildcard, and the full shortlist.

    ``shortlist`` is the MMR-diversified top of the pool — what the LLM rerank stage (next PR)
    reorders. ``picks`` is the shortlist's head (up to ``n_confident``); ``wildcard`` is chosen
    from outside them. The engine that adds the LLM re-derives ``picks`` from the reranked
    shortlist; this deterministic form is what ``--offline`` shows and what the tests pin.
    """

    picks: tuple[ScoredFilm, ...]
    wildcard: ScoredFilm | None
    shortlist: tuple[ScoredFilm, ...]


def recommend(
    films: list,
    vectors: np.ndarray,
    profile_vector: np.ndarray,
    *,
    strong_tag_positions: frozenset[int] = frozenset(),
    session_vector: np.ndarray | None = None,
    facet_scores: np.ndarray | None = None,
    n_confident: int = 3,
    shortlist_size: int = 40,
    lam: float = MMR_LAMBDA,
    with_wildcard: bool = True,
) -> Recommendations:
    """Score → diversify → shape into ``n_confident`` picks plus one wildcard, no LLM involved.

    This is the whole deterministic pipeline in one call: the ``--offline`` recommender, and the
    base the online engine augments by reranking ``shortlist`` before taking its head as the
    picks. ``vectors[i]`` must be ``films[i]``'s genome row.
    """
    scored = score_pool(
        films,
        vectors,
        profile_vector,
        session_vector=session_vector,
        facet_scores=facet_scores,
    )
    if not scored:
        return Recommendations(picks=(), wildcard=None, shortlist=())
    shortlist = mmr_rank(scored, vectors, lam=lam, k=shortlist_size)
    picks = shortlist[:n_confident]
    wildcard = None
    if with_wildcard:
        picked_ids = frozenset(f.movie_id for f in picks)
        wildcard = select_wildcard(
            scored, vectors, strong_tag_positions, exclude_movie_ids=picked_ids
        )
    return Recommendations(
        picks=tuple(picks),
        wildcard=wildcard,
        shortlist=tuple(shortlist),
    )


def candidate_vectors(films: list, matrix: np.ndarray) -> np.ndarray:
    """Stack the genome rows of ``films`` (each carrying a ``genome_row``) into one array.

    A small convenience so callers don't hand-roll the ``matrix[[...]]`` gather that scoring,
    MMR, and the wildcard all need aligned to the candidate list.
    """
    if not films:
        return np.zeros((0, int(matrix.shape[1])), dtype=np.float32)
    return np.asarray(matrix[[f.genome_row for f in films]], dtype=np.float32)


def with_score(film: ScoredFilm, score: float) -> ScoredFilm:
    """A copy of ``film`` with a new combined ``score`` (used when a later stage re-weights)."""
    return replace(film, score=score)
