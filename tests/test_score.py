"""Unit tests for the deterministic scorer: the weighted combination, MMR, and the wildcard.

Scoring is pure numpy, so these build genome vectors and candidates by hand and assert the exact
arithmetic — this is where the real recommender bugs live (plan.md §12), so the maths is pinned
rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cinegeist.recommend import score
from cinegeist.recommend.retrieve import Candidate


def _candidate(
    movie_id: int,
    genome_row: int,
    *,
    vote_average: float | None = None,
    vote_count: int | None = None,
    popularity: float | None = None,
    genome_source: str = "measured",
) -> Candidate:
    return Candidate(
        movie_id=movie_id,
        genome_row=genome_row,
        title=f"Film {movie_id}",
        year=2000 + movie_id,
        runtime=100,
        language="en",
        vote_average=vote_average,
        vote_count=vote_count,
        popularity=popularity,
        genome_source=genome_source,
    )


# -- the component priors ------------------------------------------------------------


def test_bayesian_quality_uses_the_prior_when_unrated() -> None:
    assert score.bayesian_quality(None, None) == pytest.approx(score.QUALITY_PRIOR_MEAN / 10.0)


def test_bayesian_quality_shrinks_few_votes_toward_the_mean() -> None:
    # A 9.0 from 5 people barely moves off the 6.2 prior; a 9.0 from thousands nearly reaches it.
    shy = score.bayesian_quality(9.0, 5)
    confident = score.bayesian_quality(9.0, 5000)
    assert score.QUALITY_PRIOR_MEAN / 10.0 < shy < confident
    assert confident == pytest.approx(0.8986, abs=1e-3)


def test_popularity_penalty_is_zero_when_unknown_and_saturates() -> None:
    assert score.popularity_penalty(None) == 0.0
    assert score.popularity_penalty(0.0) == 0.0
    assert score.popularity_penalty(score.POPULARITY_REFERENCE) == pytest.approx(1.0)
    assert score.popularity_penalty(10_000.0) == 1.0  # capped, never above 1
    assert 0.0 < score.popularity_penalty(5.0) < 1.0


def test_confidence_discounts_predicted_vectors() -> None:
    assert score.confidence_for("measured") == 1.0
    assert score.confidence_for("none") == 1.0
    assert score.confidence_for("predicted") == score.PREDICTED_CONFIDENCE


# -- the combined score --------------------------------------------------------------


def test_score_pool_computes_the_weighted_combination_exactly() -> None:
    # Two orthogonal tags; the profile points entirely at tag 0.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    profile = np.array([1.0, 0.0], dtype=np.float32)
    films = [_candidate(1, 0), _candidate(2, 1)]  # both unrated → quality is the prior

    scored = score.score_pool(films, vectors, profile)

    prior_quality = score.QUALITY_PRIOR_MEAN / 10.0
    a, b = scored
    assert a.cosine == pytest.approx(1.0)
    assert a.score == pytest.approx(score.W_COSINE * 1.0 + score.W_QUALITY * prior_quality)
    assert b.cosine == pytest.approx(0.0)
    assert b.score == pytest.approx(score.W_QUALITY * prior_quality)
    assert a.pool_index == 0 and b.pool_index == 1


def test_quality_and_popularity_move_the_score() -> None:
    vectors = np.array([[0.0, 1.0]], dtype=np.float32)  # orthogonal to the profile → cosine 0
    profile = np.array([1.0, 0.0], dtype=np.float32)
    film = _candidate(
        1, 0, vote_average=8.0, vote_count=10_000, popularity=score.POPULARITY_REFERENCE
    )

    (scored,) = score.score_pool([film], vectors, profile)
    assert scored.cosine == pytest.approx(0.0)
    expected = score.W_QUALITY * score.bayesian_quality(8.0, 10_000) - score.W_POPULARITY * 1.0
    assert scored.score == pytest.approx(expected)


def test_session_vector_adds_a_session_fit_term() -> None:
    vectors = np.array([[0.0, 1.0]], dtype=np.float32)
    profile = np.array([1.0, 0.0], dtype=np.float32)  # no taste overlap
    session = np.array([0.0, 1.0], dtype=np.float32)  # but tonight's mood points at the film

    without = score.score_pool([_candidate(1, 0)], vectors, profile)[0]
    with_session = score.score_pool([_candidate(1, 0)], vectors, profile, session_vector=session)[0]
    assert with_session.session_fit == pytest.approx(1.0)
    assert with_session.score - without.score == pytest.approx(score.W_SESSION * 1.0)


def test_predicted_confidence_scales_the_whole_score() -> None:
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    profile = np.array([1.0, 0.0], dtype=np.float32)
    measured = score.score_pool([_candidate(1, 0)], vectors, profile)[0]
    predicted = score.score_pool([_candidate(1, 0, genome_source="predicted")], vectors, profile)[0]
    assert predicted.score == pytest.approx(measured.score * score.PREDICTED_CONFIDENCE)


# -- MMR diversity -------------------------------------------------------------------


def _mmr_setup() -> tuple[list[Candidate], np.ndarray]:
    # A and B are identical (a duplicate); C is orthogonal but genuinely relevant to nothing here.
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    films = [_candidate(1, 0), _candidate(2, 1), _candidate(3, 2)]
    return films, vectors


def test_mmr_first_pick_is_the_most_relevant() -> None:
    films, vectors = _mmr_setup()
    profile = np.array([1.0, 0.0], dtype=np.float32)
    scored = score.score_pool(films, vectors, profile)
    order = score.mmr_rank(scored, vectors, lam=score.MMR_LAMBDA)
    assert order[0].movie_id == 1  # highest cosine


def test_pure_relevance_keeps_the_duplicate_but_diversity_drops_it() -> None:
    films, vectors = _mmr_setup()
    profile = np.array([1.0, 0.0], dtype=np.float32)
    scored = score.score_pool(films, vectors, profile)

    # λ=1.0 is pure relevance: the identical film 2 is as good as film 1, so it ranks second.
    relevance_only = score.mmr_rank(scored, vectors, lam=1.0, k=2)
    assert [f.movie_id for f in relevance_only] == [1, 2]

    # λ=0.3 leans on diversity: the redundant film 2 is pushed below the distinct film 3.
    diverse = score.mmr_rank(scored, vectors, lam=0.3, k=2)
    assert [f.movie_id for f in diverse] == [1, 3]


def test_mmr_respects_k_and_handles_empty() -> None:
    films, vectors = _mmr_setup()
    scored = score.score_pool(films, vectors, np.array([1.0, 0.0], dtype=np.float32))
    assert len(score.mmr_rank(scored, vectors, k=2)) == 2
    assert len(score.mmr_rank(scored, vectors, k=10)) == 3  # k above the pool size is clamped
    assert score.mmr_rank([], vectors) == []


# -- the wildcard --------------------------------------------------------------------


def _wildcard_pool() -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    # 8-dim space; the profile likes tags 0 and 1. The wildcard E loads on 0 and 1 (so it shares
    # two strong tags) but also on six others, which pulls its cosine to the profile below 0.35.
    profile = np.zeros(8, dtype=np.float32)
    profile[0] = profile[1] = 1.0
    vectors = np.array(
        [
            [1.0, 1.0, 0, 0, 0, 0, 0, 0],  # F (near): cosine 1.0 — too close to be a wildcard
            [0.6, 0.0, 1, 1, 1, 1, 1, 1],  # G (far, shares only tag 0) — not grounded enough
            [0.6, 0.6, 1, 1, 1, 1, 1, 1],  # E (far, shares tags 0 and 1) — the wildcard
        ],
        dtype=np.float32,
    )
    films = [_candidate(10, 0), _candidate(20, 1), _candidate(30, 2)]
    return films, vectors, profile


def test_wildcard_is_far_but_shares_strong_tags() -> None:
    films, vectors, profile = _wildcard_pool()
    scored = score.score_pool(films, vectors, profile)
    wildcard = score.select_wildcard(scored, vectors, frozenset({0, 1}))
    assert wildcard is not None
    assert wildcard.movie_id == 30  # E: far enough, and grounded in two shared tags
    assert wildcard.cosine <= score.WILDCARD_MAX_COSINE


def test_wildcard_excludes_already_picked_and_needs_strong_tags() -> None:
    films, vectors, profile = _wildcard_pool()
    scored = score.score_pool(films, vectors, profile)
    # With no known strong axes there is nothing to ground a wildcard on.
    assert score.select_wildcard(scored, vectors, frozenset()) is None
    # Excluding the only qualifier leaves no wildcard.
    assert (
        score.select_wildcard(scored, vectors, frozenset({0, 1}), exclude_movie_ids=frozenset({30}))
        is None
    )


# -- the whole pipeline --------------------------------------------------------------


def test_recommend_returns_picks_wildcard_and_shortlist() -> None:
    # n_confident=1 so the one far-but-grounded film (E) stays out of the picks and is free to be
    # the wildcard — with a real pool the picks and the wildcard rarely collide like a tiny one.
    films, vectors, profile = _wildcard_pool()
    result = score.recommend(
        films, vectors, profile, strong_tag_positions=frozenset({0, 1}), n_confident=1
    )
    assert len(result.shortlist) == 3
    assert len(result.picks) == 1
    assert result.picks[0].movie_id == 10  # F is the closest match, ranked first
    assert result.wildcard is not None
    assert result.wildcard.movie_id == 30
    # The wildcard is never also one of the confident picks.
    assert result.wildcard.movie_id not in {p.movie_id for p in result.picks}


def test_recommend_on_empty_pool_is_empty() -> None:
    result = score.recommend([], np.zeros((0, 4), dtype=np.float32), np.zeros(4, dtype=np.float32))
    assert result.picks == () and result.wildcard is None and result.shortlist == ()
