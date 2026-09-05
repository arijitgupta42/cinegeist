"""The offline eval harness: it runs, it is deterministic, and it reports a real precision number.

These tests are the CI face of ``cinegeist eval`` (plan.md session 8). They don't pin the exact
precision — that is meant to move when the scorer changes — but they guard the floor (so a real
regression turns CI red), the determinism (so the number is trustworthy), and the discrimination
(so a flat-scoring bug that gives every persona the same picks can't pass silently).
"""

from __future__ import annotations

import pytest

from cinegeist.eval import run_eval, run_persona, run_returning_eval
from cinegeist.eval.catalog import CLUSTERS, build_synthetic_catalog
from cinegeist.eval.personas import PERSONAS

# Random precision@3 over six clusters is ~1/6 ≈ 0.17. The recommender must clear this comfortably;
# the floor sits well below the ~0.86 the fixture currently reports, so ordinary refactors don't
# trip it but a genuine regression (a broken cosine, a dropped profile) does.
_PRECISION_FLOOR = 0.6


def test_eval_reports_precision_above_the_floor() -> None:
    report = run_eval(seed=0)
    assert report.mean_precision_at_3 >= _PRECISION_FLOOR
    assert 0.0 <= report.mean_precision_at_3 <= 1.0


def test_eval_covers_every_persona_and_film() -> None:
    report = run_eval(seed=0)
    assert len(report.results) == len(PERSONAS)
    assert report.catalog_size == len(CLUSTERS) * 60  # FILMS_PER_CLUSTER
    for result in report.results:
        assert result.n_picks == 3  # three confident picks, the precision@3 denominator
        assert 0.0 <= result.precision_at_3 <= 1.0
        assert len(result.pick_titles) == 3


def test_eval_is_deterministic() -> None:
    first = run_eval(seed=0)
    second = run_eval(seed=0)
    assert first.mean_precision_at_3 == second.mean_precision_at_3
    assert [r.precision_at_3 for r in first.results] == [r.precision_at_3 for r in second.results]


def test_a_different_seed_is_a_different_catalog() -> None:
    # Determinism is per-seed, not global: another seed generates different films (and generally a
    # different precision), which is what lets the harness be run over several seeds to average.
    titles0 = {t for r in run_eval(seed=0).results for t in r.pick_titles}
    titles1 = {t for r in run_eval(seed=1).results for t in r.pick_titles}
    assert titles0 != titles1


def test_personas_get_distinct_recommendations() -> None:
    # The failure this guards against: a flat profile makes every persona get the same diverse
    # picks. Different tastes must produce different picks, or precision is meaningless.
    report = run_eval(seed=0)
    pick_sets = {frozenset(r.pick_titles) for r in report.results}
    assert len(pick_sets) > 1


def test_a_clear_persona_lands_in_its_cluster() -> None:
    # The arthouse purist (loves slow european drama, no confusable neighbour in its seeds) is an
    # easy case that must score perfectly — a canary that the seed → converse → present path works.
    catalog = build_synthetic_catalog(seed=0)
    purist = next(p for p in PERSONAS if p.name == "arthouse purist")
    result = run_persona(purist, catalog)
    assert result.precision_at_3 == pytest.approx(1.0)
    assert all(cluster == "slow european drama" for cluster in result.pick_clusters)


# -- the returning viewer (persistence + decay across sessions, plan.md §16) -----------------


def test_returning_profile_sharpens_and_old_evidence_fades() -> None:
    # The core invariant the multi-session eval exists to guard: a viewer coming back over weeks
    # ends up with a profile closer to their true taste, while the visit-1 misfire (an off-taste
    # like) loses its grip as fresher evidence accrues and it decays. Both are strict, not just
    # non-worsening, so a broken decay (e.g. no ageing) that keeps the misfire dominant fails here.
    result = run_returning_eval(seed=0)
    assert result.sharpened
    assert result.old_evidence_faded
    first, last = result.visits[0], result.visits[-1]
    assert last.taste_cosine > first.taste_cosine
    assert last.misfire_share < first.misfire_share


def test_returning_profile_persists_and_grows_across_visits() -> None:
    # History persists under one user id: each visit is its own session and adds evidence, so the
    # session and event counts climb monotonically and the profile is never empty.
    result = run_returning_eval(seed=0)
    assert len(result.visits) == result.n_visits
    sessions = [v.session_count for v in result.visits]
    events = [v.event_count for v in result.visits]
    assert sessions == sorted(sessions) and sessions[-1] == result.n_visits
    assert events == sorted(events) and events[0] < events[-1]
    assert all(v.total_weight > 0 for v in result.visits)


def test_returning_viewer_is_greeted_as_returning() -> None:
    # The shipping engine must recognise the persisted profile on the next visit, not cold-start.
    assert run_returning_eval(seed=0).welcomed_back


def test_returning_eval_is_deterministic() -> None:
    first = run_returning_eval(seed=0)
    second = run_returning_eval(seed=0)
    assert [v.taste_cosine for v in first.visits] == [v.taste_cosine for v in second.visits]
    assert [v.misfire_share for v in first.visits] == [v.misfire_share for v in second.visits]
