"""Run the deterministic reference against a ``spec/`` fixture case (plan.md §8.6).

This is the single definition of *how a fixture case is executed against the real Python*, shared
by two callers so they can never disagree:

* ``scripts/build_spec_fixtures.py`` authors the input half of each case, calls the matching
  ``run_*`` here to compute the expected half, and writes both to ``spec/``.
* ``tests/test_spec.py`` loads the committed cases and calls the same ``run_*`` on the input half,
  asserting the fresh output still matches the committed expected within :data:`TOLERANCE`.

Because both paths funnel through here, a change to the scoring, decay, probe, or coverage code
changes the fresh output, the committed fixtures stop matching, and CI goes red until someone
regenerates them — which is the whole point: the fixtures are what stop the Python and TypeScript
scorers drifting apart, and regenerating forces a review that must carry over to the port.

Everything here is deliberately plain: a case is a JSON-shaped ``dict`` in, a JSON-shaped ``dict``
out, so the exact same case files drive ``vitest`` in session 6 with no translation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from cinegeist.catalog import db
from cinegeist.convo import probes
from cinegeist.profile import store, update
from cinegeist.profile.model import PreferenceEvent
from cinegeist.recommend import coverage, score
from cinegeist.recommend.retrieve import Candidate

# Floats travel through JSON and are recomputed in float64 here but float32 inside the genome
# maths (and, in session 6, float64 in JavaScript). This tolerance is what "identical within float
# tolerance" means for the whole spec; keep it loose enough for a float32↔float64 gap and tight
# enough to catch a real arithmetic change.
TOLERANCE = 1e-5

# A fixed reference instant for decay cases. Event ages are expressed as days before this, so the
# absolute clock never enters a fixture and the result is reproducible forever.
DECAY_NOW = datetime(2025, 1, 1, tzinfo=UTC)


# -- the canonical constant registry -------------------------------------------------


def constants() -> dict[str, Any]:
    """Every shared tunable, read straight from the module that owns it (§8.6).

    ``spec/constants.json`` is generated from this, and ``test_spec`` asserts the two still agree,
    so a constant edited in Python but not regenerated (or vice versa) turns CI red — the same
    drift guard the behavioural fixtures give, for the bare numbers.
    """
    return {
        "scoring": {
            "W_COSINE": score.W_COSINE,
            "W_FACET": score.W_FACET,
            "W_QUALITY": score.W_QUALITY,
            "W_SESSION": score.W_SESSION,
            "W_POPULARITY": score.W_POPULARITY,
            "QUALITY_PRIOR_MEAN": score.QUALITY_PRIOR_MEAN,
            "QUALITY_PRIOR_COUNT": score.QUALITY_PRIOR_COUNT,
            "RATING_SCALE": score._RATING_SCALE,
            "POPULARITY_REFERENCE": score.POPULARITY_REFERENCE,
            "PREDICTED_CONFIDENCE": score.PREDICTED_CONFIDENCE,
            "MMR_LAMBDA": score.MMR_LAMBDA,
            "WILDCARD_MAX_COSINE": score.WILDCARD_MAX_COSINE,
            "WILDCARD_TAG_RELEVANCE": score.WILDCARD_TAG_RELEVANCE,
            "WILDCARD_MIN_SHARED_TAGS": score.WILDCARD_MIN_SHARED_TAGS,
        },
        "decay": {
            "HALF_LIFE_DAYS": update.HALF_LIFE_DAYS,
            "AXES_PER_SIGN": update._AXES_PER_SIGN,
            "AXIS_EPSILON": update._AXIS_EPSILON,
        },
        "probes": {
            "POOL_TOP": probes._POOL_TOP,
            "MIN_SPREAD": probes._MIN_SPREAD,
            "MAX_TURNS": probes.MAX_TURNS,
            "MIN_TURNS": probes.MIN_TURNS,
            "STABLE_TURNS": probes.STABLE_TURNS,
            "MARGIN_THRESHOLD": probes.MARGIN_THRESHOLD,
        },
        "coverage": {
            "COVERAGE_SIMILARITY": coverage.COVERAGE_SIMILARITY,
            "COVERAGE_TOP_K": coverage.COVERAGE_TOP_K,
            "REGION_COVERAGE_MIN": coverage.REGION_COVERAGE_MIN,
            "NEAREST_COSINE_MIN": coverage.NEAREST_COSINE_MIN,
        },
    }


# -- comparison ----------------------------------------------------------------------


def assert_close(fresh: Any, committed: Any, *, tol: float = TOLERANCE, path: str = "") -> None:
    """Assert two JSON-shaped values match: numbers within ``tol``, everything else exactly.

    Raises ``AssertionError`` with the path to the first mismatch, so a regenerated fixture that
    disagrees with the code points straight at the field that moved.
    """
    if isinstance(committed, bool) or isinstance(fresh, bool):
        assert fresh == committed, f"{path or '<root>'}: {fresh!r} != {committed!r}"
    elif isinstance(committed, (int, float)) and isinstance(fresh, (int, float)):
        assert abs(float(fresh) - float(committed)) <= tol, (
            f"{path or '<root>'}: {fresh!r} != {committed!r} (tol {tol})"
        )
    elif isinstance(committed, dict):
        assert isinstance(fresh, dict), f"{path or '<root>'}: expected dict, got {type(fresh)}"
        assert set(fresh) == set(committed), (
            f"{path or '<root>'}: keys {sorted(fresh)} != {sorted(committed)}"
        )
        for key in committed:
            assert_close(fresh[key], committed[key], tol=tol, path=f"{path}.{key}" if path else key)
    elif isinstance(committed, list):
        assert isinstance(fresh, list), f"{path or '<root>'}: expected list, got {type(fresh)}"
        assert len(fresh) == len(committed), (
            f"{path or '<root>'}: length {len(fresh)} != {len(committed)}"
        )
        for i, (f, c) in enumerate(zip(fresh, committed, strict=True)):
            assert_close(f, c, tol=tol, path=f"{path}[{i}]")
    else:
        assert fresh == committed, f"{path or '<root>'}: {fresh!r} != {committed!r}"


# -- scoring (pure numpy: score.py) --------------------------------------------------


def _vecs(films: list[dict], n_tags: int) -> np.ndarray:
    return np.array([f["vector"] for f in films], dtype=np.float32).reshape(len(films), n_tags)


def _candidate(f: dict, genome_row: int) -> Candidate:
    return Candidate(
        movie_id=f["movie_id"],
        genome_row=genome_row,
        title=f.get("title", f"Film {f['movie_id']}"),
        year=f.get("year"),
        runtime=f.get("runtime"),
        language=f.get("language"),
        vote_average=f.get("vote_average"),
        vote_count=f.get("vote_count"),
        popularity=f.get("popularity"),
        genome_source=f.get("genome_source", "measured"),
    )


def run_scoring_case(case: dict) -> dict:
    """Score a pool and shape it into picks + wildcard, returning the component breakdown."""
    n_tags = case["n_tags"]
    films = case["films"]
    vectors = _vecs(films, n_tags)
    candidates = [_candidate(f, i) for i, f in enumerate(films)]
    profile = np.array(case["profile"], dtype=np.float32)
    session = (
        np.array(case["session"], dtype=np.float32) if case.get("session") is not None else None
    )
    facets = (
        np.array(case["facet_scores"], dtype=np.float32)
        if case.get("facet_scores") is not None
        else None
    )

    scored = score.score_pool(
        candidates, vectors, profile, session_vector=session, facet_scores=facets
    )
    result = score.recommend(
        candidates,
        vectors,
        profile,
        strong_tag_positions=frozenset(case.get("strong_tag_positions", [])),
        session_vector=session,
        facet_scores=facets,
        n_confident=case.get("n_confident", 3),
        shortlist_size=case.get("shortlist_size", 40),
        lam=case.get("lam", score.MMR_LAMBDA),
        with_wildcard=case.get("with_wildcard", True),
    )
    return {
        "scored": [
            {
                "movie_id": s.movie_id,
                "score": s.score,
                "cosine": s.cosine,
                "quality": s.quality,
                "session_fit": s.session_fit,
                "facet_match": s.facet_match,
                "popularity_penalty": s.popularity_penalty,
                "confidence": s.confidence,
            }
            for s in scored
        ],
        "shortlist_ids": [s.movie_id for s in result.shortlist],
        "picks_ids": [s.movie_id for s in result.picks],
        "wildcard_id": result.wildcard.movie_id if result.wildcard else None,
    }


# -- decay (update.py, via a throwaway in-memory catalog) ----------------------------


def _memory_catalog(case: dict) -> tuple[Any, np.ndarray]:
    """Build a ``:memory:`` catalog and genome matrix from a fixture's movies and tags."""
    conn = db.connect(":memory:")
    db.migrate(conn)
    tags = case.get("tags", [])
    if tags:
        conn.executemany(
            "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
            [(t["tag_id"], t["position"], t["name"]) for t in tags],
        )
    movies = case.get("movies", [])
    if movies:
        conn.executemany(
            "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
            "VALUES (?, ?, ?, ?, ?, 'measured')",
            [
                (
                    m["movie_id"],
                    m.get("title", f"M{m['movie_id']}"),
                    m.get("clean_title"),
                    m.get("year"),
                    m["genome_row"],
                )
                for m in movies
            ],
        )
    conn.commit()
    n_tags = case["n_tags"]
    n_rows = (max((m["genome_row"] for m in movies), default=-1) + 1) if movies else 0
    matrix = np.zeros((n_rows, n_tags), dtype=np.float32)
    for m in movies:
        matrix[m["genome_row"]] = np.array(m["vector"], dtype=np.float32)
    return conn, matrix


def run_decay_case(case: dict) -> dict:
    """Fold an event log into the decayed centroid, total weight, and ranked axes."""
    conn, matrix = _memory_catalog(case)
    half_life = case.get("half_life_days", update.HALF_LIFE_DAYS)
    events = [
        PreferenceEvent(
            kind=e["kind"],
            subject_kind=e["subject_kind"],
            subject=str(e["subject"]),
            value=e["value"],
            weight=e.get("weight", 1.0),
            evidence=e.get("evidence"),
            session_id=e.get("session_id"),
            ts=DECAY_NOW - timedelta(days=e["age_days"]),
        )
        for e in case["events"]
    ]
    store.append_events(conn, events, now=DECAY_NOW)
    profile = update.compute_profile(conn, matrix, now=DECAY_NOW, half_life=half_life)
    conn.close()
    return {
        "centroid": [float(x) for x in profile.genome_vector],
        "total_weight": float(profile.total_weight),
        "axes": [
            {
                "position": a.position,
                "name": a.name,
                "weight": float(a.weight),
                "source": a.source,
                "evidence": a.evidence,
            }
            for a in profile.axes
        ],
    }


# -- probes (probes.py, via a throwaway in-memory catalog) ---------------------------


def _probe_catalog(case: dict) -> tuple[Any, np.ndarray]:
    conn = db.connect(":memory:")
    db.migrate(conn)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(t["tag_id"], t["position"], t["name"]) for t in case["tags"]],
    )
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, 'measured')",
        [
            (
                m["movie_id"],
                m.get("title", f"M{m['movie_id']}"),
                m.get("clean_title"),
                m.get("year"),
                m["genome_row"],
            )
            for m in case["movies"]
        ],
    )
    conn.commit()
    n_tags = case["n_tags"]
    n_rows = max(m["genome_row"] for m in case["movies"]) + 1
    matrix = np.zeros((n_rows, n_tags), dtype=np.float32)
    for m in case["movies"]:
        matrix[m["genome_row"]] = np.array(m["vector"], dtype=np.float32)
    return conn, matrix


def run_probe_case(case: dict) -> dict:
    """Choose the most informative probe for a profile, or report that nothing divides the pool."""
    conn, matrix = _probe_catalog(case)
    profile = np.array(case["profile"], dtype=np.float32)
    uncertainty = (
        np.array(case["uncertainty"], dtype=np.float32)
        if case.get("uncertainty") is not None
        else None
    )
    # Omit pool_top entirely when the case doesn't set it, so choose_probe's own default applies.
    extra = {"pool_top": case["pool_top"]} if "pool_top" in case else {}
    probe = probes.choose_probe(
        conn,
        matrix,
        profile,
        excluded_movie_ids=frozenset(case.get("excluded_movie_ids", [])),
        asked_positions=frozenset(case.get("asked_positions", [])),
        uncertainty=uncertainty,
        **extra,
    )
    conn.close()
    if probe is None:
        return {"probe": None}
    return {
        "probe": {
            "axis_position": probe.axis_position,
            "axis_name": probe.axis_name,
            "spread": float(probe.spread),
            "film_high_id": probe.film_high.movie_id,
            "film_low_id": probe.film_low.movie_id,
            "question": probe.question,
        }
    }


def run_stopping_case(case: dict) -> dict:
    """Run the deterministic stopping rule for one turn's state."""
    decision = probes.should_stop(
        turn=case["turn"],
        top5_history=[list(h) for h in case["top5_history"]],
        top_scores=case.get("top_scores"),
        user_requested=case.get("user_requested", False),
    )
    return {"stop": decision.stop, "reason": decision.reason}


def run_escape_hatch_case(case: dict) -> dict:
    """Whether a line of user text trips the always-available escape hatch."""
    return {"wants_to_stop": probes.wants_to_stop(case["text"])}


# -- coverage (coverage.py) ----------------------------------------------------------


def run_coverage_region_case(case: dict) -> dict:
    """Measure the weighted region coverage and nearest-neighbour cosine for a centroid."""
    centroid = np.array(case["centroid"], dtype=np.float32)
    vectors = np.array(case["vectors"], dtype=np.float32).reshape(
        len(case["vectors"]), len(case["centroid"])
    )
    cov = np.array(case["coverage"], dtype=np.float64)
    top_k = case.get("top_k", coverage.COVERAGE_TOP_K)
    return {
        "region_coverage": coverage.region_coverage(centroid, vectors, cov, top_k=top_k),
        "nearest_cosine": coverage.nearest_cosine(centroid, vectors),
    }


def run_coverage_verdict_case(case: dict) -> dict:
    """Decide the honesty verdict from a region coverage and a nearest cosine."""
    reasons = coverage.honesty_reasons(case["region_coverage"], case["nearest_cosine"])
    return {"honest": bool(reasons), "reasons": list(reasons)}
