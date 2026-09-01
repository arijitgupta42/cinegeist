"""Generate the shared ``spec/`` fixtures from the live Python reference (plan.md §8.6).

The *inputs* of each case are authored here by hand — small, tie-free scenarios with mostly
exactly-representable numbers, so the float32/float64/JavaScript gap never decides a ranking. The
*expected outputs* are computed by running the real scoring, decay, probe, and coverage code
through :mod:`tests.spec_runner`, the same entry points the pytest reader uses. So this script and
the test can never disagree about what the code does; regenerating simply re-derives the committed
answers, and any genuine behaviour change shows up as a reviewable diff in ``spec/``.

Run it with ``make spec`` (or ``python scripts/build_spec_fixtures.py``) whenever a scoring rule
changes on purpose, then commit the updated fixtures alongside the code — and, from session 6 on,
the matching change to the TypeScript port that reads these same files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import spec_runner  # noqa: E402  (needs the tests dir on sys.path first)

SPEC = ROOT / "spec"


# -- scoring cases -------------------------------------------------------------------


def scoring_cases() -> list[dict]:
    # The 8-dim wildcard pool from the unit tests: F is too close, G shares one strong tag, E
    # shares two but sits far — so E is the only valid wildcard.
    wildcard_films = [
        {"movie_id": 10, "vector": [1.0, 1.0, 0, 0, 0, 0, 0, 0]},
        {"movie_id": 20, "vector": [0.6, 0.0, 1, 1, 1, 1, 1, 1]},
        {"movie_id": 30, "vector": [0.6, 0.6, 1, 1, 1, 1, 1, 1]},
    ]
    return [
        {
            "name": "profile axis and the quality prior",
            "n_tags": 2,
            "profile": [1.0, 0.0],
            "films": [
                {"movie_id": 1, "vector": [1.0, 0.0]},
                {"movie_id": 2, "vector": [0.0, 1.0]},
            ],
        },
        {
            "name": "quality lifts and popularity penalises an off-taste film",
            "n_tags": 2,
            "profile": [1.0, 0.0],
            "films": [
                {
                    "movie_id": 1,
                    "vector": [0.0, 1.0],
                    "vote_average": 8.0,
                    "vote_count": 10000,
                    "popularity": 50.0,
                }
            ],
        },
        {
            "name": "session intent adds a fit term for tonight",
            "n_tags": 2,
            "profile": [1.0, 0.0],
            "session": [0.0, 1.0],
            "films": [{"movie_id": 1, "vector": [0.0, 1.0]}],
        },
        {
            "name": "a predicted vector is discounted",
            "n_tags": 2,
            "profile": [1.0, 0.0],
            "films": [
                {"movie_id": 1, "vector": [1.0, 0.0], "genome_source": "measured"},
                {"movie_id": 2, "vector": [1.0, 0.0], "genome_source": "predicted"},
            ],
        },
        {
            "name": "mmr keeps a duplicate under pure relevance but drops it for diversity",
            "n_tags": 2,
            "profile": [1.0, 0.0],
            "lam": 0.3,
            "n_confident": 2,
            "with_wildcard": False,
            "films": [
                {"movie_id": 1, "vector": [1.0, 0.0]},
                {"movie_id": 2, "vector": [1.0, 0.0]},
                {"movie_id": 3, "vector": [0.0, 1.0]},
            ],
        },
        {
            "name": "wildcard is far from taste yet shares two strong tags",
            "n_tags": 8,
            "profile": [1.0, 1.0, 0, 0, 0, 0, 0, 0],
            "strong_tag_positions": [0, 1],
            "n_confident": 1,
            "films": wildcard_films,
        },
        {
            "name": "a facet match nudges an otherwise off-taste film",
            "n_tags": 3,
            "profile": [1.0, 0.0, 0.0],
            "facet_scores": [0.0, 1.0],
            "films": [
                {"movie_id": 1, "vector": [1.0, 0.0, 0.0]},
                {"movie_id": 2, "vector": [0.0, 1.0, 0.0]},
            ],
        },
        {
            "name": "an empty pool yields nothing",
            "n_tags": 4,
            "profile": [0.0, 0.0, 0.0, 0.0],
            "films": [],
        },
    ]


# -- decay cases ---------------------------------------------------------------------

_TAGS3 = [
    {"tag_id": 1, "position": 0, "name": "warm"},
    {"tag_id": 2, "position": 1, "name": "bleak"},
    {"tag_id": 3, "position": 2, "name": "space"},
]


def _movie_event(subject: int, value: float, age_days: float, evidence: str) -> dict:
    kind = "liked_movie" if value >= 0 else "disliked_movie"
    return {
        "kind": kind,
        "subject_kind": "movie",
        "subject": subject,
        "value": value,
        "weight": 1.0,
        "age_days": age_days,
        "evidence": evidence,
    }


def decay_cases() -> list[dict]:
    return [
        {
            "name": "one like at exactly one half-life halves the evidence weight",
            "n_tags": 3,
            "tags": _TAGS3,
            "half_life_days": 270.0,
            "movies": [
                {"movie_id": 1, "genome_row": 0, "clean_title": "Toy Story",
                 "vector": [0.9, 0.1, 0.2]}
            ],
            "events": [_movie_event(1, 1.0, 270.0, "loved it as a kid")],
        },
        {
            # Distinct averaged components on purpose: equal-weight axes have no defined order
            # (see spec/README.md), so tie-free vectors keep the ranked axes reproducible.
            "name": "two fresh likes average into the centroid",
            "n_tags": 3,
            "tags": _TAGS3,
            "movies": [
                {"movie_id": 1, "genome_row": 0, "clean_title": "A", "vector": [0.9, 0.1, 0.2]},
                {"movie_id": 2, "genome_row": 1, "clean_title": "B", "vector": [0.3, 0.7, 0.4]},
            ],
            "events": [
                _movie_event(1, 1.0, 0.0, "a"),
                _movie_event(2, 1.0, 0.0, "b"),
            ],
        },
        {
            "name": "a dislike pushes away from a film's vector",
            "n_tags": 3,
            "tags": _TAGS3,
            "movies": [
                {"movie_id": 1, "genome_row": 0, "clean_title": "A", "vector": [1.0, 0.0, 0.0]},
                {"movie_id": 2, "genome_row": 1, "clean_title": "B", "vector": [0.0, 1.0, 0.0]},
            ],
            "events": [
                _movie_event(1, 1.0, 0.0, "loved A"),
                _movie_event(2, -1.0, 0.0, "hated B"),
            ],
        },
        {
            "name": "an axis answer pulls toward a single tag",
            "n_tags": 3,
            "tags": _TAGS3,
            "movies": [],
            "events": [
                {
                    "kind": "axis_answer",
                    "subject_kind": "tag",
                    "subject": 2,
                    "value": 0.8,
                    "weight": 1.0,
                    "age_days": 0.0,
                    "evidence": "I like it bleak",
                }
            ],
        },
        {
            "name": "decay shrinks evidence weight but not the centroid direction",
            "n_tags": 3,
            "tags": _TAGS3,
            "movies": [
                {"movie_id": 1, "genome_row": 0, "clean_title": "A", "vector": [0.6, 0.8, 0.0]}
            ],
            "events": [
                _movie_event(1, 1.0, 0.0, "recent"),
                _movie_event(1, 1.0, 270.0, "long ago"),
            ],
        },
        {
            "name": "ranked axes attribute each affinity to its strongest evidence",
            "n_tags": 3,
            "tags": _TAGS3,
            "movies": [
                {"movie_id": 1, "genome_row": 0, "clean_title": "Spirited Away",
                 "vector": [0.9, 0.1, 0.2]},
                {"movie_id": 2, "genome_row": 1, "clean_title": "Solaris",
                 "vector": [0.1, 0.8, 0.6]},
            ],
            "events": [
                _movie_event(1, 1.0, 0.0, "the animation was gorgeous"),
                _movie_event(2, -1.0, 0.0, "too cold and cerebral"),
            ],
        },
    ]


# -- probe cases ---------------------------------------------------------------------

_PROBE_TAGS = [
    {"tag_id": 100, "position": 0, "name": "warm"},
    {"tag_id": 200, "position": 1, "name": "bleak"},
    {"tag_id": 300, "position": 2, "name": "even"},
]
_PROBE_MOVIES = [
    {"movie_id": 1, "genome_row": 0, "title": "F1 (2001)", "clean_title": "F1", "year": 2001,
     "vector": [0.2, 0.0, 0.9]},
    {"movie_id": 2, "genome_row": 1, "title": "F2 (2002)", "clean_title": "F2", "year": 2002,
     "vector": [0.8, 1.0, 0.9]},
    {"movie_id": 3, "genome_row": 2, "title": "F3 (2003)", "clean_title": "F3", "year": 2003,
     "vector": [0.8, 0.0, 0.9]},
    {"movie_id": 4, "genome_row": 3, "title": "F4 (2004)", "clean_title": "F4", "year": 2004,
     "vector": [0.2, 1.0, 0.9]},
]


def _probe_case(name: str, **over: Any) -> dict:
    base = {"name": name, "n_tags": 3, "tags": _PROBE_TAGS, "movies": _PROBE_MOVIES,
            "profile": [0.0, 0.0, 0.0]}
    base.update(over)
    return base


def probe_selection_cases() -> list[dict]:
    return [
        _probe_case("cold start picks the highest-spread axis"),
        _probe_case("uncertainty weight can flip the axis", uncertainty=[1.0, 0.1, 1.0]),
        _probe_case("an already-asked axis is not repeated", asked_positions=[1]),
        _probe_case("excluding films changes the contested set", excluded_movie_ids=[2, 4]),
        _probe_case("a profile narrows the contested set", profile=[0.0, 0.0, 1.0], pool_top=2),
        {
            "name": "nothing divides an identical pool",
            "n_tags": 1,
            "tags": [{"tag_id": 1, "position": 0, "name": "flat"}],
            "movies": [
                {"movie_id": 1, "genome_row": 0, "title": "A (1)", "clean_title": "A", "year": 1,
                 "vector": [0.5]},
                {"movie_id": 2, "genome_row": 1, "title": "B (2)", "clean_title": "B", "year": 2,
                 "vector": [0.5]},
            ],
            "profile": [0.0],
        },
    ]


def stopping_cases() -> list[dict]:
    return [
        {"name": "user request stops immediately even on turn zero", "turn": 0,
         "top5_history": [], "user_requested": True},
        {"name": "hard turn cap", "turn": 9, "top5_history": [[1, 2, 3, 4, 5]]},
        {"name": "does not stop before the minimum", "turn": 1,
         "top5_history": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]},
        {"name": "stops when the top five holds steady", "turn": 4,
         "top5_history": [[9, 8, 7, 6, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]},
        {"name": "stops on a confident margin", "turn": 4,
         "top5_history": [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5]],
         "top_scores": [0.9, 0.5, 0.4, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.2]},
        {"name": "keeps going when no rule fires", "turn": 4,
         "top5_history": [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5]],
         "top_scores": [0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41]},
    ]


def escape_hatch_cases() -> list[dict]:
    return [
        {"text": "just show me something"},
        {"text": "Just tell me already"},
        {"text": "ok, STOP ASKING and pick one"},
        {"text": "I loved Heat and Prisoners"},
    ]


# -- coverage cases ------------------------------------------------------------------


def coverage_region_cases() -> list[dict]:
    return [
        {
            "name": "a well-covered neighbourhood",
            "centroid": [1.0, 0.0],
            "vectors": [[1.0, 0.0], [0.96, 0.28], [0.0, 1.0]],
            "coverage": [1.0, 0.8, 0.2],
            "top_k": 2,
        },
        {
            "name": "a thin neighbourhood the shard barely kept",
            "centroid": [1.0, 0.0],
            "vectors": [[1.0, 0.0], [0.96, 0.28]],
            "coverage": [0.2, 0.1],
        },
        {
            "name": "top_k above the pool size is clamped",
            "centroid": [0.0, 1.0],
            "vectors": [[0.0, 1.0], [0.28, 0.96]],
            "coverage": [0.5, 0.5],
            "top_k": 25,
        },
        {
            "name": "nothing points toward the centroid",
            "centroid": [1.0, 0.0],
            "vectors": [[-1.0, 0.0], [0.0, 1.0]],
            "coverage": [0.9, 0.9],
        },
    ]


def coverage_verdict_cases() -> list[dict]:
    return [
        {"name": "served well", "region_coverage": 0.9, "nearest_cosine": 0.9},
        {"name": "thin region only", "region_coverage": 0.2, "nearest_cosine": 0.9},
        {"name": "no close neighbour only", "region_coverage": 0.9, "nearest_cosine": 0.4},
        {"name": "both triggers fire", "region_coverage": 0.1, "nearest_cosine": 0.3},
        {"name": "region coverage exactly at the threshold stays served",
         "region_coverage": 0.25, "nearest_cosine": 0.9},
        {"name": "nearest cosine exactly at the threshold stays served",
         "region_coverage": 0.9, "nearest_cosine": 0.45},
    ]


# -- driver --------------------------------------------------------------------------


def _attach(cases: list[dict], runner: Callable[[dict], dict]) -> list[dict]:
    return [{**case, "expected": runner(case)} for case in cases]


def _write(relpath: str, obj: Any) -> None:
    path = SPEC / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force LF regardless of platform, so the committed fixtures are byte-identical everywhere and
    # vitest reads exactly what pytest wrote.
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    _write("constants.json", spec_runner.constants())
    _write("scoring/cases.json", {"cases": _attach(scoring_cases(), spec_runner.run_scoring_case)})
    _write("scoring/decay.json", {"cases": _attach(decay_cases(), spec_runner.run_decay_case)})
    _write(
        "scoring/probes.json",
        {
            "probe_selection": _attach(probe_selection_cases(), spec_runner.run_probe_case),
            "stopping": _attach(stopping_cases(), spec_runner.run_stopping_case),
            "escape_hatch": _attach(escape_hatch_cases(), spec_runner.run_escape_hatch_case),
        },
    )
    _write(
        "scoring/coverage.json",
        {
            "region": _attach(coverage_region_cases(), spec_runner.run_coverage_region_case),
            "verdict": _attach(coverage_verdict_cases(), spec_runner.run_coverage_verdict_case),
        },
    )


if __name__ == "__main__":
    main()
