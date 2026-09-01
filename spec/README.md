# `spec/` — the shared scoring contract

The recommender's deterministic core exists twice: once in Python (`src/cinegeist/`, the CLI) and,
from session 6, once in TypeScript (`web/src/`, the browser demo). They **will** drift, and the
failure is silent — the demo quietly recommends different films than the app and nobody notices for
weeks (plan.md §8.6).

This directory is what stops that. It holds language-neutral JSON: a registry of every shared
constant, and a set of `(input, expected output)` cases for scoring, decay, probe selection,
stopping, and coverage. **Both** test suites load these same files and assert their implementation
reproduces them:

- Python — `tests/test_spec.py`, part of `make test`, runs in CI now.
- TypeScript — `web/` via `vitest`, arrives in session 6, runs in the same CI.

Change a scoring rule and the committed answers stop matching the code; CI goes red until the
fixtures are regenerated — which forces a review that has to carry over to the other language.
That red build is the entire point.

## How it fits together

```
scripts/build_spec_fixtures.py   authors the INPUT of each case, runs the real Python to
                                 compute the EXPECTED output, writes the JSON here
tests/spec_runner.py             the one definition of "run a case against the reference",
                                 imported by BOTH the generator and the pytest reader
tests/test_spec.py               loads the committed cases, re-runs them, asserts they match
```

Because the generator and the reader funnel through the same `spec_runner`, they can never
disagree about what the code does. Regenerate with:

```bash
make spec        # == python scripts/build_spec_fixtures.py
```

Then commit the changed fixtures **with** the code change (and, from session 6, the TypeScript
change that the diff implies).

## Constants — `constants.json`

The canonical registry of every tunable weight and threshold, grouped by area. The constants still
*live* in the Python modules that use them at runtime; this file mirrors them, and
`test_constants_match_the_modules` asserts the two are equal. The TypeScript port reads its values
from here. To change one: edit the Python constant, run `make spec`, review the diff, update the
port. Editing only one side turns CI red.

## The algorithm, in prose (the tiebreaker)

When a fixture and your reading of the code disagree, this section decides. When this section and
the code disagree, that is a bug in one of them — resolve it, don't paper over it.

### Scoring — `scoring/cases.json` (mirrors `recommend/score.py`)

Per film, a weighted combination, each term degrading to a neutral prior when its input is missing:

```
score = ( W_COSINE   · cosine(profile, film)
        + W_QUALITY  · quality               # Bayesian-shrunk TMDB rating, in [0,1]
        + W_SESSION  · cosine(session, film) # 0 when there is no session vector
        + W_FACET    · facet                 # per-film facet match in [-1,1]; 0 in the demo
        − W_POPULARITY · popularity_penalty ) # log1p popularity vs a reference, in [0,1]
        × confidence                         # PREDICTED_CONFIDENCE for predicted vectors, else 1
```

- **quality** = `(votes·avg + QUALITY_PRIOR_COUNT·QUALITY_PRIOR_MEAN) / (votes + QUALITY_PRIOR_COUNT)
  / RATING_SCALE`, clamped to `[0,1]`. An unrated film returns the prior mean, not zero.
- **popularity_penalty** = `log1p(pop) / log1p(POPULARITY_REFERENCE)`, capped at 1; 0 when
  popularity is missing or non-positive.
- **cosine** floors at 0 for a zero vector or zero query (never NaN).

Then two reshaping steps:

- **MMR** (`MMR_LAMBDA`) reorders for relevance *and* diversity: each pick maximises
  `λ·relevance − (1−λ)·(max cosine to an already-picked film)`, where relevance is the combined
  score min-max normalised to `[0,1]` across the pool. This is what stops three near-identical
  films taking all three slots.
- **wildcard** — the highest-scoring film whose cosine to the profile is ≤ `WILDCARD_MAX_COSINE`
  and which loads ≥ `WILDCARD_TAG_RELEVANCE` on at least `WILDCARD_MIN_SHARED_TAGS` of the
  profile's strong axes. `null` when nothing qualifies. Never one of the confident picks.

### Decay — `scoring/decay.json` (mirrors `profile/update.py`)

The profile is a weighted centroid over the tag genome:

```
w_i     = value_i · weight_i · 0.5 ^ (age_days_i / HALF_LIFE_DAYS)
centroid = Σ (w_i · v_i) / Σ |w_i|      # v_i is the film's genome row, or a one-hot for a tag event
total_weight = Σ |w_i|                   # the evidence mass, our confidence signal
```

`decay_factor` is clamped to `≤ 1` (a future timestamp can't amplify). A key property: uniform time
decay scales numerator and denominator equally, so the centroid *direction* doesn't move as time
passes on its own — only new or forgotten evidence moves it; only `total_weight` shrinks.

**Ranked axes** carry the strongest signed affinities for display and explanation: take axes by
descending `|centroid|`, keep up to `AXES_PER_SIGN` per sign (dropping anything below
`AXIS_EPSILON`), attribute each to the single event that contributed most to it (its verbatim
evidence and a source label), then order the kept axes by signed weight, descending.

> **Tie caveat.** The order among axes with *exactly equal signed weight* is unspecified — it
> depends on a sort that isn't stable across languages. The fixtures are authored to avoid such
> ties. Don't rely on the ordering of equal-weight axes, and don't add a fixture that does.

### Probes and stopping — `scoring/probes.json` (mirrors `convo/probes.py`)

**Probe selection.** Score the genome-covered pool against the profile; the contested set is the
top `POOL_TOP` by cosine (or everything, at cold start / small pools). For each axis compute the
variance of relevance across the contested set, weighted by profile uncertainty; pick the maximum.
If it clears `MIN_SPREAD`, ground it in two real films: the high pole is the film strongest on the
axis, the low pole is the below-mean film otherwise closest (cosine) to it. `null` when nothing
divides the pool.

**Stopping** fires at the first of, in this order: the user asked (`user_request`, always wins);
`turn ≥ MAX_TURNS` (`max_turns`); then, only once `turn ≥ MIN_TURNS`, the top-5 identical across the
last `STABLE_TURNS` turns (`top5_stable`), or rank-1 minus rank-10 score `≥ MARGIN_THRESHOLD`
(`margin`). Otherwise `continue`. The escape hatch (`wants_to_stop`) matches a fixed set of
"just show me something" phrases, case-insensitively, and is offered every turn regardless.

### Coverage — `scoring/coverage.json` (mirrors `recommend/coverage.py`)

The honesty check for the demo's 2,000-film shard (plan.md §8.4). Each shard film carries a
per-film **coverage** fraction in `[0,1]`, measured at build time as the share of its
full-catalog neighbourhood (cosine ≥ `COVERAGE_SIMILARITY`) that survived into the shard.

```
nearest_cosine  = max cosine(centroid, film) over the shard          # 0 for an empty shard
region_coverage = Σ (w_k · coverage_k) / Σ w_k                        # 0 if Σ w_k == 0
                  over the COVERAGE_TOP_K films nearest the centroid,
                  w_k = max(0, cosine(centroid, k))
```

Take the honesty path when **either** `region_coverage < REGION_COVERAGE_MIN` (reason
`region_coverage_below_min`) **or** `nearest_cosine < NEAREST_COSINE_MIN` (reason
`no_close_neighbour`). Both comparisons are strict, so a value exactly at the threshold stays
served. Reasons are reported region-coverage first, nearest-neighbour second.

## Float tolerance

Numbers travel through JSON and are recomputed in float32 inside the genome maths, float64 in the
readers, and float64 again in JavaScript. "Match" means **within `1e-5` absolute**; integers,
strings, booleans, and `null` must match exactly. Cases are authored with mostly
exactly-representable values so this gap never decides a discrete outcome (a ranking, a chosen
axis, a verdict).
