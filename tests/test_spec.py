"""Assert the Python reference still reproduces the committed ``spec/`` fixtures (plan.md §8.6).

These are the fixtures the browser's TypeScript port loads too (from session 6), so this test is
one half of the drift guard: if a scoring, decay, probe, or coverage rule changes and the fixtures
aren't regenerated, the committed answers stop matching the code and this goes red — forcing a
deliberate regeneration and a matching change to the port. The other half is ``web/`` running the
same JSON through ``vitest``.

Every case is executed through :mod:`tests.spec_runner`, the exact code path
``scripts/build_spec_fixtures.py`` used to write the expected values, so a passing suite means the
two agree bit-for-bit (within the documented float tolerance).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import spec_runner

SPEC = Path(__file__).resolve().parent.parent / "spec"


def _load(relpath: str) -> dict:
    return json.loads((SPEC / relpath).read_text(encoding="utf-8"))


def _ids(cases: list[dict], key: str = "name") -> list[str]:
    return [str(case.get(key, i)) for i, case in enumerate(cases)]


# -- constants -----------------------------------------------------------------------


def test_spec_dir_and_constants_exist() -> None:
    assert (SPEC / "constants.json").exists(), "run `make spec` to generate the fixtures"
    assert (SPEC / "README.md").exists()


def test_constants_match_the_modules() -> None:
    committed = _load("constants.json")
    spec_runner.assert_close(spec_runner.constants(), committed, tol=0.0)


# -- scoring -------------------------------------------------------------------------

_SCORING = _load("scoring/cases.json")["cases"]


@pytest.mark.parametrize("case", _SCORING, ids=_ids(_SCORING))
def test_scoring_cases(case: dict) -> None:
    spec_runner.assert_close(spec_runner.run_scoring_case(case), case["expected"])


# -- decay ---------------------------------------------------------------------------

_DECAY = _load("scoring/decay.json")["cases"]


@pytest.mark.parametrize("case", _DECAY, ids=_ids(_DECAY))
def test_decay_cases(case: dict) -> None:
    spec_runner.assert_close(spec_runner.run_decay_case(case), case["expected"])


# -- probes and stopping -------------------------------------------------------------

_PROBES = _load("scoring/probes.json")


@pytest.mark.parametrize("case", _PROBES["probe_selection"], ids=_ids(_PROBES["probe_selection"]))
def test_probe_selection_cases(case: dict) -> None:
    spec_runner.assert_close(spec_runner.run_probe_case(case), case["expected"])


@pytest.mark.parametrize("case", _PROBES["stopping"], ids=_ids(_PROBES["stopping"]))
def test_stopping_cases(case: dict) -> None:
    spec_runner.assert_close(spec_runner.run_stopping_case(case), case["expected"])


@pytest.mark.parametrize("case", _PROBES["escape_hatch"], ids=_ids(_PROBES["escape_hatch"], "text"))
def test_escape_hatch_cases(case: dict) -> None:
    spec_runner.assert_close(spec_runner.run_escape_hatch_case(case), case["expected"])


# -- coverage ------------------------------------------------------------------------

_COVERAGE = _load("scoring/coverage.json")


@pytest.mark.parametrize("case", _COVERAGE["region"], ids=_ids(_COVERAGE["region"]))
def test_coverage_region_cases(case: dict) -> None:
    spec_runner.assert_close(spec_runner.run_coverage_region_case(case), case["expected"])


@pytest.mark.parametrize("case", _COVERAGE["verdict"], ids=_ids(_COVERAGE["verdict"]))
def test_coverage_verdict_cases(case: dict) -> None:
    spec_runner.assert_close(spec_runner.run_coverage_verdict_case(case), case["expected"])
