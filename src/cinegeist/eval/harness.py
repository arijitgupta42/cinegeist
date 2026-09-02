"""Run a synthetic persona through the real offline engine and score the result.

This is the whole harness: seed a persona with a couple of opening reactions (the seed-question
equivalent), let the deterministic engine hold the rest of the conversation against the persona
oracle, then measure how many of the three confident picks land in the persona's loved cluster —
precision@3. Nothing here reimplements the recommender; it drives the same
:class:`~cinegeist.convo.engine.Engine` the CLI runs in ``--offline`` mode, so the number reflects
the shipping retrieval, scoring, MMR, probe selection, and stopping rules. That is what makes it a
knob you can turn: change a weight in ``score.py`` and this precision moves.

Each persona runs under its own ``user_id`` in one shared in-memory catalog, so their event logs
and profiles never bleed together. The run is fully deterministic (seeded catalog, fixed clock),
so a change in the reported precision is a change in the recommender, not the fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ..config import Settings
from ..convo.engine import Engine
from ..profile import store
from ..profile.model import PreferenceEvent
from .catalog import SyntheticCatalog, build_synthetic_catalog
from .oracle import PersonaOracle, ProbeTurn
from .personas import PERSONAS, Persona, relevant_ids, seed_film_ids, taste_direction

# A fixed clock so decay is constant across runs: every seed event is stamped "now", age zero.
_CLOCK = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class PersonaResult:
    """What one persona's simulated conversation produced, and how well it scored."""

    persona: Persona
    precision_at_3: float
    hits: int
    n_picks: int
    n_probes: int
    pick_titles: tuple[str, ...]
    pick_clusters: tuple[str, ...]
    wildcard_title: str | None
    probe_turns: tuple[ProbeTurn, ...]


@dataclass(frozen=True)
class EvalReport:
    """The precision of every persona and the mean across them — the headline number."""

    seed: int
    catalog_size: int
    results: tuple[PersonaResult, ...]

    @property
    def mean_precision_at_3(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.precision_at_3 for r in self.results) / len(self.results)


def run_persona(persona: Persona, catalog: SyntheticCatalog) -> PersonaResult:
    """Seed, converse, and score one persona against the shared synthetic catalog."""
    conn = catalog.conn
    user_id = persona.name
    session_id = f"eval-{persona.name}"
    store.reset_profile(conn, user_id)  # a clean slate if the harness is run more than once

    loved_seeds, hated_seeds = seed_film_ids(persona, catalog)
    seed_events: list[PreferenceEvent] = [
        PreferenceEvent.liked_movie(
            mid, evidence=f"loved this {persona.loved}", session_id=session_id
        )
        for mid in loved_seeds
    ]
    seed_events += [
        PreferenceEvent.disliked_movie(
            mid, evidence=f"bounced off this {persona.hated}", session_id=session_id
        )
        for mid in hated_seeds
    ]
    # The event factories default user_id to "default"; stamp this persona's id so the engine,
    # which reads the log under that id, actually sees these seeds.
    seed_events = [replace(event, user_id=user_id) for event in seed_events]
    store.append_events(conn, seed_events, now=_CLOCK)

    oracle = PersonaOracle(catalog=catalog, direction=taste_direction(persona, catalog))
    engine = Engine(
        conn,
        catalog.matrix,
        oracle,
        Settings(),
        offline=True,
        user_id=user_id,
        session_id=session_id,
        now=_CLOCK,
    )
    engine.run()

    relevant = relevant_ids(persona, catalog)
    picks = list(oracle.presentation.picks) if oracle.presentation else []
    pick_ids = [p.film.movie_id for p in picks]
    hits = sum(1 for pid in pick_ids if pid in relevant)
    n_picks = len(pick_ids)
    precision = hits / n_picks if n_picks else 0.0
    wildcard = oracle.presentation.wildcard if oracle.presentation else None

    return PersonaResult(
        persona=persona,
        precision_at_3=precision,
        hits=hits,
        n_picks=n_picks,
        n_probes=len(oracle.probe_turns),
        pick_titles=tuple(p.film.title for p in picks),
        pick_clusters=tuple(catalog.cluster_of[pid] for pid in pick_ids),
        wildcard_title=wildcard.film.title if wildcard else None,
        probe_turns=tuple(oracle.probe_turns),
    )


def run_eval(*, seed: int = 0) -> EvalReport:
    """Build the fixture catalog, run every persona, and aggregate precision@3."""
    catalog = build_synthetic_catalog(seed=seed)
    results = tuple(run_persona(persona, catalog) for persona in PERSONAS)
    return EvalReport(seed=seed, catalog_size=catalog.matrix.shape[0], results=results)
