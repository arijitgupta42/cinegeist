"""A returning viewer over several sessions, weeks apart — the eval for persistence and decay.

The precision harness (:mod:`harness`) runs one session and asks *how good are the picks*. This
asks a different question the plan calls out (§16): does the *persisted* profile genuinely adjust
across repeated real sessions the way the decay maths intends? A profile is meant to sharpen as
consistent evidence accumulates and to let old, off-taste evidence fade rather than being deleted
(``update.py``: ``w = value × weight × 0.5 ** (age_days / HALF_LIFE)``).

So we simulate one viewer across ``n_visits`` visits ``gap_days`` apart:

* **Visit 1** — still figuring their taste out: two loved-cluster reactions and one *misfire*, a
  liked film from the cluster they'll turn out to dislike.
* **Later visits** — settled: two fresh loved-cluster reactions each, at that visit's timestamp.

Every reaction is a real append-only event under one ``user_id`` (``store.append_events``), and the
profile is recomputed from the whole log at each visit (``update.compute_profile``) — the exact
persist-and-decay path the CLI runs. We then read, per visit:

* the profile's cosine to the persona's true loved-cluster direction (how *sharp* it is),
* the decayed-weight share still carried by the visit-1 misfire (how far old evidence has *faded*),
* the session and event counts and the evidence mass (that history persisted and grew).

A healthy system sharpens (cosine rises) while the misfire fades (its share falls), and the engine
still greets the viewer as returning. Those are the invariants the CI test guards; the numbers move
when the decay maths does, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np

from ..catalog import genome
from ..config import Settings
from ..convo.engine import Engine
from ..profile import store, update
from ..profile.model import PreferenceEvent
from .catalog import SyntheticCatalog, build_synthetic_catalog
from .oracle import PersonaOracle
from .personas import PERSONAS, Persona, taste_direction

# A fixed first-visit clock so the whole run is deterministic; later visits are offsets from it.
_CLOCK = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

# The default returning viewer: a clear taste with a distinct cluster it dislikes, so the visit-1
# misfire (a film from the disliked cluster) is genuinely off-taste and its fading is measurable.
_DEFAULT_PERSONA = "neo noir"


@dataclass(frozen=True)
class Visit:
    """One visit's snapshot of the persisted, decayed profile."""

    label: str
    session_count: int
    event_count: int
    total_weight: float
    taste_cosine: float  # cosine of the decayed centroid to the persona's true loved direction
    misfire_share: float  # decayed-weight fraction still carried by the visit-1 off-taste like


@dataclass(frozen=True)
class ReturningResult:
    """A returning viewer's trajectory across visits, plus whether the engine recognised them."""

    persona: Persona
    n_visits: int
    gap_days: int
    visits: tuple[Visit, ...]
    welcomed_back: bool  # the engine's greeting recognised the returning viewer

    @property
    def sharpened(self) -> bool:
        """Did the profile get closer to the true taste from the first visit to the last?"""
        return self.visits[-1].taste_cosine > self.visits[0].taste_cosine

    @property
    def old_evidence_faded(self) -> bool:
        """Did the visit-1 misfire's share of the evidence shrink as fresh evidence accrued?"""
        return self.visits[-1].misfire_share < self.visits[0].misfire_share


def _cosine(vector: np.ndarray, direction: np.ndarray) -> float:
    return float(genome.cosine_scores(vector[np.newaxis, :], direction)[0])


def _misfire_share(
    conn, user_id: str, misfire_movie_id: int, now: datetime, half_life: float
) -> float:
    """The fraction of the log's total decayed weight still carried by the misfire event."""
    total = 0.0
    misfire = 0.0
    for event in store.iter_events(conn, user_id):
        ts = event.ts or now
        age_days = (now - ts).total_seconds() / 86_400.0
        weight = abs(event.value * event.weight * update.decay_factor(age_days, half_life))
        total += weight
        if event.movie_id == misfire_movie_id:
            misfire += weight
    return misfire / total if total > 0 else 0.0


def run_returning_persona(
    catalog: SyntheticCatalog,
    *,
    persona_name: str = _DEFAULT_PERSONA,
    n_visits: int = 4,
    gap_days: int = 45,
    now: datetime = _CLOCK,
    half_life: float = update.HALF_LIFE_DAYS,
) -> ReturningResult:
    """Simulate a returning viewer over several visits, measuring the profile at each one."""
    if n_visits < 2:
        raise ValueError("a returning viewer needs at least two visits")
    conn = catalog.conn
    persona = next(p for p in PERSONAS if p.name == persona_name)
    if persona.hated is None:
        raise ValueError("the returning persona needs a disliked cluster for the misfire")

    user_id = f"returning-{persona.name}"
    store.reset_profile(conn, user_id)  # clean slate if the eval is run more than once

    loved = catalog.films_in_cluster[persona.loved]
    hated = catalog.films_in_cluster[persona.hated]
    needed = 2 * n_visits
    if len(loved) < needed:
        raise ValueError(f"cluster {persona.loved!r} has too few films for {n_visits} visits")
    misfire_id = hated[len(hated) // 2]  # a mid-cluster film, well away from the seed dislikes
    true_direction = catalog.cluster_vector[persona.loved].astype(np.float32)

    def append(events: list[PreferenceEvent], at: datetime) -> None:
        store.append_events(conn, [replace(e, user_id=user_id) for e in events], now=at)

    visits: list[Visit] = []
    for k in range(n_visits):
        at = now + timedelta(days=gap_days * k)
        session_id = f"visit-{k + 1}"
        a, b = loved[2 * k], loved[2 * k + 1]
        events = [
            PreferenceEvent.liked_movie(a, evidence="loved this", session_id=session_id),
            PreferenceEvent.liked_movie(b, evidence="loved this", session_id=session_id),
        ]
        if k == 0:  # the first visit's misfire: a like from the cluster they'll dislike
            events.append(
                PreferenceEvent.liked_movie(
                    misfire_id, evidence="thought I'd like this", session_id=session_id
                )
            )
        append(events, at)

        profile = update.compute_profile(
            conn, catalog.matrix, user_id=user_id, now=at, half_life=half_life
        )
        visits.append(
            Visit(
                label=session_id,
                session_count=profile.session_count,
                event_count=profile.event_count,
                total_weight=profile.total_weight,
                taste_cosine=_cosine(profile.genome_vector, true_direction),
                misfire_share=_misfire_share(conn, user_id, misfire_id, at, half_life),
            )
        )

    # The viewer comes back once more: does the shipping engine recognise them as returning?
    last_at = now + timedelta(days=gap_days * (n_visits - 1))
    oracle = PersonaOracle(catalog=catalog, direction=taste_direction(persona, catalog))
    Engine(
        conn,
        catalog.matrix,
        oracle,
        Settings(),
        offline=True,
        user_id=user_id,
        session_id="return-visit",
        now=last_at,
    ).run()
    welcomed_back = bool(oracle.narration) and oracle.narration[0].startswith("Welcome back")

    return ReturningResult(
        persona=persona,
        n_visits=n_visits,
        gap_days=gap_days,
        visits=tuple(visits),
        welcomed_back=welcomed_back,
    )


def run_returning_eval(*, seed: int = 0, n_visits: int = 4, gap_days: int = 45) -> ReturningResult:
    """Build the fixture catalog and run the default returning viewer over it."""
    catalog = build_synthetic_catalog(seed=seed)
    return run_returning_persona(catalog, n_visits=n_visits, gap_days=gap_days)
