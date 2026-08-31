"""Choose the next question by how much it teaches us, and decide when to stop asking.

This is the interesting part of the conversation, and it is maths, not prompting (plan.md §5.2).
Given the current profile, we score the genome-covered catalog, keep the high-scoring films still
in contention, and ask: which taste axis most *divides* that contested set? An axis where the
contenders all look the same teaches nothing; an axis that spreads them wide halves the search.
We pick the axis of maximum spread, weighted down where the profile is already confident, then
ground the question in two real films at opposite poles of that axis — the low-pole film chosen
to be otherwise as similar as possible to the high-pole one, so the contrast is about that axis
and little else. The LLM's only job, later, is to phrase the pair; the choice is made here.

The same code serves the cold open: with no profile yet, every film is in contention and the
axis of greatest spread across the catalog is the most discriminating first question.

Stopping is deterministic too (plan.md §5.3): stop when the top five stop moving, when we're
clearly confident, at a hard turn cap, or the moment the user asks — and an escape hatch is
offered on every turn regardless.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from ..catalog import genome

# How many of the highest-scoring films form the "contested" set an axis is judged against.
_POOL_TOP = 200
# An axis must spread the contested films at least this much (variance of relevance) to be worth
# asking about; below it, everything looks the same and there is no question to ask.
_MIN_SPREAD = 1e-4

# -- stopping-rule constants (provisional; they move to spec/ in session 5) ----------
MAX_TURNS = 9
MIN_TURNS = 2  # don't declare victory after a single answer (the user's stop always overrides)
STABLE_TURNS = 2  # stop once the top-5 is identical across this many most-recent turns
MARGIN_THRESHOLD = 0.15  # rank-1 minus rank-10 score gap that means "confident enough"

# Phrases that mean "stop quizzing me and show me something" — the escape hatch, honoured always.
_STOP_PATTERNS = (
    "just show me",
    "show me something",
    "just tell me",
    "just pick",
    "just recommend",
    "stop asking",
    "enough questions",
    "get on with it",
)


@dataclass(frozen=True)
class ProbeFilm:
    """One pole of a probe: a real film the user is asked to react to."""

    movie_id: int
    title: str
    year: int | None


@dataclass(frozen=True)
class Probe:
    """The next question: an axis to split, grounded in two films at its poles.

    ``film_high`` sits high on ``axis_name``; ``film_low`` sits low but is otherwise as close to
    ``film_high`` as the contested pool allows. ``question`` is a plain deterministic phrasing used
    in offline mode; online, the LLM rephrases it. Which film the user picks tells us the sign of
    their feeling on this axis.
    """

    axis_position: int
    axis_name: str
    spread: float
    film_high: ProbeFilm
    film_low: ProbeFilm
    question: str


@dataclass
class _Pool:
    ids: list[int]
    titles: list[str]
    years: list[int | None]
    vectors: np.ndarray  # (n_films, n_tags), the films' genome rows


def _load_pool(
    conn: sqlite3.Connection, matrix: np.ndarray, excluded: frozenset[int]
) -> _Pool | None:
    rows = conn.execute(
        "SELECT movie_id, clean_title, title, year, genome_row "
        "FROM movies WHERE genome_row IS NOT NULL ORDER BY genome_row"
    ).fetchall()
    ids: list[int] = []
    titles: list[str] = []
    years: list[int | None] = []
    genome_rows: list[int] = []
    for row in rows:
        if row["movie_id"] in excluded:
            continue
        ids.append(row["movie_id"])
        titles.append(row["clean_title"] or row["title"])
        years.append(row["year"])
        genome_rows.append(row["genome_row"])
    if not ids:
        return None
    vectors = np.asarray(matrix[genome_rows], dtype=np.float32)
    return _Pool(ids=ids, titles=titles, years=years, vectors=vectors)


def _default_uncertainty(profile_vector: np.ndarray, n_tags: int) -> np.ndarray:
    """Per-axis weight: axes the profile already has a strong opinion on are worth asking less.

    A proxy — the magnitude of the centroid on an axis stands in for how settled it is — so a
    caller with a real per-axis confidence can pass its own ``uncertainty`` instead.
    """
    if profile_vector is None or not profile_vector.any():
        return np.ones(n_tags, dtype=np.float32)
    return 1.0 - np.minimum(1.0, np.abs(profile_vector.astype(np.float32)))


def _tag_names(conn: sqlite3.Connection) -> dict[int, str]:
    return {
        row["position"]: row["name"]
        for row in conn.execute("SELECT position, name FROM genome_tags")
    }


def _ground_pair(contested: np.ndarray, position: int) -> tuple[int, int] | None:
    """Pick (high-pole index, low-pole index) within the contested set for one axis.

    The high pole is the film most strongly on the axis; the low pole is the film weakly on it
    that is otherwise closest to the high pole, so the two differ mainly on this axis.
    """
    relevance = contested[:, position]
    high = int(np.argmax(relevance))
    below = np.flatnonzero(relevance < relevance.mean())
    if below.size == 0:
        low = int(np.argmin(relevance))
    else:
        sims = genome.cosine_scores(contested[below], contested[high])
        low = int(below[int(np.argmax(sims))])
    if high == low:
        return None
    return high, low


def choose_probe(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    profile_vector: np.ndarray,
    *,
    excluded_movie_ids: frozenset[int] = frozenset(),
    asked_positions: frozenset[int] = frozenset(),
    uncertainty: np.ndarray | None = None,
    pool_top: int = _POOL_TOP,
) -> Probe | None:
    """Choose the most informative next probe, or ``None`` when nothing left divides the pool.

    With a zero ``profile_vector`` this is the cold open: the whole pool is contested and the axis
    of greatest spread across the catalog wins. Otherwise the contested set is the top ``pool_top``
    films by cosine to the profile, and the axis is chosen to spread *those*.
    """
    pool = _load_pool(conn, matrix, excluded_movie_ids)
    if pool is None:
        return None
    n_tags = int(matrix.shape[1])

    cold_start = not profile_vector.any()
    if cold_start or len(pool.ids) <= pool_top:
        contested_idx = np.arange(len(pool.ids))
    else:
        scores = genome.cosine_scores(pool.vectors, profile_vector)
        contested_idx = np.argsort(scores)[::-1][:pool_top]
    contested = pool.vectors[contested_idx]

    weight = (
        uncertainty if uncertainty is not None else _default_uncertainty(profile_vector, n_tags)
    )
    spread = np.var(contested, axis=0) * weight
    if asked_positions:
        spread[list(asked_positions)] = -1.0

    position = int(np.argmax(spread))
    if float(spread[position]) < _MIN_SPREAD:
        return None  # nothing divides the contenders any more

    grounded = _ground_pair(contested, position)
    if grounded is None:
        return None
    high_local, low_local = grounded
    high_idx = int(contested_idx[high_local])
    low_idx = int(contested_idx[low_local])

    names = _tag_names(conn)
    film_high = ProbeFilm(pool.ids[high_idx], pool.titles[high_idx], pool.years[high_idx])
    film_low = ProbeFilm(pool.ids[low_idx], pool.titles[low_idx], pool.years[low_idx])
    return Probe(
        axis_position=position,
        axis_name=names.get(position, f"tag#{position}"),
        spread=float(spread[position]),
        film_high=film_high,
        film_low=film_low,
        question=f"Which would you rather put on tonight — {film_high.title} or {film_low.title}?",
    )


# -- stopping rules ------------------------------------------------------------------


@dataclass(frozen=True)
class StopDecision:
    """Whether to stop asking, and why. ``reason`` is 'continue' when we keep going."""

    stop: bool
    reason: str


def wants_to_stop(text: str) -> bool:
    """True when the user's words are asking for the escape hatch ("just show me something")."""
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    return any(pattern in lowered for pattern in _STOP_PATTERNS)


def should_stop(
    *,
    turn: int,
    top5_history: list[list[int]],
    top_scores: list[float] | None = None,
    user_requested: bool = False,
    max_turns: int = MAX_TURNS,
    min_turns: int = MIN_TURNS,
    stable_turns: int = STABLE_TURNS,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> StopDecision:
    """Decide whether to stop, at the first rule that fires (plan.md §5.3).

    The user's request wins immediately. Otherwise we honour a hard turn cap, then — once past a
    small minimum so we don't stop after one answer — stop when the top five has held steady across
    the last ``stable_turns`` turns, or when rank 1 leads rank 10 by more than ``margin_threshold``.
    """
    if user_requested:
        return StopDecision(True, "user_request")
    if turn >= max_turns:
        return StopDecision(True, "max_turns")
    if turn < min_turns:
        return StopDecision(False, "continue")

    recent = top5_history[-stable_turns:]
    if len(recent) == stable_turns and all(snapshot == recent[0] for snapshot in recent):
        return StopDecision(True, "top5_stable")

    if top_scores is not None and len(top_scores) >= 10:
        if top_scores[0] - top_scores[9] >= margin_threshold:
            return StopDecision(True, "margin")

    return StopDecision(False, "continue")
