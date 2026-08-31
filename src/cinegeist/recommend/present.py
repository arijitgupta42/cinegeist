"""Assemble the whole PRESENT phase into one call (plan.md §6, steps 1–5).

This is the seam between the pure scorer and the messy world it needs — the SQLite catalog, the
decayed profile, the genome memmap, and (online) the LLM. It runs the pipeline end to end: hard
filter the catalog, score and diversify it, let the model reorder the shortlist, take the confident
picks plus a wildcard, and explain each in the user's own evidence. What comes back is display-ready
— films paired with sentences — so the engine and the CLI don't each re-derive it.

The whole thing runs with the LLM removed. Offline (or with no client) the deterministic MMR order
*is* the ranking and explanations come from the grounded template, so ``--offline`` is a real
recommender, not a stub — the same property that lets the browser demo drop the LLM entirely. When
a client is present the phase spends two calls, rerank and explain (the payoff of the conversation,
not one of its asking turns), each of which degrades back to the deterministic path on failure.

Already-seen films are always excluded — a recommendation you've reacted to is not a recommendation
— by folding the event log's seen ids into the hard filter here, so no caller can forget to.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

import numpy as np

from ..llm.client import OpenRouterClient
from ..profile.model import TasteProfile
from . import explain as explain_mod
from . import rerank as rerank_mod
from . import retrieve as retrieve_mod
from . import score as score_mod

# How many of the profile's strongest positive axes count as its "strong tags" — the ones a
# wildcard must share, and the vocabulary explanations draw on.
_STRONG_TAGS = 8
# Top genome tags shown to the rerank model per candidate, so it can reason about each film.
_RERANK_TAGS = 6


@dataclass(frozen=True)
class PresentedFilm:
    """One film ready to show: its score, its explanation, and whether it's the wildcard."""

    film: score_mod.ScoredFilm
    explanation: explain_mod.Explanation
    is_wildcard: bool = False


@dataclass(frozen=True)
class Presentation:
    """The finished PRESENT phase: the confident picks, the wildcard, and how it was produced.

    ``degraded_rerank`` / ``degraded_explain`` record where the LLM fell back to the deterministic
    path (offline sets both). ``pool_size`` is how many films survived the hard filter — the honest
    denominator behind the picks, and what tells the engine when a region is too thin to serve well.
    """

    picks: tuple[PresentedFilm, ...]
    wildcard: PresentedFilm | None
    pool_size: int
    degraded_rerank: bool
    degraded_explain: bool

    @property
    def all_films(self) -> tuple[PresentedFilm, ...]:
        """The picks followed by the wildcard, for callers that show them as one list."""
        return (*self.picks, *((self.wildcard,) if self.wildcard else ()))

    @property
    def is_empty(self) -> bool:
        return not self.picks and self.wildcard is None


def _tag_names(conn: sqlite3.Connection) -> dict[int, str]:
    return {
        row["position"]: row["name"]
        for row in conn.execute("SELECT position, name FROM genome_tags")
    }


def profile_summary(profile: TasteProfile, *, max_per_side: int = 5) -> str:
    """A short plain-text sketch of the taste the rerank model reasons over.

    Names the strongest affinities and aversions, quoting the user where a quote drove the axis, so
    the model weighs their own words. An empty profile says so, which the model reads as "by fit".
    """
    if profile.is_empty or not profile.axes:
        return "No strong taste on record yet — order by overall fit."

    def render(axes) -> str:
        parts = []
        for axis in axes[:max_per_side]:
            if axis.evidence:
                quote = " ".join(axis.evidence.split())[:60]
                parts.append(f'{axis.name} ("{quote}")')
            else:
                parts.append(axis.name)
        return ", ".join(parts)

    lines = []
    if profile.affinities:
        lines.append(f"Drawn toward: {render(profile.affinities)}.")
    if profile.aversions:
        lines.append(f"Pushed away from: {render(profile.aversions)}.")
    return "\n".join(lines)


def _top_tags_by_id(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    films: list[score_mod.ScoredFilm],
    *,
    k: int = _RERANK_TAGS,
) -> dict[int, list[str]]:
    """For each film, its highest-relevance genome tag names — context for the rerank model."""
    names = _tag_names(conn)
    if not names:
        return {}
    out: dict[int, list[str]] = {}
    for film in films:
        row = np.asarray(matrix[film.genome_row], dtype=np.float32)
        top = np.argsort(row)[::-1][:k]
        out[film.movie_id] = [names[int(p)] for p in top if int(p) in names and row[int(p)] > 0.0]
    return out


def _strong_tag_positions(profile: TasteProfile, *, limit: int = _STRONG_TAGS) -> frozenset[int]:
    """The positions of the profile's strongest positive axes (what a wildcard has to share)."""
    return frozenset(axis.position for axis in profile.affinities[:limit])


def present(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    profile: TasteProfile,
    *,
    client: OpenRouterClient | None = None,
    models: tuple[str, ...] = (),
    constraints: retrieve_mod.Constraints | None = None,
    session_vector: np.ndarray | None = None,
    also_exclude: frozenset[int] = frozenset(),
    offline: bool = False,
    n_confident: int = 3,
    shortlist_size: int = 40,
) -> Presentation:
    """Run the whole recommendation pipeline and return display-ready picks with explanations.

    ``offline`` (or a missing ``client``) keeps the LLM out entirely: the MMR order is the ranking
    and explanations come from the template. Online, the shortlist is reranked and the picks
    explained, each degrading back to the deterministic path on any model failure. Already-seen
    films and ``also_exclude`` (anything rejected earlier this session) are filtered out here.
    """
    use_llm = not offline and client is not None and bool(models)

    base = constraints or retrieve_mod.Constraints()
    excluded = base.exclude_ids | retrieve_mod.seen_movie_ids(conn, profile.user_id) | also_exclude
    candidates = retrieve_mod.retrieve(conn, replace(base, exclude_ids=excluded))
    if not candidates:
        return Presentation((), None, 0, degraded_rerank=offline, degraded_explain=offline)

    vectors = score_mod.candidate_vectors(candidates, matrix)
    scored = score_mod.score_pool(
        candidates, vectors, profile.genome_vector, session_vector=session_vector
    )
    shortlist = score_mod.mmr_rank(scored, vectors, k=shortlist_size)

    # Order the shortlist: the model reranks it online, otherwise MMR order stands.
    degraded_rerank = True
    if use_llm and len(shortlist) > 1:
        outcome = rerank_mod.rerank(
            client,
            shortlist,
            models,
            profile_summary=profile_summary(profile),
            tags_by_id=_top_tags_by_id(conn, matrix, shortlist),
        )
        shortlist = outcome.ordered
        degraded_rerank = outcome.degraded

    picks = list(shortlist[:n_confident])
    picked_ids = frozenset(f.movie_id for f in picks)
    wildcard = score_mod.select_wildcard(
        scored, vectors, _strong_tag_positions(profile), exclude_movie_ids=picked_ids
    )

    ranked = picks + ([wildcard] if wildcard else [])
    pick_vectors = {f.movie_id: np.asarray(matrix[f.genome_row], dtype=np.float32) for f in ranked}
    evidence = explain_mod.evidence_for_picks(
        profile, ranked, pick_vectors, wildcard_id=wildcard.movie_id if wildcard else None
    )

    degraded_explain = True
    if use_llm and evidence:
        explanations = explain_mod.explain(client, evidence, models)
    else:
        explanations = explain_mod.templated_explanations(evidence)
    if use_llm and evidence:
        degraded_explain = all(e.templated for e in explanations.values())

    presented_picks = tuple(
        PresentedFilm(film=film, explanation=explanations[film.movie_id], is_wildcard=False)
        for film in picks
    )
    presented_wildcard = (
        PresentedFilm(film=wildcard, explanation=explanations[wildcard.movie_id], is_wildcard=True)
        if wildcard
        else None
    )
    return Presentation(
        picks=presented_picks,
        wildcard=presented_wildcard,
        pool_size=len(candidates),
        degraded_rerank=degraded_rerank,
        degraded_explain=degraded_explain,
    )
