"""Hard-filter the catalog into a candidate pool (plan.md §6, step 1).

Before any taste maths runs we cut the catalog down to the films worth scoring at all: the ones
we have a genome vector for, minus everything the constraints rule out — films the user has
already reacted to, anything past a runtime ceiling or outside a release window, the wrong
language, or (when asked) not streamable in their region. This is a *hard* filter: a film that
fails is gone, not down-weighted. The soft preference maths happens afterwards, in :mod:`score`.

Two rules keep an un-enriched catalog usable. First, only genome-covered films are candidates —
without a vector there is nothing to score against. Second, a filter on a TMDB column that is
still ``NULL`` (runtime, language, providers on a genome-only build) never rejects the film: we
don't know it violates the constraint, so we keep it rather than silently emptying the pool. The
one thing we always know is the event log, so excluding already-seen films always works.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# Movie-subject event kinds: every reaction that is *about a film*. Their subjects are the films
# the user has already told us about, so they should never come back as a recommendation.
_MOVIE_EVENT_KINDS = ("liked_movie", "disliked_movie", "pair_choice", "post_watch_feedback")


@dataclass(frozen=True)
class Candidate:
    """One film that survived the hard filter, carrying just the columns scoring needs.

    ``genome_row`` indexes the genome memmap (never ``None`` here — a candidate always has a
    vector). The TMDB columns are ``None`` on an un-enriched catalog; :mod:`score` treats each as
    a neutral prior rather than a zero, so a genome-only build still ranks sensibly.
    """

    movie_id: int
    genome_row: int
    title: str
    year: int | None
    runtime: int | None
    language: str | None
    vote_average: float | None
    vote_count: int | None
    popularity: float | None
    genome_source: str


@dataclass(frozen=True)
class Constraints:
    """The hard filter's inputs — every field optional, an unset field constrains nothing.

    These are tonight's session intent (plan.md §4.2), not long-term taste: a runtime ceiling, a
    release window, a language allow-set ("no subtitles" → ``languages={'en'}``), and whether to
    require regional availability. ``exclude_ids`` carries the already-seen films (see
    :func:`seen_movie_ids`) plus anything rejected earlier in the session.
    """

    exclude_ids: frozenset[int] = frozenset()
    max_runtime: int | None = None
    min_year: int | None = None
    max_year: int | None = None
    languages: frozenset[str] | None = None
    region: str | None = None
    require_available: bool = False

    @property
    def is_empty(self) -> bool:
        """True when nothing here narrows the pool beyond the always-on genome/seen filters."""
        return not (
            self.exclude_ids
            or self.max_runtime is not None
            or self.min_year is not None
            or self.max_year is not None
            or self.languages
            or self.require_available
        )


def seen_movie_ids(conn: sqlite3.Connection, user_id: str = "default") -> frozenset[int]:
    """Every film the user has already reacted to, so we don't recommend it back to them.

    Reads the event log rather than any derived view — a film they loved, bounced off, chose in a
    pair, or gave post-watch feedback on is a film they've seen. Axis and constraint events have
    no movie subject and are ignored.
    """
    placeholders = ",".join("?" for _ in _MOVIE_EVENT_KINDS)
    rows = conn.execute(
        f"SELECT DISTINCT subject FROM preference_events "
        f"WHERE user_id = ? AND subject_kind = 'movie' AND kind IN ({placeholders})",
        (user_id, *_MOVIE_EVENT_KINDS),
    ).fetchall()
    ids: set[int] = set()
    for row in rows:
        try:
            ids.add(int(row["subject"]))
        except (TypeError, ValueError):
            continue  # a malformed subject is not a real movie id; skip it rather than crash
    return frozenset(ids)


def _available_ids(conn: sqlite3.Connection, region: str) -> set[int] | None:
    """Movie ids streamable in ``region`` (any monetization), or ``None`` if we have no data.

    Returning ``None`` (rather than an empty set) when the provider table is empty is what stops
    ``require_available`` from wiping out a catalog that simply hasn't had providers enriched:
    the caller reads ``None`` as "can't tell" and keeps every film.
    """
    total = conn.execute("SELECT COUNT(*) FROM movie_watch_providers").fetchone()[0]
    if not total:
        return None
    rows = conn.execute(
        "SELECT DISTINCT movie_id FROM movie_watch_providers WHERE region = ?", (region,)
    ).fetchall()
    return {row["movie_id"] for row in rows}


def retrieve(
    conn: sqlite3.Connection,
    constraints: Constraints | None = None,
) -> list[Candidate]:
    """Return the genome-covered films that pass ``constraints``, ready for scoring.

    Every returned film has a genome vector and violates none of the *known* constraints. A
    constraint on a column we haven't enriched yet (a ``NULL`` runtime or language) can't be
    checked, so it doesn't reject the film — see the module docstring for why that matters.
    """
    constraints = constraints or Constraints()
    available = (
        _available_ids(conn, constraints.region)
        if constraints.require_available and constraints.region
        else None
    )

    rows = conn.execute(
        "SELECT movie_id, genome_row, clean_title, title, year, runtime, original_language, "
        "vote_average, vote_count, popularity, genome_source "
        "FROM movies WHERE genome_row IS NOT NULL ORDER BY genome_row"
    ).fetchall()

    pool: list[Candidate] = []
    for row in rows:
        movie_id = row["movie_id"]
        if movie_id in constraints.exclude_ids:
            continue

        year = row["year"]
        if year is not None:
            if constraints.min_year is not None and year < constraints.min_year:
                continue
            if constraints.max_year is not None and year > constraints.max_year:
                continue

        runtime = row["runtime"]
        if runtime is not None and constraints.max_runtime is not None:
            if runtime > constraints.max_runtime:
                continue

        language = row["original_language"]
        if language is not None and constraints.languages and language not in constraints.languages:
            continue

        if available is not None and movie_id not in available:
            continue

        pool.append(
            Candidate(
                movie_id=movie_id,
                genome_row=row["genome_row"],
                title=row["clean_title"] or row["title"],
                year=year,
                runtime=runtime,
                language=language,
                vote_average=row["vote_average"],
                vote_count=row["vote_count"],
                popularity=row["popularity"],
                genome_source=row["genome_source"],
            )
        )
    return pool
