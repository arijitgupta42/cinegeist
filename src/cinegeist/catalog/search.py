"""Tag-vector search over the catalog — a debugging window onto retrieval.

`cinegeist search "bleak cerebral 90s"` turns a phrase into a sparse query vector over the tag
genome and ranks films by cosine similarity, narrowing by decade or year when the phrase names
one. It is deliberately simple and fully deterministic: no LLM, no learned profile — just the
maths the recommender is built on, exposed so you can eyeball whether the genome retrieves what
you'd nod at. The real conversational recommender (later sessions) layers a profile and MMR on
top of exactly this cosine core.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import genome

# A 4-digit decade ("1990s", "1970s", "2010s"), then a 2-digit one ("90s", "00s"), then a bare
# year ("1994"). Checked in that order so "1990s" isn't mis-read as the year 1990.
_DECADE4_RE = re.compile(r"\b((?:19|20)\d)0s\b")
_DECADE2_RE = re.compile(r"\b(\d0)s\b")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

YearRange = tuple[int, int]


@dataclass(frozen=True)
class SearchHit:
    """One ranked film, with the query tags that placed it and this film's relevance on each."""

    movie_id: int
    title: str
    year: int | None
    score: float
    matched: list[tuple[str, float]]


@dataclass(frozen=True)
class SearchResult:
    """A search's ranked hits plus what the query parsed into, for a transparent debug view."""

    hits: list[SearchHit]
    tags: list[str]
    year_range: YearRange | None


class NoTagsMatched(ValueError):
    """The query named no tag the genome knows, so there is nothing to rank by."""


def parse_query(text: str) -> tuple[str, YearRange | None]:
    """Split a phrase into its leftover tag words and an optional year range.

    ``"bleak 90s"`` → ``("bleak ", (1990, 1999))``. A 2-digit decade is read as 19xx for the
    30s–90s and 20xx for the 00s–20s, the usual convention for talking about films.
    """
    year_range: YearRange | None = None
    match = _DECADE4_RE.search(text)
    if match:
        start = int(match.group(1) + "0")
        year_range = (start, start + 9)
    elif (match := _DECADE2_RE.search(text)) is not None:
        tens = int(match.group(1))
        century = 1900 if tens >= 30 else 2000
        start = century + tens
        year_range = (start, start + 9)
    elif (match := _YEAR_RE.search(text)) is not None:
        year = int(match.group(1))
        year_range = (year, year)

    leftover = text[: match.start()] + text[match.end() :] if match else text
    return leftover, year_range


def resolve_tags(conn: sqlite3.Connection, text: str) -> list[tuple[int, str]]:
    """Match a phrase against genome tag names, returning ``(position, name)`` pairs.

    Single-word tags match a whole word in the query (so "war" doesn't fire on "warm");
    multi-word tags ("twist ending", "based on a book") match as a phrase substring.
    """
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9']+", lowered))
    matched: list[tuple[int, str]] = []
    for row in conn.execute("SELECT position, name FROM genome_tags"):
        name = row["name"].lower()
        if " " in name:
            if name in lowered:
                matched.append((row["position"], row["name"]))
        elif name in tokens:
            matched.append((row["position"], row["name"]))
    return matched


def search(
    conn: sqlite3.Connection,
    matrix: np.ndarray,
    text: str,
    *,
    limit: int = 10,
) -> SearchResult:
    """Rank genome-covered films by cosine similarity to the query's tags.

    Raises :class:`NoTagsMatched` when the phrase names no known tag, since there is then no
    vector to rank by — the honest failure rather than inventing results.
    """
    leftover, year_range = parse_query(text)
    tags = resolve_tags(conn, leftover)
    if not tags:
        raise NoTagsMatched(f"No known genome tags in {text!r}.")

    query = np.zeros(matrix.shape[1], dtype=genome.DTYPE)
    for position, _name in tags:
        query[position] = 1.0
    scores = genome.cosine_scores(matrix, query)

    candidates = conn.execute(
        "SELECT movie_id, clean_title, title, year, genome_row "
        "FROM movies WHERE genome_row IS NOT NULL"
    ).fetchall()

    ranked = []
    for row in candidates:
        if year_range is not None:
            year = row["year"]
            if year is None or not (year_range[0] <= year <= year_range[1]):
                continue
        ranked.append((float(scores[row["genome_row"]]), row))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    hits = []
    for score, row in ranked[:limit]:
        relevances = [(name, float(matrix[row["genome_row"], pos])) for pos, name in tags]
        relevances.sort(key=lambda pair: pair[1], reverse=True)
        hits.append(
            SearchHit(
                movie_id=row["movie_id"],
                title=row["clean_title"] or row["title"],
                year=row["year"],
                score=score,
                matched=relevances,
            )
        )
    return SearchResult(hits=hits, tags=[name for _pos, name in tags], year_range=year_range)
