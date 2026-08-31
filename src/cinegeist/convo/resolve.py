"""Resolve a film title the user named to a real catalog entry.

The LLM extracts titles from free text but we never trust one as a catalog fact (CLAUDE.md hard
rule 1). Every named film is matched back against the local ``movies`` table here: case-folded,
article-normalised ("The Matrix" ↔ MovieLens's "Matrix, The"), and fuzzy-scored so a small typo
still lands. When two films fit equally — *Solaris* the 1972 film and the 2002 one — we don't
guess; we return both as an ambiguity for the caller to confirm with the user.

No LLM, no network: a couple of indexed-free but C-fast full scans over the title column plus
:mod:`difflib` ranking. At ~86k films that is a few milliseconds, run a handful of times per turn.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

_ARTICLES = frozenset({"the", "a", "an"})

# A film scores at least this to be considered a real candidate at all.
_MATCH_FLOOR = 0.6
# Two candidates within this score of the best are "equally good" — an ambiguity, not a winner.
_AMBIGUITY_MARGIN = 0.08
# Cap on the fuzzy-ranked candidate set, so a broad LIKE can't blow up difflib work.
_CANDIDATE_CAP = 500
# We LIKE-filter on a short prefix of the key word, not the whole word, so a typo later in it
# ("amelei" for "amelie") still pulls the film into the candidate set for difflib to rank.
_LIKE_PREFIX = 4

_TRAILING_ARTICLE_RE = re.compile(r"^(.*),\s*(the|a|an)$")
_TRAILING_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# MovieLens routinely appends an original-language title in parentheses:
# "Amelie (Fabuleux destin d'Amélie Poulain, Le)", "Solaris (Solyaris)". The leading segment is
# the title people actually type, so it becomes a second comparison key alongside the full string.
_PARENTHETICAL_RE = re.compile(r"\s*\(.*\)\s*")


def _fold_accents(text: str) -> str:
    """Strip diacritics so an accented title matches the plain letters people usually type."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@dataclass(frozen=True)
class Mention:
    """A film the extractor pulled out of free text: its title, maybe a year and the user's words.

    ``sentiment`` is carried through untouched (this module doesn't interpret it); ``evidence`` is
    the verbatim phrase the user used, kept so a resulting event can cite it.
    """

    title: str
    year: int | None = None
    sentiment: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class MovieMatch:
    """One candidate film for a title, with the confidence that it is the one meant."""

    movie_id: int
    title: str
    year: int | None
    score: float


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one title: resolved to a film, ambiguous, or no match.

    ``match`` is set only when ``status == 'resolved'``; ``candidates`` carries the tied films
    when ``status == 'ambiguous'`` (and the near-misses, for context, when ``'no_match'``).
    """

    query: str
    year: int | None
    status: str  # 'resolved' | 'ambiguous' | 'no_match'
    match: MovieMatch | None = None
    candidates: list[MovieMatch] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved"


def _normalize(title: str) -> str:
    """Fold a title to a comparison key: accent-folded, article-normalised, punctuation-stripped."""
    text = _fold_accents(title.strip().lower())
    trailing = _TRAILING_ARTICLE_RE.match(text)
    if trailing:  # "matrix, the" → "the matrix"
        text = f"{trailing.group(2)} {trailing.group(1)}"
    parts = text.split()
    if parts and parts[0] in _ARTICLES:  # drop a leading article entirely
        parts = parts[1:]
    text = _NON_ALNUM_RE.sub(" ", " ".join(parts)).strip()
    return re.sub(r"\s+", " ", text)


def _candidate_keys(clean_title: str) -> set[str]:
    """The comparison keys for a catalog title: the whole thing, and its pre-parenthetical head.

    A foreign film stored as "Amelie (Fabuleux destin …)" should match "amelie" on its head, not
    lose out to "Amelia" because the full string is long.
    """
    keys = {_normalize(clean_title)}
    head = _PARENTHETICAL_RE.sub(" ", clean_title).strip()
    if head:
        keys.add(_normalize(head))
    keys.discard("")
    return keys


def _split_trailing_year(title: str) -> tuple[str, int | None]:
    """Peel a trailing ``(YYYY)`` off a title. A bare number stays part of the name.

    So "Solaris (1972)" yields a year hint, but "Blade Runner 2049" does not — 2049 is the title.
    """
    match = _TRAILING_YEAR_RE.search(title)
    if match:
        return title[: match.start()].strip(), int(match.group(1))
    return title, None


def _pick_token(normalized: str) -> str:
    """The most selective word to filter on: the longest non-article token, else the whole key."""
    tokens = [t for t in normalized.split() if t not in _ARTICLES]
    selective = [t for t in tokens if len(t) >= 3] or tokens
    return max(selective, key=len) if selective else normalized


def _fetch_candidates(conn: sqlite3.Connection, normalized: str) -> list[sqlite3.Row]:
    """Pull a bounded candidate set: exact article-less matches plus anything sharing a key word."""
    seen: dict[int, sqlite3.Row] = {}
    # Exact article-less equality guarantees short titles like "Up" are never lost under the cap.
    for row in conn.execute(
        "SELECT movie_id, clean_title, year FROM movies WHERE lower(clean_title) = ? LIMIT 50",
        (normalized,),
    ):
        seen[row["movie_id"]] = row
    like_key = _pick_token(normalized)[:_LIKE_PREFIX]
    for row in conn.execute(
        "SELECT movie_id, clean_title, year FROM movies "
        "WHERE clean_title IS NOT NULL AND lower(clean_title) LIKE ? LIMIT ?",
        (f"%{like_key}%", _CANDIDATE_CAP),
    ):
        seen.setdefault(row["movie_id"], row)
    return list(seen.values())


def _match_score(query: str, keys: set[str]) -> float:
    """Best similarity of the query key against any of a candidate's keys (exact match wins)."""
    if query in keys:
        return 1.0
    return max((SequenceMatcher(None, query, key).ratio() for key in keys), default=0.0)


def _score_candidates(normalized: str, rows: list[sqlite3.Row]) -> list[MovieMatch]:
    matches: list[MovieMatch] = []
    for row in rows:
        keys = _candidate_keys(row["clean_title"] or "")
        if not keys:
            continue
        matches.append(
            MovieMatch(
                movie_id=row["movie_id"],
                title=row["clean_title"],
                year=row["year"],
                score=_match_score(normalized, keys),
            )
        )
    # Best score first; a newer film breaks an exact score tie only for a stable order.
    matches.sort(key=lambda m: (m.score, m.year or 0), reverse=True)
    return matches


def resolve_title(
    conn: sqlite3.Connection, title: str, year: int | None = None, *, limit: int = 6
) -> Resolution:
    """Resolve one named title against the catalog.

    A ``year`` (from the extractor, or a trailing ``(YYYY)`` in the title) is used only to break a
    tie between films that match the title equally well — never to reject an otherwise clear match.
    """
    stripped, parsed_year = _split_trailing_year(title)
    year = year if year is not None else parsed_year
    normalized = _normalize(stripped)
    if not normalized:
        return Resolution(query=title, year=year, status="no_match")

    scored = _score_candidates(normalized, _fetch_candidates(conn, normalized))
    strong = [m for m in scored if m.score >= _MATCH_FLOOR]
    if not strong:
        return Resolution(query=title, year=year, status="no_match", candidates=scored[:limit])

    # If a year is known and any strong candidate has it, decide within just those.
    pool = strong
    if year is not None:
        year_hits = [m for m in strong if m.year == year]
        if year_hits:
            pool = year_hits

    top = pool[0]
    ties = [m for m in pool if top.score - m.score <= _AMBIGUITY_MARGIN]
    if len(ties) == 1:
        return Resolution(query=title, year=year, status="resolved", match=top)
    return Resolution(query=title, year=year, status="ambiguous", candidates=ties[:limit])


def resolve_mentions(
    conn: sqlite3.Connection, mentions: list[Mention]
) -> list[tuple[Mention, Resolution]]:
    """Resolve a batch of extracted mentions, pairing each with its outcome (order preserved)."""
    return [(mention, resolve_title(conn, mention.title, mention.year)) for mention in mentions]
