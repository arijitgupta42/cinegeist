"""Genome tags that describe a film's *reception, status, or a viewer's verdict* rather than its
character — and so are disingenuous to match taste on. Two films sharing ``imdb top 250`` or
``masterpiece`` are both well-regarded, not alike in any way a viewer picking what to watch cares
about, yet the raw genome would pull them together and cite the tag in the explanation.

These are excluded from the taste space at build time: their columns are zeroed before the web
shard's SVD and top-tag tables are computed (``scripts/build_web_shard.py``), and the browser demo
drops them from the top tags it shows even on the already-built shard. The list is mirrored to
``spec/excluded_tags.json`` (via ``make spec``) so the TypeScript demo reads exactly this set.

Scope is deliberate — this is the maintainer-approved Tier 1 + Tier 2 from the tag audit:
reception/curation/awards/status, and pure quality verdicts. It does **not** touch content or craft
tags (``atmospheric``, ``great cinematography``), auteur/cast tags (``kubrick``), or provenance tags
that carry a real preference (``based on a book``, ``remake``, ``indie``).
"""

from __future__ import annotations

import sqlite3

# Curation lists, awards, and cult/status — a film's standing, not its content.
_RECEPTION = {
    "imdb top 250",
    "criterion",
    "movielens top pick",
    "afi 100 (movie quotes)",
    "oscar",
    "oscar (best actor)",
    "oscar (best actress)",
    "oscar (best animated feature)",
    "oscar (best cinematography)",
    "oscar (best directing)",
    "oscar (best editing)",
    "oscar (best effects - visual effects)",
    "oscar (best foreign language film)",
    "oscar (best music - original score)",
    "oscar (best picture)",
    "oscar (best supporting actor)",
    "oscar (best supporting actress)",
    "oscar (best writing - screenplay written directly for the screen)",
    "oscar winner",
    "saturn award (best special effects)",
    "golden palm",
    "cult",
    "cult film",
    "cult classic",
}

# Subjective quality verdicts — how good/bad someone found it, applicable to any film. Content and
# affect words (funny, scary, bleak) and craft tags (great cinematography) are deliberately absent.
_VERDICTS = {
    "masterpiece",
    "great",
    "great movie",
    "good",
    "bad",
    "awful",
    "horrible",
    "excellent",
    "boring",
    "boring!",
    "lame",
    "dumb",
    "dumb but funny",
    "stupid",
    "stupid as hell",
    "stupidity",
    "idiotic",
    "cheesy",
    "pointless",
    "shallow",
    "ridiculous",
    "sappy",
    "overrated",
    "underrated",
    "predictable",
    "guilty pleasure",
    "so bad it's good",
    "so bad it's funny",
    "not funny",
    "too long",
    "too short",
    "very good",
    "very funny",
    "very interesting",
    "better than expected",
    "better than the american version",
    "disappointing",
    "entertaining",
    "fun",
    "fun movie",
    "cool",
    "interesting",
    "classic",
    "highly quotable",
    "quotable",
    "funniest movies",
    "best war films",
    "suprisingly clever",
    "unlikeable characters",
    "unrealistic",
    "bad acting",
    "bad plot",
    "bad script",
    "bad ending",
    "bad cgi",
    "bad science",
    "plot holes",
}

# The full set, lowercased. Membership tests should lowercase and strip the candidate name.
EXCLUDED_TAGS: frozenset[str] = frozenset(_RECEPTION | _VERDICTS)


def is_excluded(name: str) -> bool:
    """Whether a genome tag name is a non-content (reception/verdict) tag."""
    return name.strip().lower() in EXCLUDED_TAGS


def excluded_positions(conn: sqlite3.Connection) -> set[int]:
    """The genome column positions of the excluded tags present in this catalog."""
    return {
        row["position"]
        for row in conn.execute("SELECT position, name FROM genome_tags")
        if is_excluded(row["name"])
    }
