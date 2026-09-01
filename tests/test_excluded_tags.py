"""The non-content tag exclusion list and its spec mirror (see catalog/excluded_tags.py)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cinegeist.catalog.excluded_tags import EXCLUDED_TAGS, excluded_positions, is_excluded

SPEC = Path(__file__).resolve().parent.parent / "spec" / "excluded_tags.json"


def test_spec_mirror_matches_the_python_set() -> None:
    # The browser demo reads spec/excluded_tags.json; it must equal the Python source of truth, so
    # the two never disagree about which tags are excluded (regenerate with `make spec`).
    data = json.loads(SPEC.read_text(encoding="utf-8"))
    assert data["excluded_tags"] == sorted(EXCLUDED_TAGS)


def test_is_excluded_covers_reception_and_verdict_tags_only() -> None:
    # Excluded: reception/status and pure quality verdicts.
    assert is_excluded("imdb top 250")
    assert is_excluded("  IMDB Top 250 ")  # case- and space-insensitive
    assert is_excluded("masterpiece")
    assert is_excluded("oscar (best picture)")
    # Kept: content/affect, craft, auteur, and Tier 3 provenance.
    assert not is_excluded("atmospheric")
    assert not is_excluded("funny")
    assert not is_excluded("great cinematography")
    assert not is_excluded("kubrick")
    assert not is_excluded("based on a book")


def test_excluded_positions_maps_names_to_genome_columns() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE genome_tags (tag_id INTEGER, position INTEGER, name TEXT)")
    conn.executemany(
        "INSERT INTO genome_tags VALUES (?, ?, ?)",
        [(1, 0, "atmospheric"), (2, 1, "imdb top 250"), (3, 2, "masterpiece"), (4, 3, "kubrick")],
    )
    assert excluded_positions(conn) == {1, 2}
