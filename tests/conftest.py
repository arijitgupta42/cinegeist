"""Shared test fixtures.

The catalog tests run against a tiny synthetic ``ml-latest.zip`` — four films, three genome
tags, dense scores for two of them — so the whole MovieLens ingest and genome build can be
exercised offline in milliseconds. Film 999 appears in the scores but not in ``movies.csv`` to
prove the build drops scores for unknown films; film 3 has no scores and no year to prove
un-vectored, year-less films survive ingest; film 4 shares film 1's tmdbId, mirroring the real
dataset where several movieIds point at the same TMDB film, so ingest can't assume uniqueness.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

MOVIES_CSV = (
    "movieId,title,genres\n"
    "1,Toy Story (1995),Adventure|Animation|Children\n"
    "2,Solaris (1972),Drama|Sci-Fi\n"
    "3,Untitled Short,Documentary\n"
    "4,Story of Toys (1995),Animation\n"
)

# Film 4 reuses film 1's imdb/tmdb ids on purpose — real links.csv does this.
LINKS_CSV = "movieId,imdbId,tmdbId\n1,0114709,862\n2,0069293,593\n3,,\n4,0114709,862\n"

GENOME_TAGS_CSV = "tagId,tag\n1,animation\n2,cerebral\n3,space\n"

# Dense scores for films 1 and 2 over all three tags, sorted by movieId then tagId. Film 999
# is not in movies.csv and must be ignored by the build.
GENOME_SCORES_CSV = (
    "movieId,tagId,relevance\n"
    "1,1,0.900\n1,2,0.100\n1,3,0.200\n"
    "2,1,0.050\n2,2,0.950\n2,3,0.800\n"
    "999,1,0.500\n999,2,0.500\n999,3,0.500\n"
)

_MEMBERS = {
    "movies.csv": MOVIES_CSV,
    "links.csv": LINKS_CSV,
    "genome-tags.csv": GENOME_TAGS_CSV,
    "genome-scores.csv": GENOME_SCORES_CSV,
}


def write_movielens_archive(path: Path, *, prefix: str = "ml-latest/") -> Path:
    """Write the synthetic dataset to ``path`` as a zip, nested under ``prefix``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in _MEMBERS.items():
            archive.writestr(prefix + name, content)
    return path


@pytest.fixture
def movielens_archive(tmp_path: Path) -> Path:
    """A ready-made synthetic ``ml-latest.zip`` in a temp directory."""
    return write_movielens_archive(tmp_path / "ml-latest.zip")


@pytest.fixture
def make_movielens_archive():
    """The archive writer itself, for tests that need it at a specific path."""
    return write_movielens_archive
