"""Read (and download) the MovieLens ``ml-latest`` dataset.

The dataset ships as one zip: ``movies.csv`` (id, title, genres), ``links.csv``
(movieId → imdbId/tmdbId), and the tag genome (``genome-tags.csv`` +
``genome-scores.csv``). The genome is the whole point — a dense relevance matrix over
~1,100 tags — so it is large (~14M score rows) and the download is a few hundred MB.

Two responsibilities live here and nothing else: fetching the archive (resumably, with a
progress callback), and streaming its members as plain Python rows. Everything downstream —
what goes in SQLite, how the memmap is written — belongs to ``build.py`` and ``genome.py``.

Nothing here assumes an extraction step: members are read straight from the zip so we never
unpack a 400 MB CSV to disk. Member names are matched by basename because the archive nests
everything under an ``ml-latest/`` directory whose exact name tracks the dataset version.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

# The dataset home. ml-latest regenerates periodically (so it is never pinned to a version),
# which is exactly why we discover the genome's shape at build time instead of hardcoding it.
DEFAULT_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
ARCHIVE_NAME = "ml-latest.zip"

# The members we read, by basename. If a future dataset drops one of these, verification
# fails loudly rather than producing a half-built catalog.
REQUIRED_MEMBERS = ("movies.csv", "links.csv", "genome-tags.csv", "genome-scores.csv")

# CSV can carry very long free-text fields (titles with commas, etc.); lift the limit.
csv.field_size_limit(1 << 24)

# Trailing "(YYYY)" in a MovieLens title, e.g. "Amélie (2001)" or "Star Wars (1977)".
_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")

# One-argument progress callback: called with the number of bytes or rows just processed.
ProgressFn = Callable[[int], None]


def parse_title(raw: str) -> tuple[str, int | None]:
    """Split a MovieLens title into a clean title and its year.

    ``"Toy Story (1995)"`` → ``("Toy Story", 1995)``. Titles without a trailing year (a few
    exist) come back with the original text and ``None``.
    """
    match = _YEAR_RE.search(raw)
    if not match:
        return raw.strip(), None
    year = int(match.group(1))
    clean = raw[: match.start()].strip()
    return clean, year


def _find_member(archive: zipfile.ZipFile, basename: str) -> str | None:
    """Return the archive member whose path ends in ``basename``, or None."""
    for name in archive.namelist():
        if name == basename or name.endswith("/" + basename):
            return name
    return None


def verify_archive(path: Path) -> bool:
    """True if ``path`` is a readable zip containing every required member.

    A truncated download fails here: a zip's central directory sits at the end of the file,
    so an incomplete file raises :class:`zipfile.BadZipFile` when opened — a cheap, reliable
    truncation check that never has to decompress the giant score file.
    """
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return all(_find_member(archive, name) is not None for name in REQUIRED_MEMBERS)
    except zipfile.BadZipFile:
        return False


def _iter_member_rows(path: Path, basename: str) -> Iterator[list[str]]:
    """Yield CSV rows (excluding the header) from a member, read as a stream."""
    with zipfile.ZipFile(path) as archive:
        name = _find_member(archive, basename)
        if name is None:
            raise FileNotFoundError(f"{basename} is not in {path.name}")
        with archive.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text)
            next(reader, None)  # drop the header row
            yield from reader


def iter_movies(path: Path) -> Iterator[tuple[int, str, str, int | None, str]]:
    """Yield ``(movie_id, title, clean_title, year, genres)`` from ``movies.csv``.

    ``genres`` is MovieLens' own pipe-separated string; we pass it through untouched (the
    build currently ignores it in favour of the far richer tag genome and TMDB genres).
    """
    for row in _iter_member_rows(path, "movies.csv"):
        if len(row) < 3:
            continue
        movie_id, title, genres = int(row[0]), row[1], row[2]
        clean_title, year = parse_title(title)
        yield movie_id, title, clean_title, year, genres


def iter_links(path: Path) -> Iterator[tuple[int, str | None, int | None]]:
    """Yield ``(movie_id, imdb_id, tmdb_id)`` from ``links.csv``.

    imdbId arrives as bare digits; we restore the canonical ``tt`` prefix and zero-pad to at
    least seven digits. A blank tmdbId (a handful exist) becomes ``None``.
    """
    for row in _iter_member_rows(path, "links.csv"):
        if len(row) < 3:
            continue
        movie_id = int(row[0])
        imdb_raw = row[1].strip()
        tmdb_raw = row[2].strip()
        imdb_id = f"tt{int(imdb_raw):07d}" if imdb_raw else None
        tmdb_id = int(tmdb_raw) if tmdb_raw else None
        yield movie_id, imdb_id, tmdb_id


def iter_genome_tags(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(tag_id, name)`` from ``genome-tags.csv``."""
    for row in _iter_member_rows(path, "genome-tags.csv"):
        if len(row) < 2:
            continue
        yield int(row[0]), row[1]


def iter_genome_scores(path: Path) -> Iterator[tuple[int, int, float]]:
    """Yield ``(movie_id, tag_id, relevance)`` from ``genome-scores.csv``.

    This is the ~14M-row file; it is streamed, never materialised. Rows arrive sorted by
    movieId then tagId, but nothing downstream depends on that order.
    """
    for row in _iter_member_rows(path, "genome-scores.csv"):
        if len(row) < 3:
            continue
        yield int(row[0]), int(row[1]), float(row[2])


def download_archive(
    dest_dir: Path,
    *,
    url: str = DEFAULT_URL,
    client: httpx.Client | None = None,
    on_start: Callable[[int, int | None], None] | None = None,
    on_progress: ProgressFn | None = None,
    chunk_size: int = 1 << 20,
) -> Path:
    """Download ``ml-latest.zip`` into ``dest_dir``, resuming a partial download.

    A complete, valid archive already on disk is reused untouched. An interrupted download
    leaves a ``.part`` file; the next call sends a ``Range`` request and appends to it. Only
    once the finished file verifies does it get its final name, so a half-file is never
    mistaken for a good one.

    ``on_start`` is called once with ``(already_have_bytes, total_bytes_or_None)`` when the
    transfer size is known, so a caller can size a progress bar; ``on_progress`` then receives
    the byte count of each chunk written.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ARCHIVE_NAME
    if verify_archive(dest):
        return dest

    part = dest.with_name(dest.name + ".part")
    owns_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(30.0, read=300.0), follow_redirects=True)
    try:
        _stream_to_part(
            client, url, part, on_start=on_start, on_progress=on_progress, chunk_size=chunk_size
        )
    finally:
        if owns_client:
            client.close()

    part.replace(dest)
    if not verify_archive(dest):
        dest.unlink(missing_ok=True)
        raise OSError(f"Downloaded archive at {dest} failed verification (corrupt or truncated).")
    return dest


def _stream_to_part(
    client: httpx.Client,
    url: str,
    part: Path,
    *,
    on_start: Callable[[int, int | None], None] | None,
    on_progress: ProgressFn | None,
    chunk_size: int,
) -> None:
    """Fetch ``url`` into ``part``, resuming from its current size when possible."""
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with client.stream("GET", url, headers=headers) as response:
        # If we asked to resume but the server sent the whole file (200, not 206), start over.
        if resume_from and response.status_code == 200:
            resume_from = 0
        response.raise_for_status()
        if on_start is not None:
            length = response.headers.get("Content-Length")
            total = resume_from + int(length) if length is not None else None
            on_start(resume_from, total)
        mode = "ab" if resume_from else "wb"
        with open(part, mode) as handle:
            for block in response.iter_bytes(chunk_size):
                handle.write(block)
                if on_progress is not None:
                    on_progress(len(block))
