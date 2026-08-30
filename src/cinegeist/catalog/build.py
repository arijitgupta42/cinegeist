"""The catalog build pipeline.

``build_catalog`` runs the stages in order — download the MovieLens archive, ingest the
movie and link tables, then build the tag-genome memmap — and TMDB enrichment slots in after
these in a later PR. Every stage records its completion in the ``build_state`` scratchpad and
is skipped when already done, so ``make catalog`` is resumable after a Ctrl-C: the expensive
download resumes mid-file, and a finished ingest or genome build is not repeated.

Pass ``force=True`` to redo the ingest and genome stages from an archive already on disk
(the download itself is reused if it verifies, since re-fetching a few hundred MB to rebuild
a table is rarely what you want).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

from ..config import data_dir as default_data_dir
from . import genome
from .db import get_state, open_catalog, set_state
from .sources import movielens

# build_state keys marking each stage complete (value is an ISO timestamp).
_INGESTED_KEY = "movielens_ingested_at"
_GENOME_KEY = "genome_built_at"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_catalog(
    data_dir: Path | None = None,
    *,
    force: bool = False,
    url: str = movielens.DEFAULT_URL,
    console: Console | None = None,
) -> None:
    """Build the catalog into ``data_dir`` (``data/`` by default)."""
    console = console or Console()
    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    conn = open_catalog(data_dir / "cinegeist.db")
    try:
        archive = _stage_download(data_dir, url, console)
        _stage_ingest(conn, archive, console, force=force)
        _stage_genome(conn, archive, data_dir, console, force=force)
        console.print("[green]Catalog build complete.[/green]")
    finally:
        conn.close()


def _stage_download(data_dir: Path, url: str, console: Console) -> Path:
    console.print("[bold]Downloading the MovieLens dataset[/bold] (reused if already present)")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("ml-latest.zip", total=None, start=False)

        def on_start(have: int, total: int | None) -> None:
            progress.update(task, total=total, completed=have)
            progress.start_task(task)

        def on_progress(n: int) -> None:
            progress.advance(task, n)

        archive = movielens.download_archive(
            data_dir, url=url, on_start=on_start, on_progress=on_progress
        )
    return archive


def _stage_ingest(
    conn: sqlite3.Connection, archive: Path, console: Console, *, force: bool
) -> None:
    if not force and get_state(conn, _INGESTED_KEY):
        console.print("[dim]Movies and links already ingested; skipping.[/dim]")
        return

    # links.csv is small; hold it in a dict so movies can be inserted with their ids in one pass.
    links = {movie_id: (imdb, tmdb) for movie_id, imdb, tmdb in movielens.iter_links(archive)}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting movies and links", total=None)
        count = 0
        with conn:  # one transaction for the whole ingest
            for movie_id, title, clean_title, year, _genres in movielens.iter_movies(archive):
                imdb_id, tmdb_id = links.get(movie_id, (None, None))
                conn.execute(
                    """
                    INSERT INTO movies (movie_id, imdb_id, tmdb_id, title, clean_title, year)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (movie_id) DO UPDATE SET
                        imdb_id = excluded.imdb_id,
                        tmdb_id = excluded.tmdb_id,
                        title = excluded.title,
                        clean_title = excluded.clean_title,
                        year = excluded.year
                    """,
                    (movie_id, imdb_id, tmdb_id, title, clean_title, year),
                )
                count += 1
                if count % 5000 == 0:
                    progress.update(task, description=f"Ingesting movies and links ({count:,})")
        progress.update(task, description=f"Ingested {count:,} movies and links")

    set_state(conn, _INGESTED_KEY, _now())
    console.print(f"[green]Ingested {count:,} movies.[/green]")


def _stage_genome(
    conn: sqlite3.Connection, archive: Path, data_dir: Path, console: Console, *, force: bool
) -> None:
    genome_path = genome.default_genome_path(data_dir)
    if not force and get_state(conn, _GENOME_KEY) and genome_path.exists():
        console.print("[dim]Genome already built; skipping.[/dim]")
        return

    # The tag dictionary, in tag_id order, defines the memmap's columns.
    tags = sorted(movielens.iter_genome_tags(archive))
    positions = {tag_id: index for index, (tag_id, _name) in enumerate(tags)}
    n_tags = len(tags)
    with conn:
        conn.execute("DELETE FROM genome_tags")
        conn.executemany(
            "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
            [(tag_id, positions[tag_id], name) for tag_id, name in tags],
        )
        # A rebuild starts from a clean slate so no stale row indices survive a shrink.
        conn.execute(
            "UPDATE movies SET genome_row = NULL, genome_source = 'none' "
            "WHERE genome_source = 'measured'"
        )

    known_movies = {row[0] for row in conn.execute("SELECT movie_id FROM movies")}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reading the tag genome", total=None)
        rows = 0

        def positioned() -> Iterator[genome.PositionedScore]:
            nonlocal rows
            for movie_id, tag_id, relevance in movielens.iter_genome_scores(archive):
                if movie_id not in known_movies:
                    continue
                position = positions.get(tag_id)
                if position is None:
                    continue
                rows += 1
                if rows % 250_000 == 0:
                    progress.update(task, description=f"Reading the tag genome ({rows:,} scores)")
                yield movie_id, position, relevance

        row_of = genome.build_memmap(genome_path, n_tags, positioned())
        progress.update(task, description=f"Read {rows:,} scores for {len(row_of):,} films")

    with conn:
        conn.executemany(
            "UPDATE movies SET genome_row = ?, genome_source = 'measured' WHERE movie_id = ?",
            [(row, movie_id) for movie_id, row in row_of.items()],
        )
    set_state(conn, "genome_rows", str(len(row_of)))
    set_state(conn, "genome_cols", str(n_tags))
    set_state(conn, "genome_dtype", str(genome.DTYPE.__name__))
    set_state(conn, _GENOME_KEY, _now())
    console.print(f"[green]Built genome.npy: {len(row_of):,} films × {n_tags:,} tags.[/green]")
