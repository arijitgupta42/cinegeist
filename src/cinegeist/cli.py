"""Command line entry point for cinegeist.

Session 1 ships three commands that exercise the whole LLM plumbing end to end:

    cinegeist models --free    list the models that are free right now
    cinegeist ask "hi"         one-shot chat through a free model (needs a key)
    cinegeist config           show the effective settings (key redacted)

The heavier commands (chat, search, profile, ...) arrive in later sessions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import API_KEY_ENV, cache_dir, config_path, data_dir, load_settings
from .llm.client import LLMError, OpenRouterClient
from .llm.registry import free_models

# Attribution required by the data providers; keep it in the CLI footer (see CLAUDE.md).
_EPILOG = "This product uses the TMDB API but is not endorsed or certified by TMDB."

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A conversational movie recommender for people who can't say what they like.",
    epilog=_EPILOG,
)

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"cinegeist {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """React to films, not questionnaires."""


@app.command()
def models(
    free: bool = typer.Option(
        True, "--free", help="List the models that are currently free (the default)."
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore the 24h cache and re-fetch the list."
    ),
) -> None:
    """List currently-free OpenRouter models, best first."""
    settings = load_settings()
    ids = free_models(settings, force_refresh=refresh)
    if not ids:
        err_console.print("[yellow]No free models found.[/yellow]")
        raise typer.Exit(1)
    console.print(f"[bold]{len(ids)} free model(s), best first:[/bold]")
    for rank, model_id in enumerate(ids, start=1):
        console.print(f"  {rank:>2}. {model_id}")


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="What to say to the model."),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model id to use (overrides config and auto-select)."
    ),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="Sampling temperature."
    ),
) -> None:
    """Send one message to a model and print the reply."""
    settings = load_settings(overrides={"model": model})

    if not settings.has_api_key:
        err_console.print(
            f"[red]{API_KEY_ENV} is not set.[/red] Get a key at "
            "https://openrouter.ai/keys and export it, or copy .env.example to .env."
        )
        raise typer.Exit(1)

    # Pin the configured model if there is one; otherwise fail over across the free list.
    candidates = [settings.model] if settings.model else free_models(settings)
    messages = [{"role": "user", "content": prompt}]

    try:
        with OpenRouterClient(settings) as client:
            result = client.chat_with_failover(messages, candidates, temperature=temperature)
    except LLMError as error:
        err_console.print(f"[red]LLM request failed:[/red] {error}")
        raise typer.Exit(1) from error

    console.print(result.text)
    console.print(f"[dim]— {result.model}[/dim]")


catalog_app = typer.Typer(
    no_args_is_help=True,
    help="Build and maintain the local movie catalog (SQLite + the genome memmap).",
)
app.add_typer(catalog_app, name="catalog")


@catalog_app.command("build")
def catalog_build(
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild the ingest and genome stages from the downloaded archive.",
    ),
    region: str = typer.Option(
        "US", "--region", help="Region for TMDB watch-provider availability."
    ),
    skip_enrich: bool = typer.Option(
        False, "--skip-enrich", help="Skip the TMDB enrichment stage (genome only)."
    ),
    data: str | None = typer.Option(
        None,
        "--data-dir",
        help="Where to write cinegeist.db and genome.npy (defaults to ./data).",
    ),
) -> None:
    """Download MovieLens and build data/cinegeist.db and data/genome.npy.

    Resumable: a partial download continues, and finished stages are skipped. The first run
    fetches a few hundred MB and processes the tag genome, so it takes a while. TMDB
    enrichment runs last when a TMDB credential is set (see `cinegeist config`).
    """
    # Imported lazily so the light commands (models, config) don't pay numpy's import cost.
    from .catalog.build import build_catalog

    try:
        build_catalog(
            data_dir=Path(data) if data else None,
            force=force,
            enrich=not skip_enrich,
            tmdb_region=region,
            console=console,
        )
    except (OSError, ValueError) as error:
        err_console.print(f"[red]Catalog build failed:[/red] {error}")
        raise typer.Exit(1) from error


@catalog_app.command("enrich")
def catalog_enrich(
    scope: str = typer.Option(
        "measured",
        "--scope",
        help="Which films to enrich: 'measured' (genome-covered) or 'all'.",
    ),
    region: str = typer.Option(
        "US", "--region", help="Region for TMDB watch-provider availability."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Enrich at most this many films (for a partial run)."
    ),
    concurrency: int = typer.Option(
        16, "--concurrency", help="Number of concurrent TMDB requests."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-fetch films even if they already have TMDB data."
    ),
    data: str | None = typer.Option(
        None, "--data-dir", help="Catalog location (defaults to ./data)."
    ),
) -> None:
    """Fetch keywords, credits, countries, and providers from TMDB. Resumable and concurrent."""
    from .catalog.db import open_catalog
    from .catalog.sources import tmdb

    settings = load_settings()
    conn = open_catalog(Path(data) / "cinegeist.db" if data else None)
    try:
        tmdb.enrich_catalog(
            conn,
            settings,
            scope=scope,
            region=region,
            limit=limit,
            force=force,
            concurrency=concurrency,
            console=console,
        )
    except tmdb.TMDBAuthError as error:
        err_console.print(
            f"[red]{error}[/red] Get a key at https://www.themoviedb.org/settings/api "
            "and export TMDB_API_KEY."
        )
        raise typer.Exit(1) from error
    finally:
        conn.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Descriptive words, e.g. 'bleak cerebral 90s'."),
    limit: int = typer.Option(10, "--limit", "-n", help="How many films to show."),
    data: str | None = typer.Option(
        None, "--data-dir", help="Catalog location (defaults to ./data)."
    ),
) -> None:
    """Search the catalog by tag genome — a deterministic debug view of retrieval.

    Ranks genome-covered films by cosine similarity to the tags your phrase names, narrowed by
    a decade or year if you mention one. No LLM, no profile: just the maths.
    """
    from .catalog import genome as genome_mod
    from .catalog import search as search_mod
    from .catalog.db import open_catalog

    base = Path(data) if data else data_dir()
    genome_path = genome_mod.default_genome_path(base)
    if not genome_path.exists():
        err_console.print(
            "[red]No catalog found.[/red] Build one first with [bold]make catalog[/bold] "
            "(or `cinegeist catalog build`)."
        )
        raise typer.Exit(1)

    conn = open_catalog(base / "cinegeist.db")
    try:
        matrix = genome_mod.load_genome(genome_path)
        try:
            result = search_mod.search(conn, matrix, query, limit=limit)
        except search_mod.NoTagsMatched as error:
            err_console.print(
                f"[yellow]{error}[/yellow] Try descriptive words like "
                "'atmospheric', 'bleak', 'nonlinear', 'twist ending'."
            )
            raise typer.Exit(1) from error
    finally:
        conn.close()

    tag_line = ", ".join(result.tags)
    if result.year_range is not None:
        low, high = result.year_range
        tag_line += f"  ·  years {low}" + (f"–{high}" if high != low else "")
    console.print(f"[dim]tags:[/dim] {tag_line}")

    if not result.hits:
        console.print("[yellow]Nothing in that range.[/yellow]")
        return

    table = Table(box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("title")
    table.add_column("year", justify="right", style="dim")
    table.add_column("score", justify="right")
    table.add_column("matched tags", style="cyan")
    for rank, hit in enumerate(result.hits, start=1):
        matched = ", ".join(f"{name} {value:.2f}" for name, value in hit.matched)
        table.add_row(
            str(rank),
            hit.title,
            str(hit.year) if hit.year else "—",
            f"{hit.score:.3f}",
            matched,
        )
    console.print(table)


@app.command()
def config() -> None:
    """Show the effective settings. The API key is never printed, only whether it is set."""
    settings = load_settings()
    path = config_path()

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("model", settings.model or "auto (pick a free one)")
    table.add_row("api_base_url", settings.api_base_url)
    table.add_row("request_timeout", f"{settings.request_timeout:g}s")
    table.add_row("max_retries", str(settings.max_retries))
    table.add_row(API_KEY_ENV, "set (redacted)" if settings.has_api_key else "not set")
    table.add_row("TMDB auth", "set (redacted)" if settings.has_tmdb_auth else "not set")
    table.add_row("config file", f"{path} ({'exists' if path.exists() else 'not present'})")
    table.add_row("cache dir", str(cache_dir()))
    table.add_row("data dir", str(data_dir()))
    console.print(table)


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 so Rich's progress glyphs survive Windows.

    The progress spinners use Braille characters. On a legacy Windows code page, or when output
    is redirected through a pipe, Python's stream encoder defaults to something like cp1252 and
    raises UnicodeEncodeError on those glyphs. Reconfiguring to UTF-8 (with a lenient error
    handler as a backstop) keeps the catalog build from dying on cosmetics.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (ValueError, OSError):
                pass


def run() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    _force_utf8_output()
    app()


if __name__ == "__main__":
    app()
