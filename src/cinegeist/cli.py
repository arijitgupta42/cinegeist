"""Command line entry point for cinegeist.

Session 1 ships three commands that exercise the whole LLM plumbing end to end:

    cinegeist models --free    list the models that are free right now
    cinegeist ask "hi"         one-shot chat through a free model (needs a key)
    cinegeist config           show the effective settings (key redacted)

The heavier commands (chat, search, profile, ...) arrive in later sessions.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import API_KEY_ENV, cache_dir, config_path, load_settings
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
    data: str | None = typer.Option(
        None,
        "--data-dir",
        help="Where to write cinegeist.db and genome.npy (defaults to ./data).",
    ),
) -> None:
    """Download MovieLens and build data/cinegeist.db and data/genome.npy.

    Resumable: a partial download continues, and finished stages are skipped. The first run
    fetches a few hundred MB and processes the tag genome, so it takes a while.
    """
    # Imported lazily so the light commands (models, config) don't pay numpy's import cost.
    from .catalog.build import build_catalog

    try:
        build_catalog(data_dir=Path(data) if data else None, force=force, console=console)
    except (OSError, ValueError) as error:
        err_console.print(f"[red]Catalog build failed:[/red] {error}")
        raise typer.Exit(1) from error


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
    table.add_row("config file", f"{path} ({'exists' if path.exists() else 'not present'})")
    table.add_row("cache dir", str(cache_dir()))
    console.print(table)


def run() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    app()
