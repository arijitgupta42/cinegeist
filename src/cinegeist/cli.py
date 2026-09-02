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
from rich.prompt import Prompt
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


class _RichConsoleIO:
    """The engine's :class:`ConversationIO` for a real terminal, built on ``rich``.

    Kept structural (no import of the engine's types) so the light CLI commands don't pay numpy's
    import cost. Choices are numbered; typing a number picks it, and empty input takes the first —
    the escape hatch is always the last option, so ``[enter][enter]…`` never traps the user.
    """

    def __init__(self, out: Console) -> None:
        self._console = out

    def say(self, text: str) -> None:
        self._console.print(f"\n{text}")

    def ask_text(self, prompt: str) -> str:
        self._console.print(f"\n[bold]{prompt}[/bold]")
        return Prompt.ask("[cyan]»[/cyan]", console=self._console, default="").strip()

    def ask_choice(self, prompt: str, options) -> str:
        self._console.print(f"\n[bold]{prompt}[/bold]")
        for index, option in enumerate(options, start=1):
            self._console.print(f"  [cyan]{index}[/cyan]. {option.label}")
        while True:
            raw = Prompt.ask("[cyan]»[/cyan]", console=self._console, default="1").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1].key
            self._console.print("[dim]Pick one of the numbers.[/dim]")

    def show_presentation(self, presentation, *, header: str) -> None:
        self._console.print(f"\n[bold]{header}[/bold]")
        position = 1
        for pick in presentation.picks:
            self._render_film(position, pick)
            position += 1
        if presentation.wildcard is not None:
            self._render_film(position, presentation.wildcard)

    def _render_film(self, index: int, presented) -> None:
        film = presented.film
        year = f" [dim]({film.year})[/dim]" if film.year else ""
        flag = " [magenta]· wildcard[/magenta]" if presented.is_wildcard else ""
        self._console.print(f"[bold]{index}. {film.title}[/bold]{year}{flag}")
        self._console.print(f"   [dim]{presented.explanation.text}[/dim]")


@app.command()
def chat(
    offline: bool = typer.Option(
        False, "--offline", help="No LLM calls: fixed phrasing, click-driven, works without a key."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model id to use (overrides config and auto-select)."
    ),
    data: str | None = typer.Option(
        None, "--data-dir", help="Catalog location (defaults to ./data)."
    ),
) -> None:
    """React to a few films and get tonight's picks — the main experience.

    Online it phrases questions and explains picks with a free model (needs OPENROUTER_API_KEY);
    with --offline, or no key, it runs the same recommender click-only, with fixed phrasing.
    """
    from .convo.engine import Engine

    settings = load_settings(overrides={"model": model})
    conn, matrix = _open_catalog_and_genome(data)

    online = not offline
    models: tuple[str, ...] = ()
    if online and not settings.has_api_key:
        err_console.print(
            f"[yellow]{API_KEY_ENV} is not set — running offline.[/yellow] "
            "Set a key for phrased questions and explanations, or pass --offline to silence this."
        )
        online = False
    if online:
        models = (settings.model,) if settings.model else tuple(free_models(settings))
        if not models:
            err_console.print("[yellow]No models available — running offline.[/yellow]")
            online = False

    io = _RichConsoleIO(console)
    try:
        if online:
            with OpenRouterClient(settings) as client:
                Engine(conn, matrix, io, settings, client=client, models=models).run()
        else:
            Engine(conn, matrix, io, settings, offline=True).run()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Okay — come back any time.[/dim]")
    finally:
        conn.close()


@app.command("eval")
def evaluate(
    seed: int = typer.Option(
        0, "--seed", help="Seed for the synthetic catalog; the run is deterministic in it."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Also print each persona's pair-question transcript."
    ),
) -> None:
    """Score the recommender against synthetic personas and report precision@3.

    Builds a small catalog with known ground truth, runs each persona through the offline engine,
    and measures how many of its three confident picks land in the taste it was built to have. No
    real catalog and no API key needed — it is self-contained and deterministic, so the number
    moves only when the recommender does. Tune a weight in the scorer and watch it here.
    """
    from .eval import run_eval

    report = run_eval(seed=seed)
    _render_eval(report, verbose=verbose)


def _render_eval(report, *, verbose: bool) -> None:
    """Print the precision table, the headline mean, and (optionally) the persona transcripts."""
    from collections import Counter

    table = Table(title=f"Recommender precision on {len(report.results)} synthetic personas")
    for name, justify in (
        ("Persona", "left"),
        ("Loves", "left"),
        ("Bounces off", "left"),
        ("Probes", "right"),
        ("Picks land in", "left"),
        ("p@3", "right"),
    ):
        table.add_column(name, justify=justify)

    for res in report.results:
        counts = Counter(res.pick_clusters)
        lands = ", ".join(f"{cluster} ×{n}" for cluster, n in counts.most_common()) or "—"
        precision = res.precision_at_3
        colour = "green" if precision >= 0.99 else "yellow" if precision >= 0.5 else "red"
        table.add_row(
            res.persona.name,
            res.persona.loved,
            res.persona.hated or "—",
            str(res.n_probes),
            lands,
            f"[{colour}]{precision:.2f}[/{colour}]",
        )
    console.print(table)
    console.print(
        f"\n[bold]Mean precision@3: {report.mean_precision_at_3:.3f}[/bold] "
        f"[dim]across {len(report.results)} personas · {report.catalog_size} films · "
        f"seed {report.seed}[/dim]"
    )
    if verbose:
        for res in report.results:
            console.print(
                f"\n[bold]{res.persona.name}[/bold] [dim]— {res.persona.description}[/dim]"
            )
            for turn in res.probe_turns:
                console.print(
                    f"  [cyan]{turn.high_label}[/cyan] or [cyan]{turn.low_label}[/cyan] "
                    f"→ chose {turn.chosen_label}"
                )
            console.print(f"  [dim]picks:[/dim] {', '.join(res.pick_titles) or '—'}")


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
    skip_predict: bool = typer.Option(
        False,
        "--skip-predict",
        help="Skip predicting genome vectors for uncovered films from TMDB features.",
    ),
    data: str | None = typer.Option(
        None,
        "--data-dir",
        help="Where to write cinegeist.db and genome.npy (defaults to ./data).",
    ),
) -> None:
    """Download MovieLens and build data/cinegeist.db and data/genome.npy.

    Resumable: a partial download continues, and finished stages are skipped. The first run
    fetches a few hundred MB and processes the tag genome, so it takes a while. TMDB enrichment
    runs next when a TMDB credential is set (see `cinegeist config`), then genome vectors are
    predicted for films the tag genome doesn't cover but TMDB does.
    """
    # Imported lazily so the light commands (models, config) don't pay numpy's import cost.
    from .catalog.build import build_catalog

    try:
        build_catalog(
            data_dir=Path(data) if data else None,
            force=force,
            enrich=not skip_enrich,
            predict_missing=not skip_predict,
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


profile_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and correct the taste profile the recommender has learned about you.",
)
app.add_typer(profile_app, name="profile")


def _open_catalog_and_genome(data: str | None):
    """Open the catalog and load the genome memmap, or exit with the build hint."""
    from .catalog import genome as genome_mod
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
    return conn, genome_mod.load_genome(genome_path)


def _open_catalog_only(data: str | None):
    """Open just the catalog database (no genome), or exit if it is not there."""
    from .catalog.db import open_catalog

    base = Path(data) if data else data_dir()
    db_path = base / "cinegeist.db"
    if not db_path.exists():
        err_console.print(
            "[red]No catalog found.[/red] Build one first with [bold]make catalog[/bold]."
        )
        raise typer.Exit(1)
    return open_catalog(db_path)


def _truncate(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe_event(conn, event) -> str:
    """A short human description of one event for forget/confirm output."""
    if event.subject_kind == "movie":
        row = conn.execute(
            "SELECT clean_title, title FROM movies WHERE movie_id = ?", (event.movie_id,)
        ).fetchone()
        title = (row["clean_title"] or row["title"]) if row else f"movie {event.subject}"
        verb = "liked" if event.value >= 0 else "disliked"
        label = f"{verb} '{title}'"
    elif event.subject_kind == "tag":
        row = conn.execute(
            "SELECT name FROM genome_tags WHERE tag_id = ?", (event.tag_id,)
        ).fetchone()
        name = row["name"] if row else f"tag {event.subject}"
        label = f"axis '{name}' {event.value:+.2f}"
    else:
        label = f"{event.subject} = {event.value:g}"
    if event.evidence:
        label += f" — “{_truncate(event.evidence)}”"
    return label


def _render_axes(heading: str, axes: list) -> None:
    if not axes:
        return
    console.print(f"\n[bold]{heading}[/bold]")
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column("tag")
    table.add_column("weight", justify="right")
    table.add_column("why")
    for axis in axes:
        if axis.evidence:
            why = f"[cyan]“{_truncate(axis.evidence)}”[/cyan]"
        else:
            why = f"[dim]{axis.source}[/dim]"
        table.add_row(axis.name, f"{axis.weight:+.2f}", why)
    console.print(table)


@profile_app.command("show")
def profile_show(
    limit: int = typer.Option(8, "--limit", "-n", help="How many axes to show per side."),
    data: str | None = typer.Option(
        None, "--data-dir", help="Catalog location (defaults to ./data)."
    ),
) -> None:
    """Print the taste profile: its strongest affinities and aversions, with your own words."""
    from .profile import update as profile_update

    conn, matrix = _open_catalog_and_genome(data)
    try:
        profile = profile_update.compute_profile(conn, matrix)
    finally:
        conn.close()

    if profile.is_empty:
        console.print(
            "[yellow]No taste profile yet.[/yellow] React to a few films with "
            "[bold]cinegeist chat[/bold] and this fills in."
        )
        return

    sessions = f"{profile.session_count} session" + ("" if profile.session_count == 1 else "s")
    events = f"{profile.event_count} event" + ("" if profile.event_count == 1 else "s")
    console.print(
        f"[bold]Your taste profile[/bold]  [dim]({events} across {sessions} · "
        f"evidence mass {profile.total_weight:.1f})[/dim]"
    )
    _render_axes("Drawn toward", profile.affinities[:limit])
    _render_axes("Pushed away from", profile.aversions[:limit])


@profile_app.command("forget")
def profile_forget(
    event_id: int = typer.Argument(..., help="The id of the event to delete (see `profile show`)."),
    data: str | None = typer.Option(
        None, "--data-dir", help="Catalog location (defaults to ./data)."
    ),
) -> None:
    """Delete one piece of evidence when the system latched onto something wrong."""
    from .profile import store as profile_store

    conn = _open_catalog_only(data)
    try:
        event = profile_store.get_event(conn, event_id)
        if event is None:
            err_console.print(f"[yellow]No event with id {event_id}.[/yellow]")
            raise typer.Exit(1)
        description = _describe_event(conn, event)
        profile_store.forget_event(conn, event_id)
    finally:
        conn.close()
    console.print(
        f"[green]Forgot event {event_id}[/green] ({description}). "
        "The profile recomputes without it."
    )


@profile_app.command("reset")
def profile_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    data: str | None = typer.Option(
        None, "--data-dir", help="Catalog location (defaults to ./data)."
    ),
) -> None:
    """Erase the whole taste profile and start over. Asks first unless you pass --yes."""
    from .profile import store as profile_store

    conn = _open_catalog_only(data)
    try:
        count = profile_store.count_events(conn)
        if count == 0:
            console.print("Nothing to reset — the profile is already empty.")
            return
        if not yes:
            typer.confirm(f"Delete all {count} events and start over?", abort=True)
        removed = profile_store.reset_profile(conn)
    finally:
        conn.close()
    console.print(f"[green]Reset.[/green] Removed {removed} events.")


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
