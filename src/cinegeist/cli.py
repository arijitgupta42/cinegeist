"""Command line entry point for cinegeist.

This scaffold defines the Typer application and a ``--version`` flag only. The real
commands (``models``, ``ask``, ``config``) are added in later changes so that every
commit leaves the package installable and runnable.
"""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A conversational movie recommender for people who can't say what they like.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"cinegeist {__version__}")
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


def run() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    app()
