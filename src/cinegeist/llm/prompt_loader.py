"""Load prompt templates from ``llm/prompts/*.md``.

Prompts live in Markdown files, one per job, never as inline strings (CLAUDE.md code style) — so
they can be edited and reviewed as prose without touching Python. This is the single reader for
them; later jobs (rerank, explain, probe phrasing) load their prompts the same way.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

_PACKAGE = "cinegeist.llm"
_PROMPTS_DIR = "prompts"


@cache
def load_prompt(name: str) -> str:
    """Return the text of ``llm/prompts/<name>.md``. Cached — prompts don't change at runtime."""
    resource = resources.files(_PACKAGE).joinpath(_PROMPTS_DIR, f"{name}.md")
    return resource.read_text(encoding="utf-8").strip()
