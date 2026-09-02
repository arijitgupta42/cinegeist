"""A scripted persona standing in for the user, so a whole conversation runs with no human.

The engine talks to the user through one narrow interface, :class:`~cinegeist.convo.engine.
ConversationIO` (say / ask_text / ask_choice / show_presentation). The terminal is one
implementation; this oracle is another. It answers every pair question the way its persona would —
by preferring the film closer to the persona's taste direction — and gives fixed, sensible answers
to the constraint and "show me more" questions, so the run reaches a set of recommendations without
a person in the loop. It records what it was asked and what it chose, which is the replayable
transcript the harness reports.

It only ever chooses a real option; it never takes the escape hatch or "I haven't seen either", so
the conversation plays out in full. Free text never comes up — the engine's seed phase is online
only, and the eval runs offline — but ``ask_text`` returns empty defensively.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..catalog import genome
from ..convo.engine import Choice
from ..recommend import present as present_mod
from .catalog import SyntheticCatalog


@dataclass
class ProbeTurn:
    """One pair question the persona was asked and the film it chose."""

    question: str
    high_label: str
    low_label: str
    chosen_label: str


@dataclass
class PersonaOracle:
    """Drives the conversation as one persona, recording the turns and the picks it is shown."""

    catalog: SyntheticCatalog
    direction: np.ndarray
    probe_turns: list[ProbeTurn] = field(default_factory=list)
    narration: list[str] = field(default_factory=list)
    presentation: present_mod.Presentation | None = None
    _row_by_title: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._row_by_title:
            self._row_by_title = {
                row["clean_title"]: row["genome_row"]
                for row in self.catalog.conn.execute(
                    "SELECT clean_title, genome_row FROM movies WHERE genome_row IS NOT NULL"
                )
            }

    # -- ConversationIO -----------------------------------------------------------------

    def say(self, text: str) -> None:
        self.narration.append(text)

    def ask_text(self, prompt: str) -> str:
        return ""  # offline conversations never ask for free text

    def ask_choice(self, prompt: str, options: Sequence[Choice]) -> str:
        keys = {option.key for option in options}
        if {"high", "low"} <= keys:
            return self._answer_probe(prompt, options)
        if "any" in keys:  # how much time tonight — take no limit so retrieval isn't narrowed
            return "any"
        if keys == {"yes", "no"}:  # subtitles — fine, so no language filter
            return "yes"
        if "refine" in keys:  # returning-user offer — keep answering questions
            return "refine"
        if "more" in keys and "done" in keys:  # present loop — one page is enough
            return "done"
        return options[0].key  # any other menu (e.g. feedback): the first option is harmless

    def show_presentation(self, presentation: present_mod.Presentation, *, header: str) -> None:
        if self.presentation is None:  # the first page is the one precision is measured on
            self.presentation = presentation

    # -- taste ---------------------------------------------------------------------------

    def _answer_probe(self, prompt: str, options: Sequence[Choice]) -> str:
        """Prefer whichever paired film sits closer to the persona's taste direction."""
        high = next(o for o in options if o.key == "high")
        low = next(o for o in options if o.key == "low")
        high_fit = self._fit(high.label)
        low_fit = self._fit(low.label)
        chosen = high if high_fit >= low_fit else low
        self.probe_turns.append(
            ProbeTurn(
                question=prompt,
                high_label=high.label,
                low_label=low.label,
                chosen_label=chosen.label,
            )
        )
        return chosen.key

    def _fit(self, title: str) -> float:
        """Cosine of a film (looked up by its title) against the persona's taste direction."""
        row = self._row_by_title.get(title)
        if row is None:
            return -1.0  # a label we can't map (shouldn't happen) is the least preferred
        vector = self.catalog.matrix[row][np.newaxis, :]
        return float(genome.cosine_scores(vector, self.direction)[0])
