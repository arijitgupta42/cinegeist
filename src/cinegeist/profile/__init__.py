"""The taste profile: an append-only event log and the decayed view derived from it.

``store`` persists reactions as immutable :class:`~cinegeist.profile.model.PreferenceEvent`
rows; ``update`` replays them through a time-decayed weighting into a
:class:`~cinegeist.profile.model.TasteProfile`. Nothing here mutates a profile in place —
the profile is always a function of the log (plan.md §4.2, CLAUDE.md hard rule 7).
"""

from __future__ import annotations

from .model import PreferenceEvent, TagAffinity, TasteProfile

__all__ = ["PreferenceEvent", "TagAffinity", "TasteProfile"]
