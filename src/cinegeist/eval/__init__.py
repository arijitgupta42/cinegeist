"""Offline evaluation: run synthetic personas through the real recommender and report precision@3.

The pieces: :mod:`catalog` generates a clustered synthetic catalog with known ground truth,
:mod:`personas` declares the synthetic viewers, :mod:`oracle` answers the conversation as one of
them, and :mod:`harness` runs the whole thing and scores it. See :func:`run_eval` for the entry
point the ``cinegeist eval`` command and the CI test both call.
"""

from __future__ import annotations

from .harness import EvalReport, PersonaResult, run_eval, run_persona

__all__ = ["EvalReport", "PersonaResult", "run_eval", "run_persona"]
