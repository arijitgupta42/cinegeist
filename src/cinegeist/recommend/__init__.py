"""The recommender: turn a taste profile into a handful of films to watch.

This is plan.md §6 — hard filter, score, diversify, shortlist, present — split across small
deterministic modules. :mod:`retrieve` builds the candidate pool; :mod:`score` ranks it by taste
and diversifies it. The LLM rerank and the explanations (later PRs) sit on top and never invent a
film: they only ever reorder or describe ids that came out of here (CLAUDE.md hard rule 1).
"""
