"""The conversation layer: opening questions, title resolution, extraction, probes.

This package holds the deterministic conversation machinery. The LLM only ever phrases and
parses; the choices — which question, which films, how an answer maps to the catalog — are made
here in plain Python so the whole thing is testable and reproducible (plan.md §3, §5).
"""
