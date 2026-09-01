"""Build the static browser-demo shard from the full catalog (plan.md §8.3).

The demo can't ship a 59 MB genome, so this package reduces the catalog to ~2,000 films with
SVD-compressed int8 taste vectors, 3D coordinates, top tags, and (from a later PR) per-film
coverage — a ~300 KB bundle the browser scores in plain JavaScript. The logic here is pure and
importable so it can be unit-tested without the real catalog; ``scripts/build_web_shard.py`` is the
thin orchestrator that reads ``data/`` and writes the shard.
"""
