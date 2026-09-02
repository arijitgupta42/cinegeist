"""A local HTTP surface over the conversation engine — the backend for the browser's full mode.

The browser demo on GitHub Pages ships no key and makes no LLM calls (plan.md §8.1); that stays
true. This package is the *other* half of the web frontend's two modes (plan.md §10, session 8):
when the same frontend is served by a local backend that someone set up themselves, it runs the
whole real conversation — free-text answers, LLM-phrased questions, full-catalog retrieval, real
explanations — by calling this server. The catalog, the profile, and the ``OPENROUTER_API_KEY``
stay on that person's machine; nothing about them enters the browser. It's the full local app with
a web UI instead of the terminal, not a hosted service, so it binds to localhost only by default.

The engine is unchanged: :mod:`cinegeist.serve.conversation` adapts its blocking, one-turn loop to
request/response HTTP, and :mod:`cinegeist.serve.server` routes the JSON API and serves the built
frontend.
"""

from __future__ import annotations

from .conversation import SessionManager, Turn
from .server import DEFAULT_HOST, DEFAULT_PORT, build_server, run_server

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "SessionManager",
    "Turn",
    "build_server",
    "run_server",
]
