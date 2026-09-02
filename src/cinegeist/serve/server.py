"""The HTTP layer: a small JSON API over the conversation, plus static serving of the frontend.

Deliberately built on the standard library's :mod:`http.server` — a local, single-user backend
does not justify a web framework dependency (CLAUDE.md's dependency rule). It binds to localhost by
default; this reads the machine's real catalog, profile, and API key, so it is not something to
expose on the network.

Routes:

* ``GET  /api/health``                 — liveness plus mode, so the frontend can detect full mode
* ``POST /api/session``                — start a conversation; returns the opening turn
* ``POST /api/session/{id}/answer``    — answer the pending question; returns the next turn
* ``DELETE /api/session/{id}``         — end a conversation early (tab closed)
* ``GET  /*``                          — the built frontend from ``web_dir`` (SPA fallback)

The API is same-origin — the frontend is served from here too — so there is no CORS surface. For a
split dev setup (Vite on another port) the dev server proxies ``/api`` instead.
"""

from __future__ import annotations

import json
import mimetypes
import re
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .conversation import SessionBusy, SessionDone, SessionManager

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# POST /api/session/<id>/answer  and  DELETE /api/session/<id>
_ANSWER_RE = re.compile(r"^/api/session/([0-9a-f]{32})/answer$")
_SESSION_RE = re.compile(r"^/api/session/([0-9a-f]{32})$")
# Cap a request body so a bad client can't make us buffer without bound; answers are tiny.
_MAX_BODY = 64 * 1024

HealthFn = Callable[[], dict[str, Any]]


def build_server(
    manager: SessionManager,
    *,
    health: HealthFn,
    web_dir: Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Construct (but don't start) the threaded HTTP server. Port 0 binds an ephemeral port."""
    resolved_web = web_dir.resolve() if web_dir else None

    class Handler(_CineGeistHandler):
        pass

    Handler.manager = manager
    Handler.health = staticmethod(health)
    Handler.web_dir = resolved_web
    return ThreadingHTTPServer((host, port), Handler)


def run_server(
    manager: SessionManager,
    *,
    health: HealthFn,
    web_dir: Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Build and run the server until interrupted, then shut sessions down cleanly."""
    server = build_server(manager, health=health, web_dir=web_dir, host=host, port=port)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    url = f"http://{bound_host}:{bound_port}"
    if on_ready is not None:
        on_ready(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        manager.shutdown()


class _CineGeistHandler(BaseHTTPRequestHandler):
    """One request. Class attributes carry the shared manager, health probe, and web root."""

    manager: SessionManager
    health: HealthFn
    web_dir: Path | None

    server_version = "cinegeist"
    protocol_version = "HTTP/1.1"

    # -- routing ----------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's dispatch names
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, self.health())
            return
        if path.startswith("/api/"):
            self._send_json(404, {"error": "not found"})
            return
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/session":
            self._start_session()
            return
        match = _ANSWER_RE.match(path)
        if match:
            self._answer(match.group(1))
            return
        self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        match = _SESSION_RE.match(path)
        if match:
            self.manager.end(match.group(1))
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not found"})

    # -- API handlers -----------------------------------------------------------------

    def _start_session(self) -> None:
        self._read_body()  # drain any body so keep-alive stays in sync; contents are ignored
        try:
            session_id, turn = self.manager.create()
        except SessionBusy:
            self._send_json(429, {"error": "too many conversations in progress"})
            return
        self._send_json(200, turn.to_json(session_id))

    def _answer(self, session_id: str) -> None:
        body = self._read_body()
        if body is None:
            return  # _read_body already sent the error
        answer = body.get("answer")
        if not isinstance(answer, str):
            self._send_json(400, {"error": 'body must be {"answer": "..."}'})
            return
        try:
            turn = self.manager.answer(session_id, answer)
        except KeyError:
            self._send_json(404, {"error": "no such session"})
            return
        except SessionDone:
            self._send_json(409, {"error": "this conversation has ended"})
            return
        except SessionBusy:
            self._send_json(409, {"error": "still working on the previous answer"})
            return
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        self._send_json(200, turn.to_json(session_id))

    # -- static frontend --------------------------------------------------------------

    def _serve_static(self, path: str) -> None:
        if self.web_dir is None:
            self._send_no_frontend()
            return
        target = self._safe_path(path)
        if target is None or not target.is_file():
            target = self.web_dir / "index.html"  # SPA fallback
        if not target.is_file():
            self._send_no_frontend()
            return
        data = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if target.name == "index.html":
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _safe_path(self, path: str) -> Path | None:
        """Map a URL path to a file under ``web_dir``, refusing anything that escapes it."""
        assert self.web_dir is not None
        relative = path.lstrip("/") or "index.html"
        candidate = (self.web_dir / relative).resolve()
        if candidate != self.web_dir and self.web_dir not in candidate.parents:
            return None
        return candidate

    def _send_no_frontend(self) -> None:
        body = (
            "<!doctype html><meta charset=utf-8><title>cinegeist</title>"
            "<body style='font:16px system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem'>"
            "<h1>cinegeist is serving the API</h1>"
            "<p>The JSON API is live at <code>/api/health</code>. To serve the web UI from here, "
            "build the frontend (<code>cd web &amp;&amp; npm run build</code>) and restart with "
            "<code>--web-dir web/dist</code>, or run the Vite dev server, which proxies "
            "<code>/api</code> to this backend.</p></body>"
        )
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    # -- plumbing ---------------------------------------------------------------------

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > _MAX_BODY:
            # We won't read the oversized body, so this connection can't be safely reused.
            self.close_connection = True
            self._send_json(413, {"error": "request too large"})
            return None
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid JSON"})
            return None
        if not isinstance(parsed, dict):
            self._send_json(400, {"error": "body must be a JSON object"})
            return None
        return parsed

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args: Any) -> None:
        """Stay quiet: no access log, and in particular never echo request bodies."""
