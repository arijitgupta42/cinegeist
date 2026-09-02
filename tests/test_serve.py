"""Tests for the local HTTP surface over the conversation engine.

Everything runs offline — the engine makes no LLM calls — so a whole conversation over the API is
deterministic. Two levels are covered: the :class:`SessionManager` bridge directly (the thread and
queue handoff, validation, and the end of a run), and one real HTTP round trip through the server
to prove the routing, JSON shapes, and error codes.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.config import load_settings
from cinegeist.convo.engine import Engine
from cinegeist.serve import SessionManager, build_server
from cinegeist.serve.conversation import SessionDone, Turn

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# The same six-film, eight-tag shape as the engine tests, so probe selection has real axes to ask.
_MATRIX = np.array(
    [
        [1.0, 1.0, 0, 0, 0, 0, 0, 0],
        [0.95, 0.9, 0.2, 0, 0, 0, 0, 0],
        [0.9, 0.95, 0, 0.3, 0, 0, 0, 0],
        [0.6, 0.6, 1, 1, 1, 1, 1, 1],
        [0.0, 0.0, 1, 1, 0, 0, 0, 0],
        [0.2, 0.0, 0.5, 0, 0.4, 0, 0, 0],
    ],
    dtype=np.float32,
)


def _settings():
    return load_settings(env={}, config_file=Path("nope.toml"))


def _build_catalog(path: Path) -> None:
    conn = db.open_catalog(path)
    conn.executemany(
        "INSERT INTO genome_tags (tag_id, position, name) VALUES (?, ?, ?)",
        [(100 + i, i, f"t{i}") for i in range(8)],
    )
    conn.executemany(
        "INSERT INTO movies (movie_id, title, clean_title, year, genome_row, genome_source) "
        "VALUES (?, ?, ?, ?, ?, 'measured')",
        [(mid, f"F{mid} ({2000 + mid})", f"F{mid}", 2000 + mid, mid - 1) for mid in range(1, 7)],
    )
    conn.commit()
    conn.close()


def _manager(db_path: Path) -> SessionManager:
    def make_runner(_io):
        def runner(io):
            conn = db.open_catalog(db_path)
            try:
                Engine(conn, _MATRIX, io, _settings(), offline=True, now=_NOW).run()
            finally:
                conn.close()

        return runner

    return SessionManager(make_runner)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "cinegeist.db"
    _build_catalog(path)
    return path


def _pick(prompt: dict) -> str:
    """A sensible answer to any choice, so a whole run completes without scripting each turn."""
    keys = [o["key"] for o in prompt["options"]]
    for preferred in ("done", "high", "standard", "yes"):
        if preferred in keys:
            return preferred
    if "refine" in keys:
        return "refine"
    return keys[0]


def _drive(manager: SessionManager, session_id: str, turn: Turn) -> list[Turn]:
    turns = [turn]
    while not turn.done:
        assert turn.prompt is not None, "a non-final turn must carry a question"
        turn = manager.answer(session_id, _pick(turn.prompt))
        turns.append(turn)
    return turns


# -- the manager bridge --------------------------------------------------------------


def test_a_whole_conversation_runs_over_the_bridge(db_path: Path) -> None:
    manager = _manager(db_path)
    session_id, first = manager.create()

    assert first.prompt is not None
    assert first.prompt["kind"] == "choice"
    # The first question a fresh visitor gets is a pair probe: it offers "neither" and the escape.
    keys = {o["key"] for o in first.prompt["options"]}
    assert "neither" in keys and "__stop__" in keys

    turns = _drive(manager, session_id, first)

    assert turns[-1].done
    picks = [m for t in turns for m in t.messages if m["type"] == "picks"]
    assert picks, "the conversation must end by presenting picks"
    assert picks[0]["picks"], "there should be at least one confident pick"
    assert manager.count() == 0, "a finished conversation is discarded"


def test_escape_hatch_ends_the_conversation(db_path: Path) -> None:
    manager = _manager(db_path)
    session_id, first = manager.create()
    turn = manager.answer(session_id, "__stop__")
    # Escaping jumps straight to the picks and ends the run.
    while not turn.done:
        turn = manager.answer(session_id, _pick(turn.prompt))
    assert turn.done


def test_a_choice_answer_must_be_one_of_the_offered_keys(db_path: Path) -> None:
    manager = _manager(db_path)
    session_id, _first = manager.create()
    with pytest.raises(ValueError, match="not one of"):
        manager.answer(session_id, "definitely-not-a-key")
    # The session survives a rejected answer and still accepts a valid one.
    turn = manager.answer(session_id, "high")
    assert turn.prompt is not None or turn.done


def test_answering_an_unknown_session_raises(db_path: Path) -> None:
    manager = _manager(db_path)
    with pytest.raises(KeyError):
        manager.answer("f" * 32, "high")


def test_answering_after_the_end_is_rejected() -> None:
    # A Session whose run has ended reports it, rather than feeding a dead engine thread.
    from cinegeist.serve.conversation import HttpIO, Session

    io = HttpIO()
    session = Session(id="x", io=io, thread=threading.Thread(target=lambda: None))
    session.done = True
    session.pending = None
    with pytest.raises(SessionDone):
        session.answer("high")


def test_reaping_closes_idle_sessions(db_path: Path) -> None:
    def make_runner(_io):
        def runner(io):
            conn = db.open_catalog(db_path)
            try:
                Engine(conn, _MATRIX, io, _settings(), offline=True, now=_NOW).run()
            finally:
                conn.close()

        return runner

    manager = SessionManager(make_runner, idle_timeout_s=-1.0)  # everything is immediately stale
    manager.create()
    assert manager.count() == 1
    # The next create reaps the previous (now stale) session first.
    manager.create()
    assert manager.count() == 1


# -- the HTTP server -----------------------------------------------------------------


@pytest.fixture
def client(db_path: Path):
    manager = _manager(db_path)

    def health() -> dict:
        return {"ok": True, "offline": True, "films": 6, "sessions": manager.count()}

    server = build_server(manager, health=health, web_dir=None, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as c:
            yield c
    finally:
        server.shutdown()
        server.server_close()
        manager.shutdown()


def test_health_reports_mode_and_catalog_size(client: httpx.Client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["offline"] is True
    assert body["films"] == 6


def test_a_conversation_over_http(client: httpx.Client) -> None:
    start = client.post("/api/session")
    assert start.status_code == 200
    opening = start.json()
    session_id = opening["session_id"]
    assert opening["prompt"]["kind"] == "choice"

    # Answer probes with a valid key until the run ends, exactly as the browser would, keeping
    # every message so we can prove the picks were presented (they arrive before the final turn).
    messages = list(opening["messages"])
    turn = opening
    guard = 0
    while not turn["done"] and guard < 40:
        guard += 1
        answer = _pick(turn["prompt"])
        response = client.post(f"/api/session/{session_id}/answer", json={"answer": answer})
        assert response.status_code == 200, response.text
        turn = response.json()
        messages.extend(turn["messages"])

    assert turn["done"]
    picks = [m for m in messages if m["type"] == "picks"]
    assert picks, "the conversation must present picks over the wire"
    assert picks[0]["picks"], "at least one confident pick"


def test_a_bad_choice_answer_is_rejected(client: httpx.Client) -> None:
    session_id = client.post("/api/session").json()["session_id"]
    response = client.post(f"/api/session/{session_id}/answer", json={"answer": "nope"})
    assert response.status_code == 400


def test_answering_an_unknown_session_is_404(client: httpx.Client) -> None:
    response = client.post(f"/api/session/{'a' * 32}/answer", json={"answer": "high"})
    assert response.status_code == 404


def test_a_missing_answer_field_is_400(client: httpx.Client) -> None:
    session_id = client.post("/api/session").json()["session_id"]
    response = client.post(f"/api/session/{session_id}/answer", json={"nope": 1})
    assert response.status_code == 400


def test_deleting_a_session_succeeds(client: httpx.Client) -> None:
    session_id = client.post("/api/session").json()["session_id"]
    response = client.request("DELETE", f"/api/session/{session_id}")
    assert response.status_code == 200


def test_unknown_api_route_is_404(client: httpx.Client) -> None:
    assert client.get("/api/nope").status_code == 404


def test_the_root_explains_the_api_when_no_frontend_is_built(client: httpx.Client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "serving the API" in response.text
