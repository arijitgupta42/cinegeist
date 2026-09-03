"""Integration tests for the conversation engine, driven by a scripted IO.

The engine talks to the world only through :class:`ConversationIO`, so a scripted double stands in
for the terminal: it answers choices by a small responder function and records what the engine
said and showed. Everything here runs offline — no LLM — so a whole conversation is deterministic
and we can assert exactly what got recorded and presented.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from cinegeist.catalog import db
from cinegeist.config import load_settings
from cinegeist.convo import probes
from cinegeist.convo.engine import Engine
from cinegeist.profile import store, update
from cinegeist.profile.model import PreferenceEvent

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# Six films over an 8-tag genome (same shape as the present tests): 1–3 near tags 0/1, 4 far but
# grounded, 5–6 low. Enough spread that probe selection has real axes to ask about.
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


@pytest.fixture
def catalog() -> sqlite3.Connection:
    conn = db.connect(":memory:")
    db.migrate(conn)
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
    return conn


def _settings():
    return load_settings(env={}, config_file=__import__("pathlib").Path("nope.toml"))


class ScriptedIO:
    """A ConversationIO that answers choices via ``responder`` and records everything shown."""

    def __init__(self, responder=None, texts=None) -> None:
        self.responder = responder or _default_responder
        self.texts = list(texts or [])
        self.said: list[str] = []
        self.presentations: list[tuple[str, object]] = []
        self.choice_prompts: list[tuple[str, list[str]]] = []
        self.state: dict = {}

    def say(self, text: str) -> None:
        self.said.append(text)

    def ask_text(self, prompt: str) -> str:
        return self.texts.pop(0) if self.texts else ""

    def ask_choice(self, prompt: str, options) -> str:
        keys = [o.key for o in options]
        self.choice_prompts.append((prompt, keys))
        key = self.responder(prompt, keys, self.state)
        assert key in keys, f"responder returned {key!r} not in {keys!r}"
        return key

    def show_presentation(self, presentation, *, header: str) -> None:
        self.presentations.append((header, presentation))


def _default_responder(prompt: str, keys: list[str], state: dict) -> str:
    """A sensible default answer for any choice, so a whole run completes without scripting each."""
    for preferred in ("done", "high", "standard", "yes"):
        if preferred in keys:
            return preferred
    if "picks" in keys:  # returning-user prompt: refine so the full flow runs
        return "refine"
    return keys[0]


def _engine(conn, io, **kwargs) -> Engine:
    return Engine(conn, _MATRIX, io, _settings(), offline=True, now=_NOW, **kwargs)


def _axis_events(conn) -> list[PreferenceEvent]:
    return [e for e in store.iter_events(conn) if e.kind == "axis_answer"]


# -- a whole offline run -------------------------------------------------------------


def test_offline_run_reaches_picks_and_records_evidence(catalog) -> None:
    io = ScriptedIO()  # answers "high" to every probe, then the constrain + present defaults
    _engine(catalog, io).run()

    assert io.presentations, "the run must end by presenting picks"
    header, presentation = io.presentations[0]
    assert header == "Your picks"
    assert presentation.picks, "there should be at least one confident pick"

    # Every probe answered "high" recorded one axis_answer, so the profile is no longer empty.
    assert _axis_events(catalog), "probe choices must be recorded as evidence"
    profile = update.compute_profile(catalog, _MATRIX, now=_NOW)
    assert not profile.is_empty


def test_every_probe_offers_the_escape_hatch(catalog) -> None:
    io = ScriptedIO()
    _engine(catalog, io).run()
    # Every this-or-that probe (the ones offering "neither") must also offer the stop option.
    probe_prompts = [keys for _p, keys in io.choice_prompts if "neither" in keys]
    assert probe_prompts, "there should have been at least one probe"
    for keys in probe_prompts:
        assert probes._STOP_PATTERNS  # sanity: stop vocabulary exists
        assert "__stop__" in keys


class _StubReply:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubClient:
    """Returns a scripted phrasing reply, so we can exercise the online phrasing path."""

    def __init__(self, text: str) -> None:
        self._text = text

    def chat_with_failover(self, _messages, _models, **_kwargs) -> _StubReply:
        return _StubReply(self._text)


def _probe():
    from types import SimpleNamespace

    return SimpleNamespace(
        axis_name="tense",
        axis_position=2,
        film_high=SimpleNamespace(title="Blue Ruin", movie_id=1),
        film_low=SimpleNamespace(title="The Tree of Life", movie_id=2),
        question="Which would you rather put on tonight — Blue Ruin or The Tree of Life?",
    )


def _online_engine(conn, io, reply_text: str) -> Engine:
    return Engine(
        conn, _MATRIX, io, _settings(), client=_StubClient(reply_text), models=("m",), now=_NOW
    )


def test_a_good_phrasing_is_used(catalog) -> None:
    good = "Raw tension or a slow meditation — Blue Ruin or Tree of Life?"
    engine = _online_engine(catalog, ScriptedIO(), good)
    assert engine._phrase_probe(_probe()) == good


def test_a_truncated_phrasing_falls_back_to_the_fixed_wording(catalog) -> None:
    # A reasoning model can burn the token budget thinking and return a stub or nothing; the engine
    # must show the fixed question, never the fragment.
    probe = _probe()
    for stub in ("Which", "Which are you putting on", "", "   "):
        engine = _online_engine(catalog, ScriptedIO(), stub)
        assert engine._phrase_probe(probe) == probe.question


def test_escape_hatch_jumps_straight_to_picks(catalog) -> None:
    # Hit the escape option on the very first probe.
    def responder(prompt, keys, state):
        if "__stop__" in keys and not state.get("escaped"):
            state["escaped"] = True
            return "__stop__"
        return _default_responder(prompt, keys, state)

    io = ScriptedIO(responder=responder)
    _engine(catalog, io).run()

    assert io.presentations, "escaping must still produce picks"
    assert not _axis_events(catalog), "no probe was actually answered before escaping"


def test_show_me_three_more_re_presents_without_repeats(catalog) -> None:
    # Answer "more" on the first present, "done" on the second.
    def responder(prompt, keys, state):
        if "more" in keys:
            if not state.get("asked_more"):
                state["asked_more"] = True
                return "more"
            return "done"
        return _default_responder(prompt, keys, state)

    io = ScriptedIO(responder=responder)
    _engine(catalog, io).run()

    assert len(io.presentations) >= 2, "‘three more’ must present a second page"
    first_ids = {p.film.movie_id for p in io.presentations[0][1].all_films}
    second_ids = {p.film.movie_id for p in io.presentations[1][1].all_films}
    assert first_ids.isdisjoint(second_ids), "a second page must not repeat the first"


# -- returning user ------------------------------------------------------------------


def test_returning_user_can_jump_straight_to_picks(catalog) -> None:
    # Seed a confident profile so the greet offers the one-turn path.
    store.append_events(
        catalog,
        [
            PreferenceEvent.liked_movie(1, weight=1.5, session_id="old"),
            PreferenceEvent.liked_movie(2, weight=1.5, session_id="old"),
        ],
        now=_NOW,
    )

    def responder(prompt, keys, state):
        if "picks" in keys:
            return "picks"  # take the shortcut
        return _default_responder(prompt, keys, state)

    io = ScriptedIO(responder=responder)
    _engine(catalog, io).run()

    # Jumping to picks means no probe was ever asked (no "neither" choice appeared).
    assert not any("neither" in keys for _p, keys in io.choice_prompts)
    assert io.presentations


def test_returning_user_is_asked_about_the_last_recommendation(catalog) -> None:
    # Prior events make the user "returning", and a stored last-rec triggers the feedback prompt.
    store.append_event(
        catalog, PreferenceEvent.liked_movie(1, weight=1.5, session_id="old"), now=_NOW
    )
    db.set_state(catalog, "last_recs:default", json.dumps([{"id": 5, "title": "F5"}]))

    def responder(prompt, keys, state):
        if "not_for_me" in keys:  # the verdict menu for "Did you get to F5?"
            return "not_for_me"
        if "picks" in keys:
            return "refine"
        return _default_responder(prompt, keys, state)

    io = ScriptedIO(responder=responder)
    _engine(catalog, io).run()

    feedback_events = [e for e in store.iter_events(catalog) if e.kind == "post_watch_feedback"]
    assert any(e.subject == "5" and e.value < 0 for e in feedback_events)


# -- constraints ---------------------------------------------------------------------


def test_runtime_constraint_is_applied_to_the_picks(catalog) -> None:
    # Give the films runtimes: only F2 and F5 are under 90 minutes.
    catalog.execute("UPDATE movies SET runtime = 120")
    catalog.execute("UPDATE movies SET runtime = 85 WHERE movie_id IN (2, 5)")
    catalog.commit()

    def responder(prompt, keys, state):
        if "short" in keys:  # the time question
            return "short"
        if "yes" in keys and "no" in keys:  # the subtitles question
            return "yes"
        return _default_responder(prompt, keys, state)

    io = ScriptedIO(responder=responder)
    _engine(catalog, io).run()

    shown = {p.film.movie_id for _h, pres in io.presentations for p in pres.all_films}
    assert shown, "something should be presented"
    assert shown <= {2, 5}, "only films under 90 minutes should appear"
