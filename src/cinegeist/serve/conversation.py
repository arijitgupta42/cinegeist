"""Bridge the blocking conversation engine to a request/response HTTP surface.

The engine (:mod:`cinegeist.convo.engine`) runs a conversation as one blocking call: it *pushes*
narration and *pulls* answers through :class:`~cinegeist.convo.engine.ConversationIO`, one turn at
a time, and only returns when the whole conversation is finished. An HTTP client is the opposite
shape — one request, one response, come back later for the next.

We reconcile the two by running each conversation in its own thread and passing control back and
forth through a pair of queues. The engine thread runs until it needs an answer (or the run ends),
parks itself on a queue, and hands the turn it accumulated to whatever HTTP request is waiting. The
next request drops the answer into the other queue, which unparks the engine thread. Nothing about
the engine or the deterministic layers changes — this is purely an adapter, so the web conversation
and the terminal one are the same conversation.

A session is one such thread. :class:`SessionManager` owns them: it builds an engine per session
(each with its own SQLite connection, since connections aren't shared across threads), enforces a
cap and an idle timeout, and reaps sessions whose tab was closed without a goodbye.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from queue import Empty, Queue
from typing import Any

from ..convo.engine import Choice, ConversationIO

# The engine that runs a session; given the queue-backed IO, it holds one whole conversation. The
# manager supplies this so the transport layer never imports the catalog or the LLM client.
Runner = Callable[[ConversationIO], None]

# A parked engine thread waits this long for the next answer before the whole session is abandoned;
# a live HTTP request waits this long for the engine to produce its next turn. Online turns make an
# LLM call, so it must comfortably clear a slow model, not just an instant offline step.
_TURN_TIMEOUT_S = 120.0
# Sessions idle longer than this are reaped — a browser tab closed mid-conversation leaves its
# engine thread parked forever otherwise.
_IDLE_TIMEOUT_S = 30 * 60.0
# A ceiling on concurrent conversations, so a script hitting POST /api/session can't spawn threads
# without bound. Generous for a single-user local backend.
_MAX_SESSIONS = 32

# Fed to a parked engine thread to unwind it cleanly when its session is closed or reaped.
_CLOSE = object()


class SessionClosed(Exception):
    """Raised inside a parked ``ask_*`` when the session is closed, to unwind the engine thread."""


class SessionBusy(Exception):
    """A second answer arrived while the engine was still working on the previous one."""


class SessionDone(Exception):
    """An answer arrived after the conversation had already ended."""


@dataclass(frozen=True)
class Turn:
    """One HTTP-shaped step of a conversation: what to show, and what (if anything) to ask next.

    ``messages`` is everything the engine narrated since the last question — ``say`` lines and
    ``picks`` blocks. ``prompt`` is the question the engine is now parked on (``None`` once the
    conversation is over). ``done`` is set on the final turn; ``error`` marks an abnormal end.
    """

    messages: tuple[dict[str, Any], ...]
    prompt: dict[str, Any] | None
    done: bool = False
    error: bool = False

    def to_json(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "messages": list(self.messages),
            "prompt": self.prompt,
            "done": self.done,
            "error": self.error,
        }


class HttpIO(ConversationIO):
    """A :class:`ConversationIO` that turns the engine's push/pull loop into queued turns.

    The engine thread calls ``say`` / ``show_presentation`` (buffered) and ``ask_*`` (which hands
    the buffered turn to the client and blocks for the answer). The serving thread calls
    :meth:`next_turn` to collect a turn and :meth:`provide` to answer it.
    """

    def __init__(self) -> None:
        self._outbox: list[dict[str, Any]] = []
        self._to_client: Queue[Turn] = Queue()
        self._from_client: Queue[Any] = Queue(maxsize=1)

    # -- engine side (runs on the session thread) -------------------------------------

    def say(self, text: str) -> None:
        self._outbox.append({"type": "say", "text": text})

    def show_presentation(self, presentation: Any, *, header: str) -> None:
        self._outbox.append(_serialize_presentation(presentation, header))

    def ask_text(self, prompt: str) -> str:
        answer = self._exchange({"kind": "text", "text": prompt})
        return "" if answer is _CLOSE else str(answer)

    def ask_choice(self, prompt: str, options: Sequence[Choice]) -> str:
        payload = {
            "kind": "choice",
            "text": prompt,
            "options": [{"key": o.key, "label": o.label} for o in options],
        }
        answer = self._exchange(payload)
        if answer is _CLOSE:
            raise SessionClosed
        return str(answer)

    def _exchange(self, prompt: dict[str, Any]) -> Any:
        """Hand the accumulated turn to the client, then block for the answer."""
        self._to_client.put(Turn(messages=self._drain(), prompt=prompt))
        answer = self._from_client.get()
        if answer is _CLOSE:
            raise SessionClosed
        return answer

    def finish(self) -> None:
        """Emit the final turn once the engine's ``run()`` returns normally."""
        self._to_client.put(Turn(messages=self._drain(), prompt=None, done=True))

    def fail(self, message: str) -> None:
        """Emit a final turn describing an abnormal end, so the client never hangs."""
        self._outbox.append({"type": "error", "text": message})
        self._to_client.put(Turn(messages=self._drain(), prompt=None, done=True, error=True))

    def _drain(self) -> tuple[dict[str, Any], ...]:
        out = tuple(self._outbox)
        self._outbox.clear()
        return out

    # -- serving side (runs on the request thread) ------------------------------------

    def next_turn(self, timeout: float = _TURN_TIMEOUT_S) -> Turn:
        try:
            return self._to_client.get(timeout=timeout)
        except Empty:
            return Turn(
                messages=({"type": "error", "text": "The conversation timed out."},),
                prompt=None,
                done=True,
                error=True,
            )

    def provide(self, answer: Any) -> None:
        self._from_client.put(answer)

    def close(self) -> None:
        """Unpark the engine thread (if parked) so it can unwind. Safe to call more than once."""
        try:
            self._from_client.put_nowait(_CLOSE)
        except Exception:  # noqa: BLE001 — queue full means it's already been signalled
            pass


@dataclass
class Session:
    """One live conversation: its engine thread, its IO bridge, and its current pending question."""

    id: str
    io: HttpIO
    thread: threading.Thread
    pending: dict[str, Any] | None = None
    done: bool = False
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _record(self, turn: Turn) -> Turn:
        self.pending = turn.prompt
        self.done = turn.done
        self.last_active = datetime.now(UTC)
        return turn

    def first_turn(self) -> Turn:
        return self._record(self.io.next_turn())

    def answer(self, text: str) -> Turn:
        """Feed one answer and return the next turn. Rejects concurrent or post-end answers."""
        if not self._lock.acquire(blocking=False):
            raise SessionBusy
        try:
            if self.done or self.pending is None:
                raise SessionDone
            self._validate(text)
            self.io.provide(text)
            return self._record(self.io.next_turn())
        finally:
            self._lock.release()

    def _validate(self, text: str) -> None:
        """A choice answer must be one of the offered keys; a text answer is anything."""
        prompt = self.pending or {}
        if prompt.get("kind") == "choice":
            keys = {o["key"] for o in prompt.get("options", [])}
            if text not in keys:
                raise ValueError(f"{text!r} is not one of the offered options")

    def close(self) -> None:
        self.done = True
        self.io.close()


class SessionManager:
    """Owns the live conversations: creates them, routes answers, and reaps the abandoned ones."""

    def __init__(
        self,
        make_runner: Callable[[HttpIO], Runner],
        *,
        max_sessions: int = _MAX_SESSIONS,
        idle_timeout_s: float = _IDLE_TIMEOUT_S,
    ) -> None:
        self._make_runner = make_runner
        self._max_sessions = max_sessions
        self._idle_timeout_s = idle_timeout_s
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple[str, Turn]:
        """Start a new conversation and return its id and opening turn."""
        self._reap()
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise SessionBusy  # surfaced as 429 — too many live conversations
            session_id = uuid.uuid4().hex
            io = HttpIO()
            runner = self._make_runner(io)
            thread = threading.Thread(
                target=_run_session,
                args=(runner, io),
                name=f"cinegeist-{session_id[:8]}",
                daemon=True,
            )
            session = Session(id=session_id, io=io, thread=thread)
            self._sessions[session_id] = session
            thread.start()
        return session_id, session.first_turn()

    def answer(self, session_id: str, text: str) -> Turn:
        session = self._get(session_id)
        turn = session.answer(text)
        if turn.done:
            self._discard(session_id)
        return turn

    def end(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.close()
            self._discard(session_id)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def _get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def _discard(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _reap(self) -> None:
        """Close sessions idle past the timeout — a closed tab leaves one parked otherwise."""
        now = datetime.now(UTC)
        with self._lock:
            stale = [
                s
                for s in self._sessions.values()
                if (now - s.last_active).total_seconds() > self._idle_timeout_s
            ]
            for session in stale:
                self._sessions.pop(session.id, None)
        for session in stale:
            session.close()


def _run_session(runner: Runner, io: HttpIO) -> None:
    """The session thread body: run the whole conversation, then always emit a final turn."""
    try:
        runner(io)
    except SessionClosed:
        return  # reaped or closed — no one is waiting for a turn
    except BaseException as error:  # noqa: BLE001 — a crashed engine must not hang the client
        io.fail(f"The conversation ended unexpectedly: {error}")
        return
    io.finish()


def _serialize_presentation(presentation: Any, header: str) -> dict[str, Any]:
    """The picks block as JSON: each film with its title, year, reason, and wildcard flag."""

    def film(presented: Any) -> dict[str, Any]:
        f = presented.film
        return {
            "id": f.movie_id,
            "title": f.title,
            "year": f.year,
            "explanation": presented.explanation.text,
            "wildcard": presented.is_wildcard,
        }

    return {
        "type": "picks",
        "header": header,
        "picks": [film(p) for p in presentation.picks],
        "wildcard": film(presentation.wildcard) if presentation.wildcard else None,
        "pool_size": presentation.pool_size,
    }
