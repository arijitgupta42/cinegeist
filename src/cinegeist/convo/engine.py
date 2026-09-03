"""The conversation state machine: greet → seed → adaptive probes → constrain → present → feedback.

This is where session 3's pieces (the profile store, the seed questions, free-text extraction, the
information-gain probes) and session 4's recommender finally run as one conversation. The engine
holds no taste logic of its own — every decision it makes is deferred to the deterministic layers
(``probes.choose_probe``, ``probes.should_stop``, ``present.present``) or to the profile it rebuilds
from the event log after every answer. Its job is orchestration and honesty: ask the next thing,
record what came back as one more immutable event, and offer the escape hatch on every single turn.

All input and output goes through :class:`ConversationIO`, so the terminal is one implementation and
a scripted test double is another. The engine never touches ``print`` or ``input``.

Two shapes of run, one code path:

* **Online** phrases questions with the LLM (one call per turn, the budget), reads free-text seed
  answers through extraction, and reranks/explains the picks. Each LLM touch degrades to the
  deterministic path on failure, so a rate-limited turn is a plainer turn, not a broken one.
* **Offline** (``--offline``, or no API key) drops the LLM entirely: no free-text seed, fixed probe
  phrasing, templated explanations. It is the same recommender the browser demo is — click-driven
  and honest — proof the maths stands on its own.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from ..catalog import db
from ..config import Settings
from ..feedback import VERDICTS, record_feedback
from ..llm.client import LLMError, OpenRouterClient
from ..llm.prompt_loader import load_prompt
from ..profile import store, update
from ..profile.model import PreferenceEvent, TasteProfile
from ..recommend import present as present_mod
from ..recommend import retrieve as retrieve_mod
from ..recommend import score as score_mod
from . import probes as probes_mod
from .extract import extract_signals
from .resolve import resolve_mentions
from .seed import SEED_QUESTIONS, events_from_resolutions

# How strong a signal one pair choice is, as a signed axis answer. Moderate: a single click is real
# evidence but not as decisive as watching a film (see feedback.py's heavier post-watch weights).
_PAIR_VALUE = 0.6
# A returning user counts as confident enough to jump straight to picks past this evidence mass.
_RETURNING_CONFIDENCE = 2.0
# Cap on "show me three more" pages, so the present loop can't spin forever.
_MAX_MORE_PAGES = 4


@dataclass(frozen=True)
class Choice:
    """One selectable option: ``key`` is what the engine reads back, ``label`` is what's shown."""

    key: str
    label: str


class ConversationIO(Protocol):
    """The engine's whole view of the user. The terminal is one of these; a test script another."""

    def say(self, text: str) -> None:
        """Show a line of narration."""

    def ask_text(self, prompt: str) -> str:
        """Ask an open question and return the raw answer (may be empty)."""

    def ask_choice(self, prompt: str, options: Sequence[Choice]) -> str:
        """Ask the user to pick one option; return the chosen option's ``key``."""

    def show_presentation(self, presentation: present_mod.Presentation, *, header: str) -> None:
        """Render the picks and the wildcard with their explanations."""


# The escape hatch, offered on every turn without exception (plan.md §5.3).
_ESCAPE = Choice("__stop__", "just show me something")


def _looks_like_question(text: str) -> bool:
    """Whether a phrased probe is usable, or the model returned a truncated stub.

    A real pair question names two films and asks something, so it carries a question mark and more
    than a couple of words. A reasoning model given a tight token budget can spend it thinking and
    return an empty reply or a bare "Which" — reject those so the caller falls back to the fixed
    wording rather than show the user a fragment.
    """
    return "?" in text and len(text) >= 15


class Engine:
    """Drives one conversation to a set of recommendations, recording evidence as it goes."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        matrix: np.ndarray,
        io: ConversationIO,
        settings: Settings,
        *,
        client: OpenRouterClient | None = None,
        models: Sequence[str] = (),
        offline: bool = False,
        user_id: str = store.DEFAULT_USER,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        self.conn = conn
        self.matrix = matrix
        self.io = io
        self.settings = settings
        self.client = client
        self.models = tuple(models)
        # Offline unless we genuinely have a client and models to fail over across.
        self.offline = offline or client is None or not self.models
        self.user_id = user_id
        self.session_id = session_id or uuid.uuid4().hex
        self.now = now
        self._stop_requested = False
        self._skip_to_present = False
        self.constraints = retrieve_mod.Constraints()

    # -- the whole run ----------------------------------------------------------------

    def run(self) -> None:
        """Hold the conversation from greeting to picks, honouring the escape hatch throughout."""
        self._greet()
        if not self._stop_requested and not self._skip_to_present:
            if not self.offline:
                self._seed()
            self._adaptive_loop()
        if not self._skip_to_present:
            self._constrain()
        self._present_loop()

    # -- profile access ---------------------------------------------------------------

    def _profile(self) -> TasteProfile:
        return update.compute_profile(self.conn, self.matrix, user_id=self.user_id, now=self.now)

    def _profile_vector(self) -> np.ndarray:
        vector, _weight = update.load_vector(
            self.conn, self.matrix, user_id=self.user_id, now=self.now
        )
        return vector

    def _record(self, events: list[PreferenceEvent]) -> None:
        if events:
            store.append_events(self.conn, events, now=self.now)

    # -- greet ------------------------------------------------------------------------

    def _greet(self) -> None:
        profile = self._profile()
        if profile.is_empty:
            self.io.say(
                "Let's find you something to watch. I'll ask you to react to a few real films "
                "rather than describe what you like — people are much better at the first."
            )
            return

        drawn = ", ".join(axis.name for axis in profile.affinities[:3]) or "a few things"
        self.io.say(f"Welcome back. Last time you leaned toward {drawn}.")
        self._maybe_ask_last_watch()

        if profile.total_weight >= _RETURNING_CONFIDENCE:
            choice = self.io.ask_choice(
                "Want to jump straight to tonight's picks, or refine a bit first?",
                [Choice("picks", "just show me picks"), Choice("refine", "ask me a few things")],
            )
            if choice == "picks":
                self._skip_to_present = True

    def _maybe_ask_last_watch(self) -> None:
        """If we remember last session's picks, ask what came of one — that's the feedback loop."""
        raw = db.get_state(self.conn, self._recs_key())
        if not raw:
            return
        try:
            recs = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not recs:
            return
        top = recs[0]
        options = [Choice(v.key, v.label) for v in VERDICTS] + [Choice("skip", "haven't seen it")]
        choice = self.io.ask_choice(
            f"Did you get to “{top.get('title', 'my last pick')}”?", options
        )
        if choice != "skip":
            record_feedback(
                self.conn, int(top["id"]), choice, session_id=self.session_id, now=self.now
            )

    # -- seed (online only: free text needs the extractor) ----------------------------

    def _seed(self) -> None:
        for question in SEED_QUESTIONS:
            if question.style != "free_text":
                continue  # the pair question is just the first adaptive probe
            answer = self.io.ask_text(question.prompt)
            if probes_mod.wants_to_stop(answer):
                self._stop_requested = True
                return
            if answer.strip():
                self._ingest_free_text(answer, context=question.key)

    def _ingest_free_text(self, answer: str, *, context: str) -> None:
        assert self.client is not None  # only reached online
        extraction = extract_signals(self.client, answer, self.models, context=context)
        resolutions = resolve_mentions(self.conn, extraction.mentions)
        resolved = [self._disambiguate(mention, res) for mention, res in resolutions]
        events = events_from_resolutions(
            [(m, r) for m, r in resolved if r is not None], session_id=self.session_id
        )
        events.extend(self._axis_events(extraction.axis_signals))
        self._record(events)

    def _disambiguate(self, mention, resolution):
        """Turn an ambiguous title into a resolved one by asking the user which film they meant."""
        if resolution.status != "ambiguous" or not resolution.candidates:
            return (mention, resolution if resolution.is_resolved else None)
        options = [
            Choice(str(c.movie_id), f"{c.title}" + (f" ({c.year})" if c.year else ""))
            for c in resolution.candidates
        ]
        options.append(Choice("none", "none of these"))
        picked = self.io.ask_choice(f"Which “{mention.title}” did you mean?", options)
        if picked == "none":
            return (mention, None)
        chosen = next(c for c in resolution.candidates if str(c.movie_id) == picked)
        from .resolve import Resolution

        return (
            mention,
            Resolution(query=mention.title, year=mention.year, status="resolved", match=chosen),
        )

    def _axis_events(self, axis_signals) -> list[PreferenceEvent]:
        """Free-text taste descriptors ("too slow") become axis answers where a tag name matches."""
        if not axis_signals:
            return []
        by_name = self._tag_id_by_name()
        events: list[PreferenceEvent] = []
        for signal in axis_signals:
            tag_id = by_name.get(signal.axis.strip().lower())
            if tag_id is None or signal.value == 0.0:
                continue
            events.append(
                PreferenceEvent.axis_answer(
                    tag_id,
                    max(-1.0, min(1.0, signal.value)),
                    evidence=signal.quote,
                    session_id=self.session_id,
                )
            )
        return events

    # -- adaptive probes --------------------------------------------------------------

    def _adaptive_loop(self) -> None:
        asked_positions: set[int] = set()
        probe_films: set[int] = set()
        top5_history: list[list[int]] = []
        turn = 0
        while True:
            profile_vector = self._profile_vector()
            top5, top_scores = self._ranking(profile_vector)
            top5_history.append(top5)
            decision = probes_mod.should_stop(
                turn=turn,
                top5_history=top5_history,
                top_scores=top_scores,
                user_requested=self._stop_requested,
            )
            if decision.stop:
                return

            seen = retrieve_mod.seen_movie_ids(self.conn, self.user_id)
            probe = probes_mod.choose_probe(
                self.conn,
                self.matrix,
                profile_vector,
                excluded_movie_ids=frozenset(probe_films | seen),
                asked_positions=frozenset(asked_positions),
            )
            if probe is None:
                return
            if not self._ask_probe(probe, asked_positions, probe_films):
                return  # user hit the escape hatch
            turn += 1

    def _ask_probe(self, probe: probes_mod.Probe, asked: set[int], films: set[int]) -> bool:
        """Ask one pair question and record the answer. Returns False if the user asked to stop."""
        question = self._phrase_probe(probe)
        options = [
            Choice("high", probe.film_high.title),
            Choice("low", probe.film_low.title),
            Choice("neither", "I haven't seen either"),
            _ESCAPE,
        ]
        choice = self.io.ask_choice(question, options)
        asked.add(probe.axis_position)
        films.update({probe.film_high.movie_id, probe.film_low.movie_id})

        if choice == _ESCAPE.key:
            self._stop_requested = True
            return False
        if choice in ("high", "low"):
            tag_id = self._tag_id_by_position().get(probe.axis_position)
            if tag_id is not None:
                chosen = probe.film_high if choice == "high" else probe.film_low
                value = _PAIR_VALUE if choice == "high" else -_PAIR_VALUE
                self._record(
                    [
                        PreferenceEvent.axis_answer(
                            tag_id,
                            value,
                            evidence=f"chose {chosen.title} over the other",
                            session_id=self.session_id,
                        )
                    ]
                )
        # "neither" is obscurity signal, not an axis answer — recorded as nothing, just moves on.
        return True

    def _phrase_probe(self, probe: probes_mod.Probe) -> str:
        """The pair question: LLM-phrased online, the fixed wording offline or on failure."""
        if self.offline or self.client is None:
            return probe.question
        prompt = load_prompt("phrase")
        user = (
            f"Axis: {probe.axis_name}\n"
            f'Films: "{probe.film_high.title}" and "{probe.film_low.title}"'
        )
        try:
            reply = self.client.chat_with_failover(
                [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                self.models,
                temperature=0.7,
                # Room for a reasoning model to think and still emit the whole question: too tight a
                # cap makes it spend the budget reasoning and return a truncated stub.
                max_tokens=200,
            )
        except LLMError:
            return probe.question
        phrased = " ".join(reply.text.split()).strip().strip('"')
        return phrased if _looks_like_question(phrased) else probe.question

    def _ranking(self, profile_vector: np.ndarray) -> tuple[list[int], list[float]]:
        """A quick deterministic top ranking of the pool for the stopping rules (no MMR, no LLM)."""
        seen = retrieve_mod.seen_movie_ids(self.conn, self.user_id)
        candidates = retrieve_mod.retrieve(self.conn, retrieve_mod.Constraints(exclude_ids=seen))
        if not candidates:
            return [], []
        vectors = score_mod.candidate_vectors(candidates, self.matrix)
        scored = score_mod.score_pool(candidates, vectors, profile_vector)
        scored.sort(key=lambda film: film.score, reverse=True)
        return [f.movie_id for f in scored[:5]], [f.score for f in scored[:10]]

    # -- constrain --------------------------------------------------------------------

    def _constrain(self) -> None:
        """One light pass of session intent: how much time, and subtitles or not (plan.md §5.1)."""
        time_choice = self.io.ask_choice(
            "Roughly how much time tonight?",
            [
                Choice("short", "under 90 minutes"),
                Choice("standard", "a couple of hours is fine"),
                Choice("any", "no limit"),
                _ESCAPE,
            ],
        )
        if time_choice == _ESCAPE.key:
            self._stop_requested = True  # honoured, but we still present with what we have
        max_runtime = 90 if time_choice == "short" else None

        languages: frozenset[str] | None = None
        if time_choice != _ESCAPE.key:
            subs = self.io.ask_choice(
                "Up for subtitles tonight?",
                [Choice("yes", "subtitles are fine"), Choice("no", "English, no subtitles")],
            )
            if subs == "no":
                languages = frozenset({"en"})

        self.constraints = retrieve_mod.Constraints(max_runtime=max_runtime, languages=languages)

    # -- present ----------------------------------------------------------------------

    def _present_loop(self) -> None:
        shown: set[int] = set()
        profile = self._profile()
        for page in range(_MAX_MORE_PAGES):
            presentation = present_mod.present(
                self.conn,
                self.matrix,
                profile,
                client=self.client,
                models=self.models,
                constraints=self.constraints,
                also_exclude=frozenset(shown),
                offline=self.offline,
            )
            if presentation.is_empty:
                if shown:
                    self.io.say("That's everything the catalog has near your taste tonight.")
                else:
                    self.io.say(
                        "Nothing in the catalog matches those constraints — try relaxing the "
                        "time or subtitle limit, or build a fuller catalog."
                    )
                return

            header = "Your picks" if page == 0 else "A few more"
            if presentation.pool_size < 5:
                self.io.say(
                    f"Only {presentation.pool_size} film(s) matched — the catalog is thin here, "
                    "so these are the closest it gets rather than a confident match."
                )
            self.io.show_presentation(presentation, header=header)
            for film in presentation.all_films:
                shown.add(film.film.movie_id)
            self._remember_recommendations(presentation)

            choice = self.io.ask_choice(
                "What now?",
                [Choice("more", "show me three more"), Choice("done", "that's great, thanks")],
            )
            if choice == "done":
                return
        self.io.say("That's a good spread for tonight — enjoy.")

    def _remember_recommendations(self, presentation: present_mod.Presentation) -> None:
        """Persist this page's picks so a future session can ask how they landed (feedback loop)."""
        recs = [
            {"id": film.film.movie_id, "title": film.film.title} for film in presentation.all_films
        ]
        db.set_state(self.conn, self._recs_key(), json.dumps(recs))

    # -- small catalog lookups --------------------------------------------------------

    def _recs_key(self) -> str:
        return f"last_recs:{self.user_id}"

    def _tag_id_by_position(self) -> dict[int, int]:
        return {
            row["position"]: row["tag_id"]
            for row in self.conn.execute("SELECT tag_id, position FROM genome_tags")
        }

    def _tag_id_by_name(self) -> dict[str, int]:
        return {
            row["name"].strip().lower(): row["tag_id"]
            for row in self.conn.execute("SELECT tag_id, name FROM genome_tags")
        }
