"""A small OpenRouter chat client.

Responsibilities, and nothing more:

* make one chat-completions request for a given model;
* retry transient failures (timeouts, 429, 5xx) with exponential backoff plus jitter;
* fail over to the next model in a list when one is rate-limited or unavailable;
* keep the API key out of every error message it raises.

Budget rule (see ``plan.md``): one LLM call per conversational turn. This client makes a
single request per ``chat`` call — retries reuse the same logical call, and failover only
happens when a model refuses to answer.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx

from ..config import Settings

# Statuses worth retrying: rate limiting and the transient server errors.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Backoff shape. Small and bounded — free tiers punish hammering, not politeness.
_BASE_BACKOFF_SECONDS = 0.5
_MAX_BACKOFF_SECONDS = 8.0

Message = dict[str, str]


class LLMError(RuntimeError):
    """Base class for every client failure. Its message is always key-redacted."""


class AuthError(LLMError):
    """The API key is missing or was rejected (401/403). Not worth failing over."""


class RateLimitedError(LLMError):
    """The model is rate-limited (429). The caller may fail over to another model."""


class ModelUnavailableError(LLMError):
    """The model is missing or its provider is failing (404, or 5xx after retries)."""


@dataclass(frozen=True)
class ChatResult:
    """The useful part of a successful response."""

    text: str
    model: str
    finish_reason: str | None = None


class OpenRouterClient:
    """Talks to OpenRouter's chat-completions endpoint.

    Inject ``transport`` (an ``httpx.MockTransport``) in tests, and ``sleep``/``rng`` to
    make backoff deterministic. Nothing here touches the network unless a real transport
    is used.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._base_url = settings.api_base_url.rstrip("/")
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                timeout=settings.request_timeout,
                transport=transport,
                headers=self._default_headers(),
            )
            self._owns_client = True

    # -- public API ---------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Send one chat request. Retries transient errors; raises a typed LLMError."""
        model = model or self._settings.model
        if not model:
            raise LLMError("No model specified: pass model= or configure one.")
        if not self._settings.has_api_key:
            raise AuthError("OPENROUTER_API_KEY is not set.")

        payload: dict[str, object] = {"model": model, "messages": list(messages)}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return self._post_with_retries(payload, model)

    def chat_with_failover(
        self,
        messages: Sequence[Message],
        models: Sequence[str],
        **kwargs: object,
    ) -> ChatResult:
        """Try each model in order, moving on when one is rate-limited or unavailable."""
        candidates = list(models)
        if not candidates:
            raise LLMError("No models to try.")
        last_error: LLMError | None = None
        for candidate in candidates:
            try:
                return self.chat(messages, model=candidate, **kwargs)  # type: ignore[arg-type]
            except (RateLimitedError, ModelUnavailableError) as error:
                last_error = error
        assert last_error is not None  # the loop ran at least once
        raise last_error

    def redact(self, text: str) -> str:
        """Mask the API key (and any other secret) wherever it appears in ``text``.

        Public so callers that build their own error strings around a response — the extractor's
        retry nudge, for one — can keep the key out of them too (hard rule 5).
        """
        return self._redact(text)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------------

    def _default_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            # OpenRouter uses these for its app-attribution rankings; both are optional.
            "HTTP-Referer": "https://github.com/arijitgupta42/cinegeist",
            "X-Title": "cinegeist",
        }
        if self._settings.has_api_key:
            assert self._settings.api_key is not None
            headers["Authorization"] = f"Bearer {self._settings.api_key.get_secret_value()}"
        return headers

    def _redact(self, text: str) -> str:
        return self._settings.redact(text)

    def _backoff_seconds(self, attempt: int) -> float:
        ceiling = min(_MAX_BACKOFF_SECONDS, _BASE_BACKOFF_SECONDS * (2**attempt))
        # Jitter in [0.5, 1.0] of the ceiling avoids a thundering herd of retries.
        return ceiling * (0.5 + 0.5 * self._rng.random())

    def _post_with_retries(self, payload: dict[str, object], model: str) -> ChatResult:
        url = f"{self._base_url}/chat/completions"
        last_error: LLMError | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = self._client.post(url, json=payload)
            except httpx.TimeoutException as error:
                last_error = LLMError(self._redact(f"Request timed out: {error}"))
            except httpx.HTTPError as error:
                last_error = LLMError(self._redact(f"Request failed: {error}"))
            else:
                terminal = self._handle_status(response, model)
                if isinstance(terminal, ChatResult):
                    return terminal
                last_error = terminal  # retryable error; maybe loop again

            if attempt < self._settings.max_retries:
                self._sleep(self._backoff_seconds(attempt))

        assert last_error is not None
        raise last_error

    def _handle_status(self, response: httpx.Response, model: str) -> ChatResult | LLMError:
        """Return a ChatResult on success, a retryable LLMError to loop, or raise a fatal one."""
        status = response.status_code
        if status == 200:
            return self._parse(response, model)
        if status in (401, 403):
            raise AuthError(self._redact(f"Authentication failed ({status})."))
        if status == 404:
            raise ModelUnavailableError(self._redact(f"Model '{model}' not found ({status})."))
        if status == 429:
            return RateLimitedError(self._redact(f"Rate limited on '{model}' ({status})."))
        if status in _RETRYABLE_STATUS:
            return ModelUnavailableError(self._redact(f"Server error from '{model}' ({status})."))
        raise LLMError(self._redact(f"Unexpected status {status}: {response.text}"))

    def _parse(self, response: httpx.Response, model: str) -> ChatResult:
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise LLMError(self._redact(f"Malformed response from '{model}': {error}")) from error
        return ChatResult(
            text=text,
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason"),
        )
