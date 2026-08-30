"""Unit tests for the OpenRouter client. All HTTP is mocked; nothing touches the network."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from cinegeist.config import Settings
from cinegeist.llm.client import (
    AuthError,
    ChatResult,
    LLMError,
    ModelUnavailableError,
    OpenRouterClient,
    RateLimitedError,
)

SECRET = "unit-test-key-abcdef"


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "model": "test-model",
        "api_key": SecretStr(SECRET),
        "max_retries": 2,
    }
    base.update(overrides)
    return Settings(**base)


def ok_body(model: str = "test-model", content: str = "hello there") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], **settings: object
) -> OpenRouterClient:
    return OpenRouterClient(
        make_settings(**settings),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,  # never actually wait in tests
    )


def test_chat_returns_the_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(200, json=ok_body())

    result = client_for(handler).chat([{"role": "user", "content": "hi"}])
    assert isinstance(result, ChatResult)
    assert result.text == "hello there"
    assert result.model == "test-model"
    assert result.finish_reason == "stop"


def test_authorization_header_carries_the_key() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json=ok_body())

    client_for(handler).chat([{"role": "user", "content": "hi"}])
    assert seen["auth"] == f"Bearer {SECRET}"


def test_missing_key_raises_auth_error_without_requesting() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request should be made without a key")

    client = OpenRouterClient(make_settings(api_key=None), transport=httpx.MockTransport(handler))
    with pytest.raises(AuthError):
        client.chat([{"role": "user", "content": "hi"}])


def test_retries_a_500_then_succeeds() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=ok_body())

    client = OpenRouterClient(
        make_settings(), transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.text == "hello there"
    assert calls["n"] == 2
    assert len(sleeps) == 1  # exactly one backoff between the two attempts


def test_persistent_429_exhausts_retries_and_raises_rate_limited() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    with pytest.raises(RateLimitedError):
        client_for(handler, max_retries=1).chat([{"role": "user", "content": "hi"}])


def test_404_raises_model_unavailable_immediately() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="no such model")

    with pytest.raises(ModelUnavailableError):
        client_for(handler, max_retries=3).chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # a 404 is fatal, not retried


def test_401_raises_auth_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    with pytest.raises(AuthError):
        client_for(handler).chat([{"role": "user", "content": "hi"}])


def test_failover_moves_to_the_next_model_on_429() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == "busy-model":
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=ok_body(model=model))

    result = client_for(handler, max_retries=0).chat_with_failover(
        [{"role": "user", "content": "hi"}], ["busy-model", "test-model"]
    )
    assert result.model == "test-model"


def test_failover_raises_the_last_error_when_all_models_fail() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    with pytest.raises(RateLimitedError):
        client_for(handler, max_retries=0).chat_with_failover(
            [{"role": "user", "content": "hi"}], ["a", "b"]
        )


def test_the_key_is_redacted_in_error_messages() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        # A provider that unhelpfully echoes the auth header back in an error body.
        return httpx.Response(400, text=f"bad request with Bearer {SECRET}")

    with pytest.raises(LLMError) as excinfo:
        client_for(handler).chat([{"role": "user", "content": "hi"}])
    assert SECRET not in str(excinfo.value)
    assert "***REDACTED***" in str(excinfo.value)


def test_malformed_success_body_raises_llm_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})  # no message to read

    with pytest.raises(LLMError):
        client_for(handler).chat([{"role": "user", "content": "hi"}])
