"""Unit tests for the CLI. No network: the client and registry are stubbed."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from typer.testing import CliRunner

from cinegeist import cli
from cinegeist.llm.client import ChatResult, LLMError

runner = CliRunner()


def all_text(result: object) -> str:
    """Stdout plus stderr, however this click version chose to separate them."""
    text = getattr(result, "output", "") or ""
    try:
        text += getattr(result, "stderr", "") or ""
    except (ValueError, AttributeError):
        pass
    return text


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep the developer's real env/config/.env out of the CLI under test."""
    monkeypatch.chdir(tmp_path)  # no stray .env in the working directory
    monkeypatch.setenv("CINEGEIST_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("CINEGEIST_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("OPENROUTER_API_KEY", "CINEGEIST_MODEL", "CINEGEIST_API_BASE"):
        monkeypatch.delenv(var, raising=False)


class FakeClient:
    """Stands in for OpenRouterClient; records the models it was asked to try."""

    last_models: list[str] = []

    def __init__(self, settings: object, **_kwargs: object) -> None:
        self.settings = settings

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def chat_with_failover(
        self, messages: Sequence[dict], models: Sequence[str], **_kwargs: object
    ) -> ChatResult:
        FakeClient.last_models = list(models)
        return ChatResult(text="pong", model=models[0])


class FailingClient(FakeClient):
    def chat_with_failover(self, *args: object, **kwargs: object) -> ChatResult:
        raise LLMError("provider exploded (key ***REDACTED***)")


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "cinegeist 0.1.0" in all_text(result)


def test_models_lists_free_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "free_models", lambda *a, **k: ["meta/a:free", "z/b:free"])
    result = runner.invoke(cli.app, ["models", "--free"])
    assert result.exit_code == 0
    text = all_text(result)
    assert "meta/a:free" in text
    assert "z/b:free" in text


def test_ask_prints_the_reply_and_fails_over_across_free_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(cli, "OpenRouterClient", FakeClient)
    monkeypatch.setattr(cli, "free_models", lambda *a, **k: ["free/1", "free/2"])
    result = runner.invoke(cli.app, ["ask", "hi"])
    assert result.exit_code == 0
    assert "pong" in all_text(result)
    assert FakeClient.last_models == ["free/1", "free/2"]


def test_ask_pins_the_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(cli, "OpenRouterClient", FakeClient)
    # If the flag is honored, discovery is never consulted.
    monkeypatch.setattr(cli, "free_models", lambda *a, **k: pytest.fail("should not auto-select"))
    result = runner.invoke(cli.app, ["ask", "hi", "--model", "pinned/model"])
    assert result.exit_code == 0
    assert FakeClient.last_models == ["pinned/model"]


def test_ask_without_a_key_explains_and_exits_nonzero() -> None:
    result = runner.invoke(cli.app, ["ask", "hi"])
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in all_text(result)


def test_ask_reports_an_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    monkeypatch.setattr(cli, "OpenRouterClient", FailingClient)
    monkeypatch.setattr(cli, "free_models", lambda *a, **k: ["free/1"])
    result = runner.invoke(cli.app, ["ask", "hi"])
    assert result.exit_code == 1
    assert "provider exploded" in all_text(result)


def test_config_shows_key_as_set_without_printing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "unit-test-secret-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    result = runner.invoke(cli.app, ["config"])
    assert result.exit_code == 0
    text = all_text(result)
    assert "set (redacted)" in text
    assert secret not in text


def test_config_reports_no_key_and_the_chosen_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CINEGEIST_MODEL", "some/model:free")
    result = runner.invoke(cli.app, ["config"])
    assert result.exit_code == 0
    text = all_text(result)
    assert "not set" in text
    assert "some/model:free" in text
