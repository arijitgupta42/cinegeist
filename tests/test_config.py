"""Unit tests for settings loading. No network, no real environment."""

from __future__ import annotations

from pathlib import Path

from cinegeist.config import (
    DEFAULT_API_BASE,
    Settings,
    _parse_dotenv,
    load_settings,
    redact_secrets,
)


def _missing(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist.toml"


def test_defaults_when_nothing_is_set(tmp_path: Path) -> None:
    settings = load_settings(env={}, config_file=_missing(tmp_path))
    assert settings.model is None
    assert settings.api_base_url == DEFAULT_API_BASE
    assert settings.request_timeout == 30.0
    assert settings.max_retries == 3
    assert settings.api_key is None
    assert settings.has_api_key is False


def test_file_supplies_values(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "from-file"\nrequest_timeout = 12.5\n', encoding="utf-8")
    settings = load_settings(env={}, config_file=cfg)
    assert settings.model == "from-file"
    assert settings.request_timeout == 12.5


def test_env_beats_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "from-file"\n', encoding="utf-8")
    settings = load_settings(env={"CINEGEIST_MODEL": "from-env"}, config_file=cfg)
    assert settings.model == "from-env"


def test_flag_beats_env(tmp_path: Path) -> None:
    settings = load_settings(
        overrides={"model": "from-flag"},
        env={"CINEGEIST_MODEL": "from-env"},
        config_file=_missing(tmp_path),
    )
    assert settings.model == "from-flag"


def test_none_override_does_not_clobber(tmp_path: Path) -> None:
    # An unset CLI flag arrives as None and must not wipe a lower-precedence value.
    settings = load_settings(
        overrides={"model": None},
        env={"CINEGEIST_MODEL": "from-env"},
        config_file=_missing(tmp_path),
    )
    assert settings.model == "from-env"


def test_api_key_comes_from_env_only(tmp_path: Path) -> None:
    secret = "unit-test-secret-value-123"
    settings = load_settings(env={"OPENROUTER_API_KEY": secret}, config_file=_missing(tmp_path))
    assert settings.has_api_key is True
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == secret


def test_api_key_is_not_read_from_the_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('api_key = "should-be-ignored"\n', encoding="utf-8")
    settings = load_settings(env={}, config_file=cfg)
    assert settings.api_key is None


def test_api_key_never_appears_in_repr_or_str(tmp_path: Path) -> None:
    secret = "unit-test-secret-value-123"
    settings = load_settings(env={"OPENROUTER_API_KEY": secret}, config_file=_missing(tmp_path))
    assert secret not in repr(settings)
    assert secret not in str(settings)


def test_redact_masks_the_key(tmp_path: Path) -> None:
    secret = "unit-test-secret-value-123"
    settings = load_settings(env={"OPENROUTER_API_KEY": secret}, config_file=_missing(tmp_path))
    masked = settings.redact(f"Authorization: Bearer {secret} failed")
    assert secret not in masked
    assert "***REDACTED***" in masked


def test_tmdb_credentials_come_from_env(tmp_path: Path) -> None:
    settings = load_settings(
        env={"TMDB_API_KEY": "tmdb-v3", "TMDB_ACCESS_TOKEN": "tmdb-v4"},
        config_file=_missing(tmp_path),
    )
    assert settings.has_tmdb_auth is True
    assert settings.tmdb_api_key is not None
    assert settings.tmdb_api_key.get_secret_value() == "tmdb-v3"
    assert settings.tmdb_access_token.get_secret_value() == "tmdb-v4"


def test_no_tmdb_auth_by_default(tmp_path: Path) -> None:
    settings = load_settings(env={}, config_file=_missing(tmp_path))
    assert settings.has_tmdb_auth is False


def test_redact_masks_the_tmdb_key_and_hides_it_from_repr(tmp_path: Path) -> None:
    secret = "tmdb-secret-xyz"
    settings = load_settings(env={"TMDB_API_KEY": secret}, config_file=_missing(tmp_path))
    assert secret not in repr(settings)
    assert secret not in settings.redact(f"GET /movie/1?api_key={secret}")


def test_unknown_config_keys_are_ignored(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "m"\nnonsense = 42\n', encoding="utf-8")
    settings = load_settings(env={}, config_file=cfg)
    assert settings.model == "m"


def test_settings_are_immutable() -> None:
    settings = Settings()
    try:
        settings.model = "changed"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Settings should be frozen")


def test_redact_secrets_helper() -> None:
    assert redact_secrets("token=abc more", "abc") == "token=***REDACTED*** more"
    assert redact_secrets("unchanged", "") == "unchanged"


def test_parse_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "OPENROUTER_API_KEY = spaced-value\n"
        'export CINEGEIST_MODEL="quoted-model"\n'
        "\n"
        "MALFORMED_LINE\n",
        encoding="utf-8",
    )
    parsed = _parse_dotenv(env_file)
    assert parsed["OPENROUTER_API_KEY"] == "spaced-value"
    assert parsed["CINEGEIST_MODEL"] == "quoted-model"
    assert "MALFORMED_LINE" not in parsed
