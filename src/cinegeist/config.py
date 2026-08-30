"""Settings and configuration loading.

Precedence, highest wins:

    CLI flag  >  CINEGEIST_* env var  >  ~/.config/cinegeist/config.toml  >  built-in default

The OpenRouter API key is special: it is read from the ``OPENROUTER_API_KEY`` environment
variable only. It is never read from the config file, never written to disk, and is held
as a :class:`pydantic.SecretStr` so it stays out of ``repr``/``str`` output, logs, and
tracebacks. Use :meth:`Settings.redact` before printing any text that might contain it.

For convenience a ``.env`` file in the current directory is loaded on a real run, but a
value already present in the real environment always wins over the file.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

APP_NAME = "cinegeist"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
REDACTION = "***REDACTED***"

# Config-file / env keys we accept. Anything else in the file is ignored rather than
# allowed to crash startup. The API key is deliberately absent — env only.
_ALLOWED_KEYS = frozenset({"model", "api_base_url", "request_timeout", "max_retries"})


def config_dir() -> Path:
    """Directory holding ``config.toml`` (``~/.config/cinegeist`` unless overridden)."""
    override = os.environ.get("CINEGEIST_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".config" / APP_NAME


def config_path() -> Path:
    """Path to the user's ``config.toml`` (may not exist)."""
    return config_dir() / "config.toml"


def cache_dir() -> Path:
    """Directory for cached data such as the free-model list."""
    override = os.environ.get("CINEGEIST_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / APP_NAME


def redact_secrets(text: str, *secrets: str) -> str:
    """Replace every occurrence of each non-empty secret in ``text`` with a fixed mask."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTION)
    return text


class Settings(BaseModel):
    """The effective configuration for one run. Immutable; build a new one to change it."""

    # `protected_namespaces=()` lets us use a field literally named `model` without a
    # pydantic warning; `frozen=True` makes an accidental in-place mutation an error.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str | None = Field(
        default=None,
        description="OpenRouter model id to use, or None to auto-pick a free one.",
    )
    api_base_url: str = DEFAULT_API_BASE
    request_timeout: float = 30.0
    max_retries: int = 3
    # Read from the environment only; hidden from repr so it never leaks into logs.
    api_key: SecretStr | None = Field(default=None, repr=False)

    @property
    def has_api_key(self) -> bool:
        """True when a non-empty API key is present in the environment."""
        return self.api_key is not None and bool(self.api_key.get_secret_value())

    def redact(self, text: str) -> str:
        """Mask this run's API key wherever it appears in ``text``."""
        if self.api_key is not None:
            return redact_secrets(text, self.api_key.get_secret_value())
        return text


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` ``.env`` file. No dependency, no interpolation."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return {key: value for key, value in data.items() if key in _ALLOWED_KEYS}


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if value := env.get("CINEGEIST_MODEL"):
        out["model"] = value
    if value := env.get("CINEGEIST_API_BASE"):
        out["api_base_url"] = value
    if value := env.get("CINEGEIST_TIMEOUT"):
        out["request_timeout"] = float(value)
    if value := env.get("CINEGEIST_MAX_RETRIES"):
        out["max_retries"] = int(value)
    return out


def load_settings(
    overrides: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    config_file: Path | None = None,
) -> Settings:
    """Build :class:`Settings` by applying the documented precedence.

    Passing ``env`` and ``config_file`` explicitly (as the tests do) keeps this pure and
    deterministic; leaving them ``None`` reads the real environment, a local ``.env``, and
    the user's ``config.toml``.
    """
    if env is None:
        dotenv = _parse_dotenv(Path(".env"))
        env = {**dotenv, **os.environ}  # a real environment variable beats the .env file
    if config_file is None:
        config_file = config_path()

    merged: dict[str, Any] = {}
    merged.update(_read_config_file(config_file))  # file beats defaults
    merged.update(_env_overrides(env))  # env beats file
    if overrides:  # a flag beats env, but only when it was actually given
        merged.update({key: value for key, value in overrides.items() if value is not None})

    key = env.get(API_KEY_ENV)
    if key:
        merged["api_key"] = SecretStr(key)

    return Settings(**merged)
