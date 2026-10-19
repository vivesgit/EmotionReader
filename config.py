"""Centralised configuration: the only module that reads environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
MODELS_DIR = PROJECT_ROOT / "models"


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default="", repr=False)
    gpt_live_model: str = "gpt-realtime-2.1"
    gpt_live_voice: str = "marin"
    response_language: str = "es"
    request_timeout_seconds: float = 30.0
    emotion_window_seconds: float = 5.0
    camera_index: int = 0

    @property
    def api_key_configured(self) -> bool:
        return bool(self.openai_api_key)

    def __str__(self) -> str:
        key_state = "configurada" if self.api_key_configured else "NO configurada"
        return (
            f"OPENAI_API_KEY: {key_state}\n"
            f"GPT_LIVE_MODEL: {self.gpt_live_model}\n"
            f"GPT_LIVE_VOICE: {self.gpt_live_voice}\n"
            f"RESPONSE_LANGUAGE: {self.response_language}\n"
            f"REQUEST_TIMEOUT_SECONDS: {self.request_timeout_seconds:g}\n"
            f"EMOTION_WINDOW_SECONDS: {self.emotion_window_seconds:g}\n"
            f"CAMERA_INDEX: {self.camera_index}"
        )


def _positive_float(values: dict[str, str], name: str, default: float) -> float:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} debe ser un número, no {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} debe ser mayor que 0, no {raw!r}")
    return value


def _non_negative_int(values: dict[str, str], name: str, default: int) -> int:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} debe ser un entero, no {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} no puede ser negativo, no {raw!r}")
    return value


def _text(values: dict[str, str], name: str, default: str) -> str:
    return values.get(name, "").strip() or default


def load_settings(env_file: Path | str | None = DEFAULT_ENV_FILE) -> Settings:
    """Build Settings from the .env file, with real environment variables taking precedence."""
    values: dict[str, str] = {}
    if env_file is not None and Path(env_file).is_file():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ)

    defaults = Settings()
    return Settings(
        openai_api_key=_text(values, "OPENAI_API_KEY", defaults.openai_api_key),
        gpt_live_model=_text(values, "GPT_LIVE_MODEL", defaults.gpt_live_model),
        gpt_live_voice=_text(values, "GPT_LIVE_VOICE", defaults.gpt_live_voice),
        response_language=_text(values, "RESPONSE_LANGUAGE", defaults.response_language),
        request_timeout_seconds=_positive_float(
            values, "REQUEST_TIMEOUT_SECONDS", defaults.request_timeout_seconds
        ),
        emotion_window_seconds=_positive_float(
            values, "EMOTION_WINDOW_SECONDS", defaults.emotion_window_seconds
        ),
        camera_index=_non_negative_int(values, "CAMERA_INDEX", defaults.camera_index),
    )


if __name__ == "__main__":
    print(load_settings())
