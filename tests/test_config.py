import pytest

import config
from config import ConfigError, Settings, load_settings

ENV_NAMES = [
    "OPENAI_API_KEY",
    "GPT_LIVE_MODEL",
    "GPT_LIVE_VOICE",
    "RESPONSE_LANGUAGE",
    "REQUEST_TIMEOUT_SECONDS",
    "EMOTION_WINDOW_SECONDS",
    "CAMERA_INDEX",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_env_file(tmp_path):
    settings = load_settings(tmp_path / "missing.env")
    assert settings == Settings()
    assert settings.gpt_live_model == "gpt-realtime-2.1"
    assert settings.emotion_window_seconds == 5
    assert not settings.api_key_configured


def test_reads_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-test\nGPT_LIVE_VOICE=cedar\nEMOTION_WINDOW_SECONDS=8\nCAMERA_INDEX=1\n"
    )
    settings = load_settings(env_file)
    assert settings.openai_api_key == "sk-test"
    assert settings.gpt_live_voice == "cedar"
    assert settings.emotion_window_seconds == 8
    assert settings.camera_index == 1


def test_environment_overrides_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GPT_LIVE_MODEL=from-file\n")
    monkeypatch.setenv("GPT_LIVE_MODEL", "from-env")
    assert load_settings(env_file).gpt_live_model == "from-env"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMOTION_WINDOW_SECONDS", "abc"),
        ("EMOTION_WINDOW_SECONDS", "0"),
        ("REQUEST_TIMEOUT_SECONDS", "-1"),
        ("CAMERA_INDEX", "1.5"),
        ("CAMERA_INDEX", "-1"),
    ],
)
def test_invalid_values_raise(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        load_settings(tmp_path / "missing.env")


def test_api_key_never_printed(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    settings = load_settings(tmp_path / "missing.env")
    assert "sk-secret-value" not in str(settings)
    assert "sk-secret-value" not in repr(settings)
    assert "OPENAI_API_KEY: configurada" in str(settings)


def test_env_example_lists_every_setting():
    example = (config.PROJECT_ROOT / ".env.example").read_text()
    for name in ENV_NAMES:
        assert f"\n{name}=" in example
