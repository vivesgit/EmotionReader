"""RealtimeClient: OpenAI Realtime session, streaming audio, history and error handling."""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal

from config import Settings
from gpt_live_client.audio import SAMPLE_RATE, AudioDeviceError, AudioPlayer

AUDIO_FORMAT = {"type": "audio/pcm", "rate": SAMPLE_RATE}
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
AUDIO_APPEND_CHUNK_BYTES = 32 * 1024
MAX_HISTORY_TURNS = 20
LANGUAGE_NAMES = {
    "es": "español",
    "en": "inglés",
    "fr": "francés",
    "de": "alemán",
    "it": "italiano",
    "pt": "portugués",
    "ca": "catalán",
}


class GptLiveError(Exception):
    """Error with a message that can be shown to the user as-is."""


class MissingApiKeyError(GptLiveError):
    def __init__(self) -> None:
        super().__init__(
            "Falta la clave de OpenAI. Copia .env.example a .env, rellena OPENAI_API_KEY "
            "y vuelve a abrir la aplicación."
        )


class AuthError(GptLiveError):
    pass


class RateLimitError(GptLiveError):
    pass


class QuotaError(GptLiveError):
    pass


class OfflineError(GptLiveError):
    pass


class RequestTimeoutError(GptLiveError):
    pass


class ApiError(GptLiveError):
    pass


def error_from_code(code: str | None, message: str, model: str) -> GptLiveError:
    code = code or ""
    if code in ("invalid_api_key", "invalid_authentication", "unauthorized"):
        return AuthError(
            "OpenAI ha rechazado la clave (OPENAI_API_KEY). Revisa que esté bien copiada "
            "en .env y que siga activa."
        )
    if code == "insufficient_quota":
        return QuotaError(
            "Tu cuenta de OpenAI no tiene saldo o cuota disponible. Revisa la facturación "
            "en platform.openai.com."
        )
    if code in ("rate_limit_exceeded", "rate_limit"):
        return RateLimitError(
            "Has alcanzado el límite de uso de OpenAI. Espera unos segundos y vuelve a preguntar."
        )
    if code in ("model_not_found", "invalid_model"):
        return ApiError(
            f"El modelo «{model}» no está disponible para tu cuenta. Revisa GPT_LIVE_MODEL en .env."
        )
    return ApiError(f"OpenAI ha devuelto un error: {message or code or 'desconocido'}")


def error_from_exception(exc: BaseException, timeout: float, model: str) -> GptLiveError:
    if isinstance(exc, GptLiveError):
        return exc
    if isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return RequestTimeoutError(
            f"GPT-Live no ha respondido en {timeout:g} s. Comprueba tu conexión y vuelve a "
            "intentarlo (puedes ampliar REQUEST_TIMEOUT_SECONDS en .env)."
        )
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return error_from_code("invalid_api_key", "", model)
    if status == 429:
        return error_from_code("rate_limit_exceeded", "", model)
    if status == 404:
        return error_from_code("model_not_found", "", model)
    try:
        from websockets.exceptions import ConnectionClosed, InvalidHandshake
    except ImportError:  # pragma: no cover - websockets ships with openai[realtime]
        ConnectionClosed = InvalidHandshake = ()  # noqa: N806
    if isinstance(exc, OSError | ConnectionClosed | InvalidHandshake):
        return OfflineError(
            "No hay conexión con OpenAI. Comprueba tu conexión a internet y vuelve a preguntar."
        )
    return ApiError(f"Error inesperado al hablar con GPT-Live: {exc}")


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


class ConversationHistory:
    """Text transcript of the chat, replayed into a fresh session after a reconnection."""

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def add(self, role: str, text: str) -> None:
        if text.strip():
            self._turns.append(Turn(role, text.strip()))
            del self._turns[: -self.max_turns]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def default_client_factory(api_key: str) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key, max_retries=0)


def session_instructions(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, language)
    return (
        "Eres GPT-Live, un asistente de voz cercano y natural. Responde siempre hablando en "
        f"{name}, de forma breve y conversacional. Cada mensaje del usuario incluye un contexto "
        "emocional estimado a partir de su expresión facial: es aproximado, úsalo solo para "
        "adaptar el tono y nunca hagas diagnósticos ni lo menciones si no aporta."
    )


class RealtimeClient:
    """One Realtime WebSocket session per chat; each `ask` is a manual (push-to-talk) turn."""

    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[[str], Any] = default_client_factory,
        max_rate_limit_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.settings = settings
        self.history = ConversationHistory()
        self._client_factory = client_factory
        self._client: Any = None
        self._connection: Any = None
        self._max_rate_limit_retries = max_rate_limit_retries
        self._backoff = retry_backoff_seconds
        self._audio_received = False

    @property
    def connected(self) -> bool:
        return self._connection is not None

    @property
    def _timeout(self) -> float:
        return self.settings.request_timeout_seconds

    def _session_config(self) -> dict:
        return {
            "type": "realtime",
            "model": self.settings.gpt_live_model,
            "instructions": session_instructions(self.settings.response_language),
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": AUDIO_FORMAT,
                    "turn_detection": None,
                    "transcription": {
                        "model": TRANSCRIPTION_MODEL,
                        "language": self.settings.response_language,
                    },
                },
                "output": {"format": AUDIO_FORMAT, "voice": self.settings.gpt_live_voice},
            },
        }

    async def _recv(self) -> Any:
        return await asyncio.wait_for(self._connection.recv(), self._timeout)

    def _raise_for_error_event(self, event: Any) -> None:
        error = _field(event, "error")
        raise error_from_code(
            _field(error, "code"), _field(error, "message", ""), self.settings.gpt_live_model
        )

    async def _connect(self) -> None:
        if not self.settings.api_key_configured:
            raise MissingApiKeyError()
        if self._client is None:
            self._client = self._client_factory(self.settings.openai_api_key)
        manager = self._client.realtime.connect(model=self.settings.gpt_live_model)
        self._connection = await asyncio.wait_for(manager.enter(), self._timeout)
        # An invalid key arrives as an `error` event right after the handshake, so read it
        # before sending anything on a socket the server is about to close.
        await self._wait_for("session.created")
        await self._connection.send({"type": "session.update", "session": self._session_config()})
        await self._wait_for("session.updated")
        for turn in self.history.turns:
            content_type = "input_text" if turn.role == "user" else "output_text"
            await self._connection.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": turn.role,
                        "content": [{"type": content_type, "text": turn.text}],
                    },
                }
            )

    async def _wait_for(self, event_type: str) -> None:
        while True:
            event = await self._recv()
            if event.type == event_type:
                return
            if event.type == "error":
                self._raise_for_error_event(event)

    async def disconnect(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                await asyncio.wait_for(connection.close(), 5)
            except Exception:
                pass

    async def _run_turn(
        self, prompt: str, audio: bytes | None, on_audio: Callable[[bytes], None]
    ) -> tuple[str, str]:
        conn = self._connection
        await conn.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            }
        )
        if audio:
            for start in range(0, len(audio), AUDIO_APPEND_CHUNK_BYTES):
                chunk = audio[start : start + AUDIO_APPEND_CHUNK_BYTES]
                await conn.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            await conn.send({"type": "input_audio_buffer.commit"})
        await conn.send({"type": "response.create"})

        transcript_parts: list[str] = []
        transcript: str | None = None
        user_transcript = ""
        while True:
            event = await self._recv()
            kind = event.type
            if kind == "response.output_audio.delta":
                self._audio_received = True
                on_audio(base64.b64decode(event.delta))
            elif kind == "response.output_audio_transcript.delta":
                transcript_parts.append(event.delta)
            elif kind == "response.output_audio_transcript.done":
                transcript = event.transcript
            elif kind == "conversation.item.input_audio_transcription.completed":
                user_transcript = event.transcript
            elif kind == "error":
                self._raise_for_error_event(event)
            elif kind == "response.done":
                response = _field(event, "response")
                if _field(response, "status") == "failed":
                    error = _field(_field(response, "status_details"), "error")
                    raise error_from_code(
                        _field(error, "code"),
                        _field(error, "message", ""),
                        self.settings.gpt_live_model,
                    )
                break
        answer = transcript if transcript is not None else "".join(transcript_parts)
        return answer, user_transcript

    async def ask(self, prompt: str, audio: bytes | None, on_audio: Callable[[bytes], None]) -> str:
        """Send emotion context + question, stream the spoken answer to `on_audio`."""
        attempt = 0
        reconnected = False
        self._audio_received = False
        while True:
            try:
                if self._connection is None:
                    await self._connect()
                answer, user_transcript = await self._run_turn(prompt, audio, on_audio)
                break
            except asyncio.CancelledError:
                await self.disconnect()
                raise
            except Exception as raw:
                error = error_from_exception(raw, self._timeout, self.settings.gpt_live_model)
                await self.disconnect()
                retryable = not self._audio_received and not isinstance(raw, AudioDeviceError)
                if retryable and isinstance(error, RateLimitError):
                    if attempt < self._max_rate_limit_retries:
                        await asyncio.sleep(self._backoff * 2**attempt)
                        attempt += 1
                        continue
                if retryable and isinstance(error, OfflineError) and not reconnected:
                    reconnected = True
                    continue
                if isinstance(raw, AudioDeviceError):
                    raise
                raise error from raw

        user_text = prompt
        if user_transcript:
            user_text += f"\n[Transcripción de la pregunta]: {user_transcript}"
        self.history.add("user", user_text)
        self.history.add("assistant", answer)
        return answer

    async def new_chat(self) -> None:
        await self.disconnect()
        self.history.clear()

    async def close(self) -> None:
        await self.disconnect()
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            try:
                await client.close()
            except Exception:
                pass


class GptLiveBridge(QObject):
    """Runs RealtimeClient on a private asyncio thread and reports back through Qt signals."""

    audio_started = Signal()
    answer_ready = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self, client: RealtimeClient, player: AudioPlayer, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.player = player
        self.player.on_drained = self._on_drained
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="gpt-live", daemon=True)
        self._thread.start()
        self._future: Future | None = None
        self._awaiting_drain = False

    @property
    def busy(self) -> bool:
        return (self._future is not None and not self._future.done()) or self._awaiting_drain

    def ask(self, prompt: str, audio: bytes | None = None) -> None:
        self.cancel()
        self._future = asyncio.run_coroutine_threadsafe(self._ask(prompt, audio), self._loop)

    async def _ask(self, prompt: str, audio: bytes | None) -> None:
        started = False

        def on_audio(chunk: bytes) -> None:
            nonlocal started
            self.player.feed(chunk)
            if not started:
                started = True
                self.audio_started.emit()

        try:
            answer = await self.client.ask(prompt, audio, on_audio)
        except asyncio.CancelledError:
            raise
        except (GptLiveError, AudioDeviceError) as exc:
            self.player.stop()
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.player.stop()
            self.failed.emit(f"Error inesperado: {exc}")
            return
        self.answer_ready.emit(answer)
        self._awaiting_drain = True
        self.player.finish()

    def _on_drained(self) -> None:
        if self._awaiting_drain:
            self._awaiting_drain = False
            self.finished.emit()

    def cancel(self) -> None:
        self._awaiting_drain = False
        if self._future is not None and not self._future.done():
            self._future.cancel()
        self._future = None
        self.player.stop()

    def new_chat(self) -> None:
        self.cancel()
        asyncio.run_coroutine_threadsafe(self.client.new_chat(), self._loop)

    def shutdown(self) -> None:
        self.cancel()
        if self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.client.close(), self._loop).result(3)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(3)
        self.player.close()
