import asyncio
import base64

import numpy as np
import pytest
from openai import AsyncOpenAI

from config import Settings
from gpt_live_client import audio as audio_module
from gpt_live_client.audio import (
    SAMPLE_RATE,
    AudioDeviceError,
    AudioPlayer,
    MicRecorder,
    resample_pcm16,
)
from gpt_live_client.client import (
    ApiError,
    AuthError,
    ConversationHistory,
    GptLiveBridge,
    MissingApiKeyError,
    OfflineError,
    QuotaError,
    RateLimitError,
    RealtimeClient,
    RequestTimeoutError,
    error_from_code,
    error_from_exception,
)
from tests.fake_realtime_server import FakeRealtimeServer, tone

PROMPT = "[Contexto emocional]\n[Pregunta del usuario]: hola"


def run(coro):
    return asyncio.run(coro)


def make_client(server, **settings):
    values = {"openai_api_key": "sk-test", "request_timeout_seconds": 2} | settings
    return RealtimeClient(
        Settings(**values),
        client_factory=lambda key: AsyncOpenAI(
            api_key=key, base_url=server.base_url, max_retries=0
        ),
        retry_backoff_seconds=0.01,
    )


async def with_server(scenario, **server_options):
    server = await FakeRealtimeServer(**server_options).start()
    try:
        return await scenario(server)
    finally:
        await server.stop()


def sent(connection, kind):
    return [event for event in connection if event["type"] == kind]


# --- errors and history -----------------------------------------------------


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        ("invalid_api_key", AuthError),
        ("insufficient_quota", QuotaError),
        ("rate_limit_exceeded", RateLimitError),
        ("model_not_found", ApiError),
        ("server_error", ApiError),
    ],
)
def test_error_codes_map_to_clear_errors(code, error_type):
    error = error_from_code(code, "detalle", "gpt-realtime-2.1")
    assert type(error) is error_type
    assert str(error)


def test_model_not_found_mentions_setting():
    assert "GPT_LIVE_MODEL" in str(error_from_code("model_not_found", "", "otro-modelo"))


def test_exceptions_map_to_offline_and_timeout():
    assert isinstance(error_from_exception(ConnectionRefusedError(), 5, "m"), OfflineError)
    timeout = error_from_exception(asyncio.TimeoutError(), 5, "m")
    assert isinstance(timeout, RequestTimeoutError)
    assert "REQUEST_TIMEOUT_SECONDS" in str(timeout)


def test_history_limits_and_ignores_blank_turns():
    history = ConversationHistory(max_turns=3)
    history.add("user", "  ")
    for i in range(5):
        history.add("user", f"t{i}")
    assert [t.text for t in history.turns] == ["t2", "t3", "t4"]
    history.clear()
    assert len(history) == 0


def test_session_config_uses_settings():
    client = RealtimeClient(Settings(gpt_live_voice="cedar", response_language="en"))
    config = client._session_config()
    assert config["model"] == "gpt-realtime-2.1"
    assert config["output_modalities"] == ["audio"]
    assert config["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "cedar",
    }
    assert config["audio"]["input"]["turn_detection"] is None
    assert "inglés" in config["instructions"]


# --- RealtimeClient against the fake Realtime server ------------------------


def test_text_question_streams_audio_and_records_history():
    async def scenario(server):
        client = make_client(server)
        chunks = []
        answer = await client.ask(PROMPT, None, chunks.append)
        await client.close()
        return server, client, chunks, answer

    server, client, chunks, answer = run(with_server(scenario))
    assert answer == server.answer
    assert len(chunks) == 3
    assert b"".join(chunks) == tone()[: len(b"".join(chunks))]
    events = server.connections[0]
    assert events[0]["type"] == "session.update"
    assert events[0]["session"]["audio"]["output"]["voice"] == "marin"
    item = sent(events, "conversation.item.create")[0]["item"]
    assert item["content"] == [{"type": "input_text", "text": PROMPT}]
    assert sent(events, "input_audio_buffer.append") == []
    assert [t.role for t in client.history.turns] == ["user", "assistant"]


def test_voice_question_sends_audio_and_keeps_transcription():
    pcm = (np.arange(40_000) % 200).astype(np.int16).tobytes()

    async def scenario(server):
        client = make_client(server)
        await client.ask(PROMPT, pcm, lambda _c: None)
        await client.close()
        return server, client

    server, client = run(with_server(scenario))
    events = server.connections[0]
    appended = b"".join(
        base64.b64decode(e["audio"]) for e in sent(events, "input_audio_buffer.append")
    )
    assert appended == pcm
    assert len(sent(events, "input_audio_buffer.append")) == 3
    assert len(sent(events, "input_audio_buffer.commit")) == 1
    assert "[Transcripción de la pregunta]: ¿Qué tal estoy?" in client.history.turns[0].text


def test_conversation_continues_on_same_session():
    async def scenario(server):
        client = make_client(server)
        await client.ask("uno", None, lambda _c: None)
        await client.ask("dos", None, lambda _c: None)
        await client.close()
        return server, client

    server, client = run(with_server(scenario))
    assert len(server.connections) == 1
    assert server.responses == 2
    assert len(client.history) == 4


def test_missing_api_key_does_not_connect():
    async def scenario(server):
        client = make_client(server, openai_api_key="")
        with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY"):
            await client.ask(PROMPT, None, lambda _c: None)
        return server

    assert run(with_server(scenario)).connections == []


def test_invalid_key_is_reported():
    async def scenario(server):
        client = make_client(server)
        with pytest.raises(AuthError, match="OPENAI_API_KEY"):
            await client.ask(PROMPT, None, lambda _c: None)
        assert not client.connected

    run(with_server(scenario, behaviour="invalid_key"))


def test_rate_limit_is_retried_then_succeeds():
    async def scenario(server):
        client = make_client(server)
        answer = await client.ask(PROMPT, None, lambda _c: None)
        await client.close()
        return server, answer

    server, answer = run(with_server(scenario, fail_next="rate_limit"))
    assert answer == server.answer
    assert server.responses == 2


def test_persistent_rate_limit_gives_up():
    async def scenario(server):
        client = make_client(server)
        with pytest.raises(RateLimitError, match="límite de uso"):
            await client.ask(PROMPT, None, lambda _c: None)
        return server

    assert run(with_server(scenario, behaviour="rate_limit_always")).responses == 3


def test_failed_response_reports_quota():
    async def scenario(server):
        client = make_client(server)
        with pytest.raises(QuotaError):
            await client.ask(PROMPT, None, lambda _c: None)

    run(with_server(scenario, behaviour="failed_response"))


def test_silent_server_times_out():
    async def scenario(server):
        client = make_client(server, request_timeout_seconds=0.3)
        with pytest.raises(RequestTimeoutError):
            await client.ask(PROMPT, None, lambda _c: None)

    run(with_server(scenario, behaviour="hang"))


def test_no_server_means_offline():
    async def scenario():
        server = await FakeRealtimeServer().start()
        await server.stop()
        client = make_client(server)
        with pytest.raises(OfflineError, match="conexión a internet"):
            await client.ask(PROMPT, None, lambda _c: None)

    run(scenario())


def test_dropped_connection_reconnects_and_replays_history():
    async def scenario(server):
        client = make_client(server)
        await client.ask("primera", None, lambda _c: None)
        server.fail_next = "drop"
        answer = await client.ask("segunda", None, lambda _c: None)
        await client.close()
        return server, answer

    server, answer = run(with_server(scenario))
    assert answer == server.answer
    assert len(server.connections) == 2
    replayed = [e["item"] for e in sent(server.connections[1], "conversation.item.create")]
    assert [i["role"] for i in replayed] == ["user", "assistant", "user"]
    assert replayed[0]["content"][0] == {"type": "input_text", "text": "primera"}
    assert replayed[1]["content"][0]["type"] == "output_text"


def test_drop_after_audio_started_is_not_retried():
    async def scenario(server):
        client = make_client(server)
        chunks = []
        with pytest.raises(OfflineError):
            await client.ask(PROMPT, None, chunks.append)
        return server, chunks

    server, chunks = run(with_server(scenario, fail_next="drop_after_audio"))
    assert len(chunks) == 1
    assert server.responses == 1


def test_new_chat_starts_fresh_session():
    async def scenario(server):
        client = make_client(server)
        await client.ask("primera", None, lambda _c: None)
        await client.new_chat()
        assert len(client.history) == 0
        await client.ask("otra", None, lambda _c: None)
        await client.close()
        return server

    server = run(with_server(scenario))
    assert len(server.connections) == 2
    assert len(sent(server.connections[1], "conversation.item.create")) == 1


# --- audio ------------------------------------------------------------------


class FakeStream:
    def __init__(self, samplerate, channels, dtype, callback, supported=(SAMPLE_RATE,)):
        if samplerate not in supported:
            raise ValueError("Invalid sample rate")
        assert (channels, dtype) == (1, "int16")
        self.samplerate = samplerate
        self.callback = callback
        self.started = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class FakeSoundDevice:
    def __init__(self, supported=(SAMPLE_RATE,), native_rate=48_000, has_device=True):
        self.streams = []
        self.supported = supported
        self.native_rate = native_rate
        self.has_device = has_device

    def query_devices(self, kind):
        if not self.has_device:
            raise ValueError("No device")
        return {"default_samplerate": float(self.native_rate)}

    def _make(self, **kwargs):
        stream = FakeStream(supported=self.supported, **kwargs)
        self.streams.append(stream)
        return stream

    def RawInputStream(self, **kwargs):  # noqa: N802
        return self._make(**kwargs)

    def RawOutputStream(self, **kwargs):  # noqa: N802
        return self._make(**kwargs)


def pull(stream, frames):
    out = bytearray(frames * 2)
    stream.callback(out, frames, None, None)
    return bytes(out)


def test_resample_pcm16_changes_length():
    pcm = np.zeros(4800, dtype=np.int16).tobytes()
    assert len(resample_pcm16(pcm, 48_000, 24_000)) == 4800
    assert resample_pcm16(pcm, 24_000, 24_000) is pcm


def test_player_streams_chunks_and_reports_drain():
    sd = FakeSoundDevice()
    drained = []
    player = AudioPlayer(lambda: drained.append(True), sd)
    player.feed(b"\x01\x00" * 100)
    stream = sd.streams[0]
    assert stream.started and stream.samplerate == SAMPLE_RATE
    player.finish()
    assert player.playing and drained == []
    assert pull(stream, 60) == b"\x01\x00" * 60
    assert pull(stream, 60) == b"\x01\x00" * 40 + b"\x00\x00" * 20
    assert drained == [True]
    assert not player.playing


def test_player_finish_without_audio_drains_immediately():
    drained = []
    AudioPlayer(lambda: drained.append(True), FakeSoundDevice()).finish()
    assert drained == [True]


def test_player_stop_discards_pending_audio():
    sd = FakeSoundDevice()
    player = AudioPlayer(None, sd)
    player.feed(b"\x05\x00" * 50)
    player.stop()
    assert pull(sd.streams[0], 10) == b"\x00\x00" * 10
    player.close()
    assert sd.streams[0].closed


def test_player_falls_back_to_native_rate():
    sd = FakeSoundDevice(supported=(48_000,))
    player = AudioPlayer(None, sd)
    player.feed(np.zeros(240, dtype=np.int16).tobytes())
    assert sd.streams[-1].samplerate == 48_000
    assert len(pull(sd.streams[-1], 480)) == 960


def test_player_without_output_device():
    with pytest.raises(AudioDeviceError, match="salida de audio"):
        AudioPlayer(None, FakeSoundDevice(has_device=False)).feed(b"\x00\x00")


def test_recorder_returns_24khz_audio():
    sd = FakeSoundDevice(supported=(48_000,))
    recorder = MicRecorder(sd)
    recorder.start()
    assert recorder.recording
    stream = sd.streams[-1]
    stream.callback(np.ones(480, dtype=np.int16).tobytes(), 480, None, None)
    pcm = recorder.stop()
    assert not recorder.recording
    assert stream.closed
    assert len(pcm) == 480


def test_recorder_without_microphone():
    with pytest.raises(AudioDeviceError, match="micrófono"):
        MicRecorder(FakeSoundDevice(has_device=False)).start()


def test_recorder_caps_duration(monkeypatch):
    monkeypatch.setattr(audio_module, "MAX_RECORDING_SECONDS", 0.01)
    sd = FakeSoundDevice()
    recorder = MicRecorder(sd)
    recorder.start()
    for _ in range(10):
        sd.streams[0].callback(b"\x00\x00" * 240, 240, None, None)
    assert len(recorder.stop()) == 480


# --- Qt bridge --------------------------------------------------------------


class ScriptedClient:
    def __init__(self, error=None, chunks=2):
        self.error = error
        self.chunks = chunks
        self.calls = []
        self.new_chats = 0
        self.closed = False

    async def ask(self, prompt, audio, on_audio):
        self.calls.append((prompt, audio))
        for _ in range(self.chunks):
            on_audio(b"\x01\x00" * 10)
        if self.error:
            raise self.error
        return "respuesta"

    async def new_chat(self):
        self.new_chats += 1

    async def close(self):
        self.closed = True


def test_bridge_emits_audio_answer_and_finished(qtbot):
    sd = FakeSoundDevice()
    client = ScriptedClient()
    bridge = GptLiveBridge(client, AudioPlayer(None, sd))
    with qtbot.waitSignals([bridge.audio_started, bridge.answer_ready], timeout=3000):
        bridge.ask(PROMPT, b"audio")
    assert client.calls == [(PROMPT, b"audio")]
    assert bridge.busy
    with qtbot.waitSignal(bridge.finished, timeout=3000):
        pull(sd.streams[0], 100)
    assert not bridge.busy
    bridge.shutdown()
    assert client.closed


def test_bridge_reports_errors(qtbot):
    bridge = GptLiveBridge(
        ScriptedClient(AuthError("clave mala")), AudioPlayer(None, FakeSoundDevice())
    )
    with qtbot.waitSignal(bridge.failed, timeout=3000) as blocker:
        bridge.ask(PROMPT)
    assert blocker.args == ["clave mala"]
    bridge.shutdown()


def test_bridge_reports_missing_speaker(qtbot):
    player = AudioPlayer(None, FakeSoundDevice(has_device=False))
    bridge = GptLiveBridge(ScriptedClient(), player)
    with qtbot.waitSignal(bridge.failed, timeout=3000) as blocker:
        bridge.ask(PROMPT)
    assert "salida de audio" in blocker.args[0]
    bridge.shutdown()


def test_bridge_new_chat(qtbot):
    client = ScriptedClient()
    bridge = GptLiveBridge(client, AudioPlayer(None, FakeSoundDevice()))
    bridge.new_chat()
    qtbot.waitUntil(lambda: client.new_chats == 1, timeout=3000)
    bridge.shutdown()
