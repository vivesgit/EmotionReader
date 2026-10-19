"""MicRecorder and AudioPlayer for PCM16 mono 24 kHz audio (the Realtime API format)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import ModuleType

import numpy as np

SAMPLE_RATE = 24_000
MAX_RECORDING_SECONDS = 60
# Python callbacks compete for the GIL with the camera and emotion threads; a low-latency
# stream misses deadlines and crackles.
STREAM_LATENCY_SECONDS = 0.2
# Audio deltas arrive in bursts; play only once this much is queued so the network jitter
# never reaches the speaker.
PREBUFFER_SECONDS = 0.3


class AudioDeviceError(RuntimeError):
    pass


def _sounddevice() -> ModuleType:
    try:
        import sounddevice
    except OSError as exc:
        raise AudioDeviceError(
            "No se ha encontrado la librería de audio PortAudio. "
            "Reinstala las dependencias con «pip install -r requirements.txt»."
        ) from exc
    return sounddevice


def resample_pcm16(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    if from_rate == to_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    target_length = max(1, round(len(samples) * to_rate / from_rate))
    positions = np.linspace(0, len(samples) - 1, target_length)
    resampled = np.interp(positions, np.arange(len(samples)), samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


class StreamResampler:
    """Resamples consecutive chunks as one continuous signal (no clicks at chunk joins)."""

    def __init__(self, from_rate: int, to_rate: int) -> None:
        self._step = from_rate / to_rate
        self._passthrough = from_rate == to_rate
        self.reset()

    def reset(self) -> None:
        self._last: np.ndarray | None = None
        self._position = 0.0

    def process(self, pcm: bytes) -> bytes:
        if self._passthrough or not pcm:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if self._last is not None:
            samples = np.concatenate((self._last, samples))
        end = len(samples) - 1
        count = 0
        if end >= self._position:
            count = int(np.floor((end - self._position) / self._step)) + 1
        positions = self._position + np.arange(count) * self._step
        resampled = np.interp(positions, np.arange(len(samples)), samples)
        self._position = self._position + count * self._step - end
        self._last = samples[-1:]
        return np.clip(np.round(resampled), -32768, 32767).astype(np.int16).tobytes()


def _open_stream(sd: ModuleType, kind: str, callback: Callable) -> tuple[object, int]:
    """Open at 24 kHz, falling back to the device's native rate when it is unsupported."""
    stream_class = sd.RawInputStream if kind == "input" else sd.RawOutputStream
    try:
        device = sd.query_devices(kind=kind)
    except Exception as exc:
        what = "micrófono" if kind == "input" else "salida de audio (altavoces o auriculares)"
        raise AudioDeviceError(f"No se ha encontrado ningún {what}.") from exc
    rates = [SAMPLE_RATE, int(device["default_samplerate"])]
    last_error: Exception | None = None
    for rate in dict.fromkeys(rates):
        try:
            stream = stream_class(
                samplerate=rate,
                channels=1,
                dtype="int16",
                latency=STREAM_LATENCY_SECONDS,
                callback=callback,
            )
            stream.start()
            return stream, rate
        except Exception as exc:
            last_error = exc
    raise AudioDeviceError(f"No se ha podido abrir el dispositivo de audio: {last_error}")


class MicRecorder:
    """Records while the Preguntar button is held; `stop()` returns PCM16 mono 24 kHz."""

    def __init__(self, sounddevice_module: ModuleType | None = None) -> None:
        self._sd = sounddevice_module
        self._stream = None
        self._rate = SAMPLE_RATE
        self._chunks: list[bytes] = []
        self._size = 0
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, _frames, _time, _status) -> None:
        with self._lock:
            if self._size < MAX_RECORDING_SECONDS * self._rate * 2:
                self._chunks.append(bytes(indata))
                self._size += len(indata)

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._chunks, self._size = [], 0
        sd = self._sd or _sounddevice()
        self._stream, self._rate = _open_stream(sd, "input", self._callback)

    def stop(self) -> bytes:
        if self._stream is None:
            return b""
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()
        with self._lock:
            pcm = b"".join(self._chunks)
            self._chunks, self._size = [], 0
        return resample_pcm16(pcm, self._rate, SAMPLE_RATE)


class AudioPlayer:
    """Plays streamed PCM16 chunks as they arrive and reports when playback has drained."""

    def __init__(
        self,
        on_drained: Callable[[], None] | None = None,
        sounddevice_module: ModuleType | None = None,
    ) -> None:
        self.on_drained = on_drained
        self._sd = sounddevice_module
        self._stream = None
        self._rate = SAMPLE_RATE
        self._buffer = bytearray()
        self._finishing = False
        self._prebuffering = True
        self._resampler = StreamResampler(SAMPLE_RATE, SAMPLE_RATE)
        self._lock = threading.Lock()

    @property
    def playing(self) -> bool:
        with self._lock:
            return bool(self._buffer) or self._finishing

    def _ensure_stream(self) -> None:
        if self._stream is None:
            sd = self._sd or _sounddevice()
            self._stream, self._rate = _open_stream(sd, "output", self._callback)
            self._resampler = StreamResampler(SAMPLE_RATE, self._rate)

    def _callback(self, outdata, frames, _time, _status) -> None:
        needed = frames * 2
        drained = False
        with self._lock:
            if self._prebuffering and not self._finishing:
                if len(self._buffer) < PREBUFFER_SECONDS * self._rate * 2:
                    outdata[:] = b"\x00" * needed
                    return
                self._prebuffering = False
            chunk = bytes(self._buffer[:needed])
            del self._buffer[:needed]
            if not self._buffer:
                self._prebuffering = True
                if self._finishing:
                    self._finishing = False
                    self._resampler.reset()
                    drained = True
        outdata[: len(chunk)] = chunk
        outdata[len(chunk) :] = b"\x00" * (needed - len(chunk))
        if drained and self.on_drained is not None:
            self.on_drained()

    def feed(self, pcm: bytes) -> None:
        self._ensure_stream()
        with self._lock:
            self._buffer.extend(self._resampler.process(pcm))

    def finish(self) -> None:
        """Mark the end of the response; `on_drained` fires once the buffer empties."""
        with self._lock:
            if self._stream is None or not self._buffer:
                drained = True
            else:
                self._finishing = True
                drained = False
        if drained and self.on_drained is not None:
            self.on_drained()

    def stop(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._finishing = False
            self._prebuffering = True
            self._resampler.reset()

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            stream, self._stream = self._stream, None
            stream.stop()
            stream.close()
