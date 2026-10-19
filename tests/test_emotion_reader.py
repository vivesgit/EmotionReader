from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt

from config import MODELS_DIR
from emotion_reader import camera, detector
from emotion_reader.aggregator import EmotionAggregator, EmotionSummary
from emotion_reader.camera import CameraWorker, ImageCapture, PermissionState, SyntheticCapture
from emotion_reader.detector import (
    EMOTION_LABELS_ES,
    EmotionDetector,
    EmotionReading,
    EmotionWorker,
    ModelDownloadError,
    ensure_model,
)
from emotion_reader.prompt import (
    VOICE_QUESTION,
    build_emotion_context,
    build_prompt,
    format_prompt_log,
)

FACE_IMAGE = Path(__file__).parent / "data" / "face.jpg"
LABELS = list(EMOTION_LABELS_ES.values())


def scores(**values):
    result = dict.fromkeys(LABELS, 0.0)
    result.update(values)
    return result


class ClosedCapture:
    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


class BrokenCapture(SyntheticCapture):
    def read(self):
        return False, None


class AsymmetricCapture(SyntheticCapture):
    def read(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, 0] = 255
        return True, frame


def test_synthetic_capture_produces_bgr_frames():
    capture = SyntheticCapture(320, 240)
    ok, frame = capture.read()
    assert ok
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    capture.release()
    assert not capture.isOpened()
    assert capture.read() == (False, None)


def test_worker_emits_frames(qtbot):
    worker = CameraWorker(0, lambda _i: SyntheticCapture(320, 240), fps=60)
    frames = []
    worker.frame_ready.connect(frames.append)
    with qtbot.waitSignal(worker.camera_opened, timeout=2000):
        worker.start()
    qtbot.waitUntil(lambda: len(frames) >= 3, timeout=2000)
    worker.stop()
    assert not worker.isRunning()
    assert frames[0].shape == (240, 320, 3)


def test_worker_mirrors_frames(qtbot):
    worker = CameraWorker(0, lambda _i: AsymmetricCapture(), fps=60)
    frames = []
    worker.frame_ready.connect(frames.append)
    worker.start()
    qtbot.waitUntil(lambda: len(frames) > 0, timeout=2000)
    worker.stop()
    assert frames[0][0, -1, 0] == 255
    assert frames[0][0, 0, 0] == 0


def test_worker_reports_unavailable_camera(qtbot):
    worker = CameraWorker(3, lambda _i: ClosedCapture(), open_timeout_seconds=0.1)
    with qtbot.waitSignal(worker.failed, timeout=3000) as blocker:
        worker.start()
    worker.wait()
    assert "cámara 3" in blocker.args[0]
    assert "CAMERA_INDEX" in blocker.args[0]


def test_worker_reports_lost_camera(qtbot):
    worker = CameraWorker(0, lambda _i: BrokenCapture(), fps=200, max_failed_reads=3)
    with qtbot.waitSignal(worker.failed, timeout=3000) as blocker:
        worker.start()
    worker.wait()
    assert blocker.args[0] == camera.CAMERA_LOST_MESSAGE


def test_stop_before_camera_opens_does_not_report_failure(qtbot):
    worker = CameraWorker(0, lambda _i: ClosedCapture(), open_timeout_seconds=5)
    failures = []
    worker.failed.connect(failures.append)
    worker.start()
    worker.stop()
    qtbot.wait(50)
    assert failures == []


@pytest.fixture
def qt_permissions(monkeypatch):
    monkeypatch.setattr(camera, "_qt_permission_dialogs_available", lambda: True)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (Qt.PermissionStatus.Granted, PermissionState.GRANTED),
        (Qt.PermissionStatus.Denied, PermissionState.DENIED),
    ],
)
def test_request_permission_uses_known_status(qapp, qt_permissions, monkeypatch, status, expected):
    monkeypatch.setattr(qapp, "checkPermission", lambda _p: status)
    results = []
    camera.request_permission("camera", results.append)
    assert results == [expected]


def test_request_permission_asks_when_undetermined(qapp, qt_permissions, monkeypatch):
    statuses = iter([Qt.PermissionStatus.Undetermined, Qt.PermissionStatus.Granted])
    monkeypatch.setattr(qapp, "checkPermission", lambda _p: next(statuses))
    requested = []

    def fake_request(permission, _context, callback):
        requested.append(type(permission).__name__)
        callback(permission)

    monkeypatch.setattr(qapp, "requestPermission", fake_request)
    results = []
    camera.request_permission("microphone", results.append)
    assert requested == ["QMicrophonePermission"]
    assert results == [PermissionState.GRANTED]


def test_unbundled_macos_defers_to_avfoundation(monkeypatch):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    monkeypatch.delattr(camera.sys, "frozen", raising=False)
    results = []
    camera.request_permission("camera", results.append)
    assert results == [PermissionState.GRANTED]


def test_privacy_settings_urls_point_to_macos_panes():
    assert camera.PRIVACY_SETTINGS_URLS["camera"].endswith("Privacy_Camera")
    assert camera.PRIVACY_SETTINGS_URLS["microphone"].endswith("Privacy_Microphone")


# --- detector ---------------------------------------------------------------


class FakeFaceDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.frames = []

    def detect(self, frame_rgb):
        self.frames.append(frame_rgb)
        return self.boxes


class FakeClassifier:
    def __init__(self, result=None):
        self.result = result or scores(alegría=0.9, neutral=0.1)
        self.faces = []

    def classify(self, face_rgb):
        self.faces.append(face_rgb)
        return self.result


def test_reading_properties():
    reading = EmotionReading(1.0, scores(tristeza=0.7, neutral=0.3), (0, 0, 10, 10))
    assert reading.face_detected
    assert reading.emotion == "tristeza"
    assert reading.confidence == pytest.approx(0.7)
    assert reading.valence == pytest.approx(-0.56)
    empty = EmotionReading(1.0)
    assert not empty.face_detected
    assert empty.emotion is None
    assert empty.confidence == 0.0


def test_detector_classifies_largest_face_with_margin():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    faces = FakeFaceDetector([(10, 10, 20, 20), (100, 50, 80, 100)])
    classifier = FakeClassifier()
    reading = EmotionDetector(faces, classifier).analyze(frame, timestamp=5.0)
    assert reading.timestamp == 5.0
    assert reading.emotion == "alegría"
    assert reading.box == (92, 40, 96, 120)
    assert classifier.faces[0].shape == (120, 96, 3)


def test_detector_converts_bgr_to_rgb():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[..., 0] = 255
    faces = FakeFaceDetector([])
    EmotionDetector(faces, FakeClassifier()).analyze(frame)
    assert faces.frames[0][0, 0].tolist() == [0, 0, 255]


def test_detector_without_face_returns_empty_reading():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    reading = EmotionDetector(FakeFaceDetector([]), FakeClassifier()).analyze(frame, 1.0)
    assert not reading.face_detected
    assert reading.box is None


def test_ensure_model_skips_download_when_present(tmp_path, monkeypatch):
    (tmp_path / "m.onnx").write_bytes(b"x")
    monkeypatch.setattr(detector.urllib.request, "urlretrieve", pytest.fail)
    assert ensure_model("m.onnx", "http://unused", tmp_path) == tmp_path / "m.onnx"


def test_ensure_model_download_failure_is_clear(tmp_path, monkeypatch):
    def offline(_url, target):
        Path(target).write_bytes(b"partial")
        raise OSError("sin red")

    monkeypatch.setattr(detector.urllib.request, "urlretrieve", offline)
    with pytest.raises(ModelDownloadError, match="conexión a internet"):
        ensure_model("m.onnx", "http://example", tmp_path / "models")
    assert list((tmp_path / "models").iterdir()) == []


def test_emotion_worker_emits_readings(qtbot):
    fake = EmotionDetector(FakeFaceDetector([(0, 0, 40, 40)]), FakeClassifier())
    worker = EmotionWorker(lambda: fake, max_fps=100)
    readings = []
    worker.reading_ready.connect(readings.append)
    with qtbot.waitSignal(worker.detector_ready, timeout=2000):
        worker.start()
    worker.submit_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    qtbot.waitUntil(lambda: len(readings) == 1, timeout=2000)
    worker.stop()
    assert readings[0].emotion == "alegría"


def test_emotion_worker_reports_load_failure(qtbot):
    def broken():
        raise ModelDownloadError("sin modelo")

    worker = EmotionWorker(broken)
    with qtbot.waitSignal(worker.failed, timeout=2000) as blocker:
        worker.start()
    worker.wait()
    assert "sin modelo" in blocker.args[0]


def test_image_capture(tmp_path):
    capture = ImageCapture(str(FACE_IMAGE))
    ok, frame = capture.read()
    assert ok and frame.shape == (320, 320, 3)
    with pytest.raises(ValueError):
        ImageCapture(str(tmp_path / "missing.jpg"))


@pytest.fixture(scope="module")
def real_detector():
    try:
        return EmotionDetector.load(MODELS_DIR)
    except ModelDownloadError as exc:
        pytest.skip(str(exc))


def test_real_models_detect_smiling_face(real_detector):
    reading = real_detector.analyze(cv2.imread(str(FACE_IMAGE)))
    assert reading.face_detected
    assert reading.emotion == "alegría"
    assert set(reading.scores) == set(LABELS)
    assert sum(reading.scores.values()) == pytest.approx(1.0, abs=1e-3)


def test_real_models_find_no_face_in_pattern(real_detector):
    _, frame = SyntheticCapture().read()
    assert not real_detector.analyze(frame).face_detected


# --- aggregator -------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make_aggregator(window=5.0):
    clock = Clock()
    return EmotionAggregator(window, clock), clock


def test_aggregator_without_readings():
    aggregator, _ = make_aggregator()
    summary = aggregator.summary()
    assert summary == EmotionSummary(5.0, 0, 0)
    assert not summary.has_face


def test_aggregator_dominant_secondary_and_share():
    aggregator, clock = make_aggregator()
    for i in range(10):
        values = (
            scores(tristeza=0.7, neutral=0.2, miedo=0.1)
            if i < 7
            else scores(neutral=0.6, tristeza=0.4)
        )
        aggregator.add(EmotionReading(clock.now, values, (0, 0, 1, 1)))
        clock.now += 0.1
    aggregator.add(EmotionReading(clock.now))
    summary = aggregator.summary()
    assert summary.total_frames == 11
    assert summary.frames_with_face == 10
    assert summary.dominant == "tristeza"
    assert summary.dominant_confidence == pytest.approx(0.61)
    assert summary.dominant_share == pytest.approx(0.7)
    assert [label for label, _ in summary.secondary] == ["neutral", "miedo"]
    assert summary.secondary[0][1] == pytest.approx(0.32)


def test_aggregator_drops_readings_outside_window():
    aggregator, clock = make_aggregator(window=5)
    aggregator.add(EmotionReading(clock.now, scores(enfado=1.0)))
    clock.now += 6
    aggregator.add(EmotionReading(clock.now, scores(alegría=1.0)))
    summary = aggregator.summary()
    assert summary.total_frames == 1
    assert summary.dominant == "alegría"
    clock.now += 10
    assert aggregator.summary().total_frames == 0


@pytest.mark.parametrize(
    ("first", "second", "trend"),
    [
        ("tristeza", "alegría", "mejorando"),
        ("alegría", "enfado", "empeorando"),
        ("neutral", "neutral", "estable"),
    ],
)
def test_aggregator_trend(first, second, trend):
    aggregator, clock = make_aggregator()
    for label in [first] * 3 + [second] * 3:
        aggregator.add(EmotionReading(clock.now, scores(**{label: 1.0})))
        clock.now += 0.1
    assert aggregator.summary().trend == trend


def test_aggregator_trend_needs_enough_frames():
    aggregator, clock = make_aggregator()
    aggregator.add(EmotionReading(clock.now, scores(alegría=1.0)))
    assert aggregator.summary().trend == "sin datos suficientes"


def test_aggregator_clear():
    aggregator, clock = make_aggregator()
    aggregator.add(EmotionReading(clock.now, scores(alegría=1.0)))
    aggregator.clear()
    assert aggregator.summary().total_frames == 0


# --- prompt -----------------------------------------------------------------


EXAMPLE_SUMMARY = EmotionSummary(
    window_seconds=5,
    total_frames=50,
    frames_with_face=48,
    dominant="tristeza",
    dominant_confidence=0.62,
    dominant_share=0.7,
    secondary=(("neutral", 0.21), ("miedo", 0.08)),
    trend="empeorando",
    valence_start=-0.15,
    valence_end=-0.40,
)


def test_prompt_matches_plan_example():
    assert build_prompt(EXAMPLE_SUMMARY, "¿Qué hago hoy?") == (
        "[Contexto emocional estimado del usuario, últimos 5 s, aproximado]\n"
        "- Emoción dominante: tristeza (confianza media 0.62, presente en el 70% de los frames)\n"
        "- Secundarias: neutral (0.21), miedo (0.08)\n"
        "- Tendencia: empeorando (valencia -0.15 -> -0.40)\n"
        "- Cara detectada en 48/50 frames\n"
        "Adapta tono y empatía a este contexto sin mencionarlo salvo que sea útil.\n"
        "[Pregunta del usuario]: ¿Qué hago hoy?"
    )


def test_prompt_for_voice_question():
    assert build_prompt(EXAMPLE_SUMMARY, None).endswith(f"[Pregunta del usuario]: {VOICE_QUESTION}")
    assert build_prompt(EXAMPLE_SUMMARY, "   ").endswith(VOICE_QUESTION)


def test_prompt_without_face():
    context = build_emotion_context(EmotionSummary(5, 12, 0))
    assert "No se ha detectado ninguna cara (12 frames analizados)" in context
    assert "Emoción dominante" not in context


def test_prompt_confidence_levels_and_missing_trend():
    summary = EmotionSummary(
        5, 2, 2, dominant="alegría", dominant_confidence=0.9, dominant_share=1.0
    )
    context = build_emotion_context(summary)
    assert "confianza alta 0.90" in context
    assert "- Secundarias: ninguna relevante" in context
    assert "- Tendencia: sin datos suficientes" in context


def test_prompt_is_short_plain_text():
    prompt = build_prompt(EXAMPLE_SUMMARY, "hola")
    assert all(ch.isprintable() or ch == "\n" for ch in prompt)
    assert len(prompt) < 1000


def test_prompt_log_shows_full_prompt_framed():
    from datetime import datetime

    prompt = build_prompt(EXAMPLE_SUMMARY, "¿Qué hago hoy?")
    log = format_prompt_log(prompt, 3, datetime(2026, 10, 19, 18, 56, 3), "gpt-realtime-2.1")
    lines = log.splitlines()
    assert lines[0].startswith("╭─ Prompt #3 · 18:56:03 · texto → gpt-realtime-2.1 ")
    assert lines[-1].startswith("╰")
    assert all(line.startswith("│") for line in lines[1:-1])
    body = " ".join(line[1:].strip() for line in lines[1:-1] if line[1:].strip())
    for original in prompt.splitlines()[:-1]:
        assert original in body
    assert "│   ¿Qué hago hoy?" in lines
    assert "\033[" not in log
    assert max(len(line) for line in lines) <= 80


def test_prompt_log_voice_colour_and_wrapping():
    from datetime import datetime

    prompt = build_prompt(EXAMPLE_SUMMARY, None)
    log = format_prompt_log(prompt, 1, datetime(2026, 1, 1), "m", 2.4, color=True, width=50)
    assert "voz, 2.4 s de audio" in log
    assert "\033[1m\033[33m[Pregunta del usuario]" in log
    plain = format_prompt_log(prompt, 1, datetime(2026, 1, 1), "m", 2.4, width=50)
    assert max(len(line) for line in plain.splitlines()[1:]) <= 50
