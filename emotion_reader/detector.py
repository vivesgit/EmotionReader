"""Face detection (MediaPipe) and emotion classification (HSEmotion ONNX), fully local."""

from __future__ import annotations

import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QObject, QThread, QWaitCondition, Signal

from config import MODELS_DIR

FACE_MODEL_FILE = "blaze_face_short_range.tflite"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
EMOTION_MODEL_NAME = "enet_b0_8_best_vgaf"
EMOTION_MODEL_FILE = f"{EMOTION_MODEL_NAME}.onnx"
EMOTION_MODEL_URL = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
    f"models/affectnet_emotions/onnx/{EMOTION_MODEL_FILE}"
)

EMOTION_LABELS_ES = {
    "Anger": "enfado",
    "Contempt": "desprecio",
    "Disgust": "asco",
    "Fear": "miedo",
    "Happiness": "alegría",
    "Neutral": "neutral",
    "Sadness": "tristeza",
    "Surprise": "sorpresa",
}

# Rough pleasantness of each emotion, used to compute the trend.
VALENCE = {
    "alegría": 1.0,
    "sorpresa": 0.2,
    "neutral": 0.0,
    "desprecio": -0.5,
    "asco": -0.6,
    "miedo": -0.7,
    "tristeza": -0.8,
    "enfado": -0.8,
}

FACE_MARGIN = 0.1

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class EmotionReading:
    timestamp: float
    scores: dict[str, float] = field(default_factory=dict)
    box: Box | None = None

    @property
    def face_detected(self) -> bool:
        return bool(self.scores)

    @property
    def emotion(self) -> str | None:
        return max(self.scores, key=self.scores.get) if self.scores else None

    @property
    def confidence(self) -> float:
        return max(self.scores.values()) if self.scores else 0.0

    @property
    def valence(self) -> float:
        return sum(VALENCE[label] * p for label, p in self.scores.items())


class ModelDownloadError(RuntimeError):
    pass


def ensure_model(filename: str, url: str, models_dir: Path = MODELS_DIR) -> Path:
    """Return the local model path, downloading it once if missing."""
    path = models_dir / filename
    if path.is_file():
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"No se ha podido descargar el modelo {filename}. Comprueba tu conexión a "
            f"internet (solo hace falta la primera vez) o cópialo manualmente en {models_dir}."
        ) from exc
    partial.replace(path)
    return path


class FaceDetector(Protocol):
    def detect(self, frame_rgb: np.ndarray) -> list[Box]: ...


class EmotionClassifier(Protocol):
    def classify(self, face_rgb: np.ndarray) -> dict[str, float]: ...


class MediaPipeFaceDetector:
    def __init__(self, model_path: Path, min_confidence: float = 0.5) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=min_confidence,
            )
        )

    def detect(self, frame_rgb: np.ndarray) -> list[Box]:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        boxes = []
        for detection in self._detector.detect(image).detections:
            b = detection.bounding_box
            boxes.append((b.origin_x, b.origin_y, b.width, b.height))
        return boxes


class HSEmotionClassifier:
    """HSEmotion recogniser loaded from models/ instead of the library's ~/.hsemotion cache."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        recognizer = HSEmotionRecognizer.__new__(HSEmotionRecognizer)
        recognizer.is_mtl = False
        recognizer.idx_to_class = dict(enumerate(EMOTION_LABELS_ES))
        recognizer.img_size = 224
        options = ort.SessionOptions()
        # A few threads are enough at ~10 analyses per second and leave the UI responsive.
        options.intra_op_num_threads = 2
        recognizer.ort_session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._recognizer = recognizer

    def classify(self, face_rgb: np.ndarray) -> dict[str, float]:
        _, probabilities = self._recognizer.predict_emotions(face_rgb, logits=False)
        return {
            EMOTION_LABELS_ES[name]: float(probabilities[i])
            for i, name in self._recognizer.idx_to_class.items()
        }


def _crop_face(frame: np.ndarray, box: Box) -> tuple[np.ndarray, Box] | None:
    height, width = frame.shape[:2]
    x, y, w, h = box
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(width, x + w + mx), min(height, y + h + my)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return frame[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


class EmotionDetector:
    def __init__(self, face_detector: FaceDetector, classifier: EmotionClassifier) -> None:
        self._face_detector = face_detector
        self._classifier = classifier

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> EmotionDetector:
        face_model = ensure_model(FACE_MODEL_FILE, FACE_MODEL_URL, models_dir)
        emotion_model = ensure_model(EMOTION_MODEL_FILE, EMOTION_MODEL_URL, models_dir)
        return cls(MediaPipeFaceDetector(face_model), HSEmotionClassifier(emotion_model))

    def analyze(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> EmotionReading:
        timestamp = time.monotonic() if timestamp is None else timestamp
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = self._face_detector.detect(frame_rgb)
        if not boxes:
            return EmotionReading(timestamp)
        largest = max(boxes, key=lambda b: b[2] * b[3])
        cropped = _crop_face(frame_rgb, largest)
        if cropped is None:
            return EmotionReading(timestamp)
        face, box = cropped
        return EmotionReading(timestamp, self._classifier.classify(face), box)


class EmotionWorker(QThread):
    """Analyses only the most recent frame, at most `max_fps` times per second."""

    reading_ready = Signal(object)
    detector_ready = Signal()
    failed = Signal(str)

    def __init__(
        self,
        detector_factory: Callable[[], EmotionDetector] = EmotionDetector.load,
        max_fps: float = 10.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._detector_factory = detector_factory
        self._interval = 1.0 / max_fps
        self._mutex = QMutex()
        self._frame_available = QWaitCondition()
        self._latest_frame: np.ndarray | None = None
        self._running = False

    def start(self, *args) -> None:
        self._running = True
        super().start(*args)

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
            self._frame_available.wakeAll()
        self.wait()

    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        with QMutexLocker(self._mutex):
            self._latest_frame = frame_bgr
            self._frame_available.wakeAll()

    def _next_frame(self) -> np.ndarray | None:
        with QMutexLocker(self._mutex):
            while self._running and self._latest_frame is None:
                self._frame_available.wait(self._mutex, 200)
            frame, self._latest_frame = self._latest_frame, None
            return frame if self._running else None

    def run(self) -> None:
        try:
            detector = self._detector_factory()
        except Exception as exc:
            self.failed.emit(f"La detección de emociones no está disponible: {exc}")
            return
        self.detector_ready.emit()
        while self._running:
            frame = self._next_frame()
            if frame is None:
                continue
            started = time.monotonic()
            try:
                reading = detector.analyze(frame, started)
            except Exception as exc:
                self.failed.emit(f"Error analizando la imagen: {exc}")
                return
            self.reading_ready.emit(reading)
            remaining = self._interval - (time.monotonic() - started)
            if remaining > 0:
                self.msleep(int(remaining * 1000))
