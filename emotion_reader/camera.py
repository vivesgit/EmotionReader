"""Camera/microphone permissions and the CameraWorker capture thread."""

from __future__ import annotations

import enum
import math
import sys
import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np
from PySide6.QtCore import (
    QCameraPermission,
    QCoreApplication,
    QMicrophonePermission,
    QObject,
    Qt,
    QThread,
    QUrl,
    Signal,
)
from PySide6.QtGui import QDesktopServices

PRIVACY_SETTINGS_URLS = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
}

CAMERA_DENIED_MESSAGE = (
    "Emotion Reader no tiene permiso para usar la cámara.\n\n"
    "Para activarlo abre Ajustes del Sistema > Privacidad y seguridad > Cámara "
    "y activa el interruptor de Emotion Reader (o de tu Terminal si lo ejecutas "
    "con «python main.py»). Después vuelve a abrir la aplicación."
)

CAMERA_UNAVAILABLE_MESSAGE = (
    "No se ha podido abrir la cámara {index}.\n\n"
    "Comprueba que la webcam está conectada y que ninguna otra aplicación la está usando. "
    "En macOS revisa también Ajustes del Sistema > Privacidad y seguridad > Cámara. "
    "Si tienes varias cámaras, prueba otro valor de CAMERA_INDEX en .env."
)

CAMERA_LOST_MESSAGE = "La cámara ha dejado de enviar imágenes. Comprueba la conexión de la webcam."


class PermissionState(enum.Enum):
    GRANTED = "granted"
    DENIED = "denied"


def _qt_permission_dialogs_available() -> bool:
    # An unbundled `python main.py` on macOS has no Info.plist usage descriptions, so Qt
    # denies without prompting. In that case AVFoundation prompts (for the Terminal)
    # when OpenCV opens the device, and a refusal surfaces as a camera open failure.
    return sys.platform != "darwin" or bool(getattr(sys, "frozen", False))


def request_permission(
    kind: str,
    on_result: Callable[[PermissionState], None],
    context: QObject | None = None,
) -> None:
    """Ask the OS for camera or microphone access and report the outcome asynchronously."""
    if not _qt_permission_dialogs_available():
        on_result(PermissionState.GRANTED)
        return

    app = QCoreApplication.instance()
    permission = QCameraPermission() if kind == "camera" else QMicrophonePermission()

    def to_state(status: Qt.PermissionStatus) -> PermissionState:
        if status == Qt.PermissionStatus.Granted:
            return PermissionState.GRANTED
        return PermissionState.DENIED

    status = app.checkPermission(permission)
    if status != Qt.PermissionStatus.Undetermined:
        on_result(to_state(status))
        return
    app.requestPermission(
        permission,
        context or app,
        lambda result: on_result(to_state(app.checkPermission(result))),
    )


def open_privacy_settings(kind: str) -> bool:
    return QDesktopServices.openUrl(QUrl(PRIVACY_SETTINGS_URLS[kind]))


class VideoCapture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


def open_opencv_capture(index: int) -> VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)


class SyntheticCapture:
    """Animated test pattern used with `--fake-camera` and in tests (no webcam needed)."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._frame = 0
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._open:
            return False, None
        self._frame += 1
        t = self._frame / 30
        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = x
        frame[:, :, 1] = 80
        frame[:, :, 2] = 255 - x
        cx = int(self.width / 2 + math.sin(t) * self.width / 4)
        cy = self.height // 2
        cv2.circle(frame, (cx, cy), self.height // 5, (200, 220, 255), -1)
        cv2.putText(
            frame, "CAMARA SINTETICA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        return True, frame

    def release(self) -> None:
        self._open = False


class ImageCapture:
    """Serves a still photo as if it were a webcam, to try emotion detection without a camera."""

    def __init__(self, path: str) -> None:
        self._image = cv2.imread(path)
        if self._image is None:
            raise ValueError(f"No se ha podido leer la imagen {path!r}")
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def read(self) -> tuple[bool, np.ndarray | None]:
        return (True, self._image.copy()) if self._open else (False, None)

    def release(self) -> None:
        self._open = False


class CameraWorker(QThread):
    """Reads frames on a background thread. Frames stay in memory and are never stored."""

    frame_ready = Signal(object)
    camera_opened = Signal()
    failed = Signal(str)

    def __init__(
        self,
        camera_index: int = 0,
        capture_factory: Callable[[int], VideoCapture] = open_opencv_capture,
        fps: float = 30.0,
        open_timeout_seconds: float = 10.0,
        max_failed_reads: int = 30,
        mirror: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self._capture_factory = capture_factory
        self._frame_interval = 1.0 / fps
        self._open_timeout = open_timeout_seconds
        self._max_failed_reads = max_failed_reads
        self._mirror = mirror
        self._running = False

    def start(self, *args) -> None:
        self._running = True
        super().start(*args)

    def stop(self) -> None:
        self._running = False
        self.wait()

    def _open(self) -> VideoCapture | None:
        # While macOS shows its first-run permission prompt the device reports closed,
        # so keep retrying until the user answers or the timeout expires.
        deadline = time.monotonic() + self._open_timeout
        while self._running:
            capture = self._capture_factory(self.camera_index)
            if capture.isOpened():
                return capture
            capture.release()
            if time.monotonic() >= deadline:
                return None
            self.msleep(500)
        return None

    def run(self) -> None:
        capture = self._open()
        if capture is None:
            if self._running:
                self.failed.emit(CAMERA_UNAVAILABLE_MESSAGE.format(index=self.camera_index))
            self._running = False
            return

        self.camera_opened.emit()
        failed_reads = 0
        try:
            while self._running:
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= self._max_failed_reads:
                        self.failed.emit(CAMERA_LOST_MESSAGE)
                        break
                else:
                    failed_reads = 0
                    self.frame_ready.emit(cv2.flip(frame, 1) if self._mirror else frame)
                remaining = self._frame_interval - (time.monotonic() - started)
                if remaining > 0:
                    self.msleep(int(remaining * 1000))
        finally:
            capture.release()
            self._running = False
