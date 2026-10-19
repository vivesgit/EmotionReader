"""Main window: live video, Preguntar / Empezar nuevo chat buttons and request status."""

from __future__ import annotations

import enum
import time
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import Settings
from emotion_reader.camera import (
    CAMERA_DENIED_MESSAGE,
    CameraWorker,
    PermissionState,
    VideoCapture,
    open_opencv_capture,
    open_privacy_settings,
    request_permission,
)

PRIVACY_NOTICE = (
    "Privacidad: el vídeo se analiza solo en este equipo y nunca se guarda ni se envía. "
    "A GPT-Live solo llega un resumen textual aproximado de tus emociones y tu pregunta."
)
HOLD_TO_TALK_HINT = "Mantén pulsado «Preguntar» mientras hablas"
MIN_HOLD_SECONDS = 0.3


class Status(enum.Enum):
    IDLE = "Listo"
    LISTENING = "Escuchando…"
    PROCESSING = "Procesando…"
    PLAYING = "Reproduciendo…"
    ERROR = "Error"


STATUS_COLORS = {
    Status.IDLE: "#3fb950",
    Status.LISTENING: "#f04a4a",
    Status.PROCESSING: "#f5a524",
    Status.PLAYING: "#00a8e8",
    Status.ERROR: "#f04a4a",
}

STYLESHEET = """
QWidget#root { background: #15181d; }
QLabel { color: #e6e9ee; font-size: 14px; }
QLabel#title { font-size: 20px; font-weight: 600; color: #ffffff; }
QLabel#subtitle { font-size: 13px; color: #8b93a1; }
QLabel#statusPill, QLabel#cameraPill {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 15px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
}
QLabel#cameraPill { background: rgba(10, 12, 16, 0.72); }
QFrame#videoCard { background: #0c0e12; border-radius: 18px; }
QLabel#video { color: #8b93a1; font-size: 15px; background: transparent; }
QFrame#controls { background: #1f242c; border-radius: 22px; }
QLineEdit#question {
    background: #2a313b;
    color: #ffffff;
    border: 2px solid transparent;
    border-radius: 24px;
    padding: 0 20px;
    min-height: 48px;
    font-size: 15px;
}
QLineEdit#question:focus { border-color: #00a8e8; }
QPushButton {
    border: none;
    border-radius: 30px;
    min-height: 60px;
    padding: 0 36px;
    font-size: 17px;
    font-weight: 600;
    color: #ffffff;
}
QPushButton#askButton { background: #0078d4; }
QPushButton#askButton:hover { background: #1a8ae0; }
QPushButton#askButton:pressed { background: #005fa8; }
QPushButton#askButton[recording="true"] { background: #e5343a; }
QPushButton#newChatButton { background: #353c47; }
QPushButton#newChatButton:hover { background: #414955; }
QPushButton#newChatButton:pressed { background: #2b313a; }
QLabel#privacy { color: #6f7785; font-size: 12px; }
"""


class MainWindow(QMainWindow):
    ask_requested = Signal(str)
    voice_question_started = Signal()
    voice_question_finished = Signal()
    new_chat_requested = Signal()

    def __init__(
        self,
        settings: Settings,
        capture_factory: Callable[[int], VideoCapture] = open_opencv_capture,
        request_permission_fn: Callable[..., None] = request_permission,
    ) -> None:
        super().__init__()
        self.settings = settings
        self._capture_factory = capture_factory
        self._request_permission = request_permission_fn
        self.camera_worker: CameraWorker | None = None
        self.message_box: QMessageBox | None = None
        self.status = Status.IDLE
        self.recording = False
        self._recording_started_at = 0.0

        self.setWindowTitle("Emotion Reader")
        self.resize(1000, 780)
        self.setMinimumSize(720, 600)

        root = QWidget(objectName="root")
        root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root.setStyleSheet(STYLESHEET)
        self.setCentralWidget(root)

        title = QLabel("Emotion Reader", objectName="title")
        subtitle = QLabel("Conversación con GPT-Live", objectName="subtitle")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.status_label = QLabel(objectName="statusPill")
        header = QHBoxLayout()
        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.video_label = QLabel("Esperando permiso de cámara…", objectName="video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.camera_indicator = QLabel(objectName="cameraPill")
        video_card = QFrame(objectName="videoCard")
        video_layout = QGridLayout(video_card)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.addWidget(self.video_label, 0, 0)
        video_layout.addWidget(
            self.camera_indicator, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )

        self.question_input = QLineEdit(objectName="question")
        self.question_input.setPlaceholderText(
            "Escribe tu pregunta (opcional) o mantén pulsado «Preguntar» para hablar"
        )
        self.question_input.returnPressed.connect(self._send_text_question)

        self.ask_button = QPushButton("Preguntar", objectName="askButton")
        self.ask_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ask_button.setProperty("recording", False)
        self.ask_button.pressed.connect(self._on_ask_pressed)
        self.ask_button.released.connect(self._on_ask_released)
        self.new_chat_button = QPushButton("Empezar nuevo chat", objectName="newChatButton")
        self.new_chat_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_chat_button.clicked.connect(self.new_chat_requested.emit)

        buttons = QHBoxLayout()
        buttons.setSpacing(14)
        buttons.addWidget(self.ask_button, stretch=3)
        buttons.addWidget(self.new_chat_button, stretch=2)
        controls = QFrame(objectName="controls")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(14)
        controls_layout.addWidget(self.question_input)
        controls_layout.addLayout(buttons)

        self.privacy_label = QLabel(PRIVACY_NOTICE, objectName="privacy")
        self.privacy_label.setWordWrap(True)
        self.privacy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addWidget(video_card, stretch=1)
        layout.addWidget(controls)
        layout.addWidget(self.privacy_label)

        self.set_camera_active(False)
        self.set_status(Status.IDLE)

    # --- camera -------------------------------------------------------------

    def start_camera(self) -> None:
        self._request_permission("camera", self._on_camera_permission, self)

    def _on_camera_permission(self, state: PermissionState) -> None:
        if state is not PermissionState.GRANTED:
            self.video_label.setText("Cámara sin permiso")
            self._show_message(
                "Permiso de cámara denegado", CAMERA_DENIED_MESSAGE, settings_kind="camera"
            )
            return
        self.video_label.setText("Abriendo cámara…")
        worker = CameraWorker(self.settings.camera_index, self._capture_factory, parent=self)
        worker.frame_ready.connect(self.show_frame)
        worker.camera_opened.connect(lambda: self.set_camera_active(True))
        worker.failed.connect(self._on_camera_failed)
        worker.finished.connect(lambda: self.set_camera_active(False))
        self.camera_worker = worker
        worker.start()

    def stop_camera(self) -> None:
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None
        self.set_camera_active(False)

    def _on_camera_failed(self, message: str) -> None:
        self.set_camera_active(False)
        self.video_label.setText("Cámara no disponible")
        self._show_message("Cámara no disponible", message, settings_kind="camera")

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        image = QImage(
            frame_bgr.data, width, height, frame_bgr.strides[0], QImage.Format.Format_BGR888
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def set_camera_active(self, active: bool) -> None:
        self.camera_active = active
        if active:
            self.camera_indicator.setText("● Cámara activa")
            self.camera_indicator.setStyleSheet("color:#ff5c5c;")
        else:
            self.camera_indicator.setText("○ Cámara inactiva")
            self.camera_indicator.setStyleSheet("color:#8b93a1;")

    # --- status and buttons -------------------------------------------------

    def set_status(self, status: Status, detail: str = "") -> None:
        self.status = status
        text = f"● {status.value}"
        if detail:
            text += f" {detail}"
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{STATUS_COLORS[status]};")

    def _on_ask_pressed(self) -> None:
        if self.question_input.text().strip():
            return
        self.recording = True
        self._recording_started_at = time.monotonic()
        self._set_ask_recording_style(True)
        self.set_status(Status.LISTENING)
        self.voice_question_started.emit()

    def _on_ask_released(self) -> None:
        if not self.recording:
            self._send_text_question()
            return
        self.recording = False
        self._set_ask_recording_style(False)
        if time.monotonic() - self._recording_started_at < MIN_HOLD_SECONDS:
            self.set_status(Status.IDLE, f"· {HOLD_TO_TALK_HINT}")
            return
        self.set_status(Status.IDLE)
        self.voice_question_finished.emit()

    def _send_text_question(self) -> None:
        question = self.question_input.text().strip()
        if question:
            self.ask_requested.emit(question)

    def _set_ask_recording_style(self, recording: bool) -> None:
        self.ask_button.setProperty("recording", recording)
        self.ask_button.setText("Escuchando… suelta para enviar" if recording else "Preguntar")
        self.ask_button.style().unpolish(self.ask_button)
        self.ask_button.style().polish(self.ask_button)

    def _show_message(self, title: str, text: str, settings_kind: str | None = None) -> None:
        box = QMessageBox(
            QMessageBox.Icon.Warning, title, text, QMessageBox.StandardButton.Ok, self
        )
        if settings_kind is not None:
            open_button = box.addButton(
                "Abrir Ajustes del Sistema", QMessageBox.ButtonRole.ActionRole
            )
            open_button.clicked.connect(lambda: open_privacy_settings(settings_kind))
        self.message_box = box
        box.open()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_camera()
        super().closeEvent(event)
