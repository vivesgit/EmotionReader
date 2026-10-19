from functools import partial

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from config import Settings
from emotion_reader.camera import (
    CAMERA_DENIED_MESSAGE,
    CameraWorker,
    PermissionState,
    SyntheticCapture,
)
from ui import main_window
from ui.main_window import PRIVACY_NOTICE, MainWindow, Status


def grant(_kind, on_result, _context=None):
    on_result(PermissionState.GRANTED)


def deny(_kind, on_result, _context=None):
    on_result(PermissionState.DENIED)


class ClosedCapture:
    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


@pytest.fixture
def make_window(qtbot):
    def factory(capture_factory=lambda _i: SyntheticCapture(), permission=grant):
        window = MainWindow(
            Settings(), capture_factory=capture_factory, request_permission_fn=permission
        )
        qtbot.addWidget(window)
        window.show()
        return window

    return factory


def test_initial_layout(make_window):
    window = make_window()
    assert window.ask_button.text() == "Preguntar"
    assert window.new_chat_button.text() == "Empezar nuevo chat"
    assert window.privacy_label.text() == PRIVACY_NOTICE
    assert "Cámara inactiva" in window.camera_indicator.text()
    assert window.status_label.text() == "● Listo"


def test_camera_starts_and_shows_frames(qtbot, make_window):
    window = make_window()
    window.start_camera()
    qtbot.waitUntil(lambda: window.camera_active, timeout=2000)
    qtbot.waitUntil(lambda: not window.video_label.pixmap().isNull(), timeout=2000)
    assert "Cámara activa" in window.camera_indicator.text()
    window.close()
    assert window.camera_worker is None
    assert not window.camera_active


def test_permission_denied_shows_instructions(make_window):
    window = make_window(permission=deny)
    window.start_camera()
    assert window.camera_worker is None
    assert not window.camera_active
    box = window.message_box
    assert box is not None and box.isVisible()
    assert box.text() == CAMERA_DENIED_MESSAGE
    assert "Ajustes del Sistema" in box.text()
    labels = [b.text() for b in box.buttons()]
    assert "Abrir Ajustes del Sistema" in labels


def test_open_settings_button_opens_camera_pane(monkeypatch, make_window):
    opened = []
    monkeypatch.setattr(main_window, "open_privacy_settings", opened.append)
    window = make_window(permission=deny)
    window.start_camera()
    button = next(
        b for b in window.message_box.buttons() if b.text() == "Abrir Ajustes del Sistema"
    )
    button.click()
    assert opened == ["camera"]


def test_unavailable_camera_shows_error(qtbot, monkeypatch, make_window):
    monkeypatch.setattr(
        main_window, "CameraWorker", partial(CameraWorker, open_timeout_seconds=0.1)
    )
    window = make_window(capture_factory=lambda _i: ClosedCapture())
    window.start_camera()
    qtbot.waitUntil(lambda: window.message_box is not None, timeout=3000)
    assert window.message_box.windowTitle() == "Cámara no disponible"
    assert window.message_box.icon() == QMessageBox.Icon.Warning
    assert not window.camera_active


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(main_window.time, "monotonic", lambda: now[0])
    return now


def test_click_with_text_sends_text_question(qtbot, make_window):
    window = make_window()
    window.question_input.setText("  ¿Qué tal estoy?  ")
    voice = []
    window.voice_question_started.connect(lambda: voice.append("start"))
    with qtbot.waitSignal(window.ask_requested) as blocker:
        window.ask_button.click()
    assert blocker.args == ["¿Qué tal estoy?"]
    assert voice == []
    assert window.ask_button.property("recording") is False


def test_enter_in_question_field_sends_text(qtbot, make_window):
    window = make_window()
    window.question_input.setText("Hola")
    with qtbot.waitSignal(window.ask_requested) as blocker:
        window.question_input.returnPressed.emit()
    assert blocker.args == ["Hola"]


def test_hold_to_talk_turns_button_red_while_pressed(qtbot, make_window, clock):
    window = make_window()
    texts = []
    window.ask_requested.connect(texts.append)

    with qtbot.waitSignal(window.voice_question_started):
        window.ask_button.pressed.emit()
    assert window.recording
    assert window.ask_button.property("recording") is True
    assert window.ask_button.text().startswith("Escuchando")
    assert window.status is Status.LISTENING

    clock[0] += 2.0
    with qtbot.waitSignal(window.voice_question_finished):
        window.ask_button.released.emit()
    assert not window.recording
    assert window.ask_button.property("recording") is False
    assert window.ask_button.text() == "Preguntar"
    assert texts == []


def test_real_mouse_hold_records(qtbot, make_window, clock):
    window = make_window()
    qtbot.mousePress(window.ask_button, Qt.MouseButton.LeftButton)
    assert window.ask_button.property("recording") is True
    clock[0] += 1.0
    with qtbot.waitSignal(window.voice_question_finished):
        qtbot.mouseRelease(window.ask_button, Qt.MouseButton.LeftButton)
    assert window.ask_button.property("recording") is False


def test_short_tap_without_text_shows_hold_hint(qtbot, make_window, clock):
    window = make_window()
    finished = []
    window.voice_question_finished.connect(lambda: finished.append(True))
    window.ask_button.pressed.emit()
    clock[0] += 0.1
    window.ask_button.released.emit()
    assert finished == []
    assert window.status is Status.IDLE
    assert main_window.HOLD_TO_TALK_HINT in window.status_label.text()
    assert window.ask_button.property("recording") is False


def test_new_chat_button_emits_signal(qtbot, make_window):
    window = make_window()
    with qtbot.waitSignal(window.new_chat_requested):
        window.new_chat_button.click()


@pytest.mark.parametrize("status", list(Status))
def test_status_label(make_window, status):
    window = make_window()
    window.set_status(status, "detalle")
    assert window.status_label.text() == f"● {status.value} detalle"
    assert window.status is status
