from functools import partial

import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QMessageBox

from config import Settings
from emotion_reader.camera import (
    CAMERA_DENIED_MESSAGE,
    CameraWorker,
    PermissionState,
    SyntheticCapture,
)
from emotion_reader.detector import EMOTION_LABELS_ES, EmotionDetector, ModelDownloadError
from emotion_reader.prompt import VOICE_QUESTION
from gpt_live_client.audio import AudioDeviceError
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


class HappyFaceDetector:
    def detect(self, _frame):
        return [(200, 140, 160, 160)]


class FixedClassifier:
    def classify(self, _face):
        return {label: 0.0 for label in EMOTION_LABELS_ES.values()} | {
            "alegría": 0.8,
            "sorpresa": 0.2,
        }


def fake_detector():
    return EmotionDetector(HappyFaceDetector(), FixedClassifier())


def broken_detector():
    raise ModelDownloadError("No se ha podido descargar el modelo de prueba.")


@pytest.fixture
def make_window(qtbot):
    def factory(
        capture_factory=lambda _i: SyntheticCapture(),
        permission=grant,
        detector_factory=fake_detector,
    ):
        window = MainWindow(
            Settings(),
            capture_factory=capture_factory,
            request_permission_fn=permission,
            detector_factory=detector_factory,
        )
        qtbot.addWidget(window)
        window.show()
        return window

    return factory


def test_live_emotion_overlay_and_summary(qtbot, make_window):
    window = make_window()
    window.start_camera()
    qtbot.waitUntil(lambda: window.latest_reading is not None, timeout=3000)
    qtbot.waitUntil(lambda: window.emotion_label.text() == "Alegría · 80%", timeout=3000)

    image = QImage(640, 480, QImage.Format.Format_RGB32)
    image.fill(QColor("black"))
    window._draw_overlay(image)
    assert window.latest_reading.box == (184, 124, 192, 192)
    border = image.pixelColor(184, 220)
    assert (border.red(), border.green(), border.blue()) == (0, 168, 232)
    assert image.pixelColor(280, 220) == QColor("black")

    qtbot.waitUntil(lambda: window.emotion_summary().frames_with_face >= 4, timeout=3000)
    summary = window.emotion_summary()
    assert summary.dominant == "alegría"
    assert summary.window_seconds == Settings().emotion_window_seconds
    window.close()
    assert window.emotion_worker is None


def test_emotion_model_failure_is_reported(qtbot, make_window):
    window = make_window(detector_factory=broken_detector)
    window.start_camera()
    qtbot.waitUntil(lambda: window.message_box is not None, timeout=3000)
    assert window.message_box.windowTitle() == "Detección de emociones no disponible"
    assert "modelo de prueba" in window.message_box.text()
    assert window.emotion_label.text() == "Emociones no disponibles"
    qtbot.waitUntil(lambda: window.camera_active, timeout=2000)


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


# --- GPT-Live flow ----------------------------------------------------------


class FakeAssistant(QObject):
    audio_started = Signal()
    answer_ready = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.asks = []
        self.cancels = 0
        self.new_chats = 0
        self.shutdowns = 0
        self.busy = False

    def ask(self, prompt, audio=None):
        self.asks.append((prompt, audio))
        self.busy = True

    def cancel(self):
        self.cancels += 1
        self.busy = False

    def new_chat(self):
        self.new_chats += 1

    def shutdown(self):
        self.shutdowns += 1


class FakeRecorder:
    def __init__(self, error=None):
        self.error = error
        self.recording = False
        self.starts = 0

    def start(self):
        if self.error:
            raise self.error
        self.starts += 1
        self.recording = True

    def stop(self):
        self.recording = False
        return b"\x01\x00" * 2400


@pytest.fixture
def make_live_window(qtbot):
    def factory(recorder=None, permission=grant):
        assistant = FakeAssistant()
        recorder = recorder or FakeRecorder()
        window = MainWindow(
            Settings(),
            capture_factory=lambda _i: SyntheticCapture(),
            request_permission_fn=permission,
            detector_factory=fake_detector,
            assistant=assistant,
            recorder=recorder,
        )
        qtbot.addWidget(window)
        window.show()
        window.request_microphone()
        return window, assistant, recorder

    return factory


def test_voice_question_records_and_sends_prompt(make_live_window, clock, qtbot):
    window, assistant, recorder = make_live_window()
    window.ask_button.pressed.emit()
    assert recorder.recording and recorder.starts == 1
    assert window.status is Status.LISTENING
    clock[0] += 1.5
    with qtbot.waitSignal(window.prompt_sent) as sent:
        window.ask_button.released.emit()
    assert not recorder.recording
    prompt, audio = assistant.asks[0]
    assert sent.args == [prompt, 0.1]
    assert audio == b"\x01\x00" * 2400
    assert prompt.startswith("[Contexto emocional estimado del usuario, últimos 5 s")
    assert prompt.endswith(f"[Pregunta del usuario]: {VOICE_QUESTION}")
    assert window.status is Status.PROCESSING
    assert window.caption_label.text() == "Tú: (pregunta por voz)"


def test_short_tap_discards_recording(make_live_window, clock):
    window, assistant, recorder = make_live_window()
    window.ask_button.pressed.emit()
    clock[0] += 0.1
    window.ask_button.released.emit()
    assert not recorder.recording
    assert assistant.asks == []


def test_text_question_sends_prompt_and_clears_field(make_live_window, qtbot):
    window, assistant, recorder = make_live_window()
    window.question_input.setText("¿Qué me recomiendas?")
    with qtbot.waitSignal(window.prompt_sent) as sent:
        window.ask_button.click()
    prompt, audio = assistant.asks[0]
    assert sent.args == [prompt, 0.0]
    assert audio is None
    assert prompt.endswith("[Pregunta del usuario]: ¿Qué me recomiendas?")
    assert recorder.starts == 0
    assert window.question_input.text() == ""
    assert window.last_prompt == prompt


def test_status_follows_answer_lifecycle(make_live_window):
    window, assistant, _ = make_live_window()
    window.question_input.setText("hola")
    window.ask_button.click()
    assert window.status is Status.PROCESSING
    assistant.audio_started.emit()
    assert window.status is Status.PLAYING
    assistant.answer_ready.emit("Hola, ¿qué tal?")
    assert window.caption_label.text() == "GPT-Live: Hola, ¿qué tal?"
    assert window.caption_label.isVisible()
    assistant.finished.emit()
    assert window.status is Status.IDLE


def test_assistant_failure_shows_error(make_live_window):
    window, assistant, _ = make_live_window()
    assistant.failed.emit("OpenAI ha rechazado la clave (OPENAI_API_KEY).")
    assert window.status is Status.ERROR
    assert window.message_box.windowTitle() == "GPT-Live no ha podido responder"
    assert "OPENAI_API_KEY" in window.message_box.text()


def test_pressing_while_answering_interrupts(make_live_window):
    window, assistant, recorder = make_live_window()
    assistant.busy = True
    window.ask_button.pressed.emit()
    assert assistant.cancels == 1
    assert recorder.recording


def test_new_chat_resets_conversation(make_live_window):
    window, assistant, recorder = make_live_window()
    window.ask_button.pressed.emit()
    window.question_input.setText("pendiente")
    window.set_caption("GPT-Live: algo")
    window.new_chat_button.click()
    assert assistant.new_chats == 1
    assert not recorder.recording and not window.recording
    assert window.question_input.text() == ""
    assert not window.caption_label.isVisible()
    assert window.status is Status.IDLE
    assert "Nuevo chat" in window.status_label.text()


def test_microphone_denied_shows_instructions(make_live_window):
    window, assistant, recorder = make_live_window(permission=deny)
    window.ask_button.pressed.emit()
    assert recorder.starts == 0
    assert not window.recording
    assert window.message_box.windowTitle() == "Permiso de micrófono denegado"
    assert "Privacidad y seguridad > Micrófono" in window.message_box.text()


def test_missing_microphone_device_is_reported(make_live_window):
    window, _, _ = make_live_window(recorder=FakeRecorder(AudioDeviceError("Sin micrófono.")))
    window.ask_button.pressed.emit()
    assert not window.recording
    assert window.status is Status.ERROR
    assert window.message_box.text() == "Sin micrófono."


def test_close_shuts_down_assistant(make_live_window):
    window, assistant, _ = make_live_window()
    window.close()
    assert assistant.shutdowns == 1


def test_prompt_printer_numbers_each_prompt():
    import io

    from main import prompt_printer

    out = io.StringIO()
    print_prompt = prompt_printer("gpt-realtime-2.1", out)
    print_prompt("[Contexto]\n[Pregunta del usuario]: hola", 0.0)
    print_prompt("[Contexto]\n[Pregunta del usuario]: adiós", 1.5)
    text = out.getvalue()
    assert "Prompt #1" in text and "Prompt #2" in text
    assert "voz, 1.5 s de audio → gpt-realtime-2.1" in text
    assert "\033[" not in text
