import numpy as np
import pytest
from PySide6.QtCore import Qt

from emotion_reader import camera
from emotion_reader.camera import CameraWorker, PermissionState, SyntheticCapture


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
