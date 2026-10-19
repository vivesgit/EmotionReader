"""Emotion Reader entry point."""

from __future__ import annotations

import argparse
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from config import ConfigError, load_settings
from emotion_reader.camera import ImageCapture, SyntheticCapture, open_opencv_capture
from ui.main_window import MainWindow


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emotion Reader + GPT-Live")
    parser.add_argument(
        "--fake-camera",
        nargs="?",
        const="",
        metavar="IMAGEN",
        help="usa un patrón animado (o la foto IMAGEN) en lugar de la webcam",
    )
    return parser.parse_args(argv)


def capture_factory_for(fake_camera: str | None):
    if fake_camera is None:
        return open_opencv_capture
    if fake_camera:
        ImageCapture(fake_camera)
        return lambda _index: ImageCapture(fake_camera)
    return lambda _index: SyntheticCapture()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Error de configuración en .env: {exc}", file=sys.stderr)
        return 1
    try:
        capture_factory = capture_factory_for(args.fake_camera)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Emotion Reader")
    window = MainWindow(settings, capture_factory=capture_factory)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    # Qt's event loop blocks Python signal handlers; waking it periodically lets Ctrl+C work.
    wake_timer = QTimer()
    wake_timer.timeout.connect(lambda: None)
    wake_timer.start(250)
    window.show()
    window.start_camera()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
