"""Emotion Reader entry point."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from config import ConfigError, load_settings
from emotion_reader.camera import SyntheticCapture, open_opencv_capture
from ui.main_window import MainWindow


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emotion Reader + GPT-Live")
    parser.add_argument(
        "--fake-camera",
        action="store_true",
        help="usa un patrón animado en lugar de la webcam (pruebas sin cámara)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Error de configuración en .env: {exc}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Emotion Reader")
    capture_factory = (
        (lambda _index: SyntheticCapture()) if args.fake_camera else (open_opencv_capture)
    )
    window = MainWindow(settings, capture_factory=capture_factory)
    window.show()
    window.start_camera()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
