"""Emotion Reader entry point."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
from collections.abc import Callable
from datetime import datetime
from typing import TextIO

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from config import ConfigError, load_settings
from emotion_reader.camera import ImageCapture, SyntheticCapture, open_opencv_capture
from emotion_reader.prompt import format_prompt_log
from gpt_live_client.audio import AudioPlayer, MicRecorder
from gpt_live_client.client import GptLiveBridge, RealtimeClient
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


def prompt_printer(model: str, stream: TextIO | None = None) -> Callable[[str, float], None]:
    """Prints each prompt sent to GPT-Live, framed and coloured when the terminal allows it."""
    count = 0

    def print_prompt(prompt: str, audio_seconds: float) -> None:
        nonlocal count
        count += 1
        out = stream or sys.stdout
        color = out.isatty() and "NO_COLOR" not in os.environ
        width = min(shutil.get_terminal_size().columns, 100)
        text = format_prompt_log(
            prompt, count, datetime.now(), model, audio_seconds, color=color, width=width
        )
        print(f"\n{text}", file=out, flush=True)

    return print_prompt


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
    assistant = GptLiveBridge(RealtimeClient(settings), AudioPlayer())
    window = MainWindow(
        settings, capture_factory=capture_factory, assistant=assistant, recorder=MicRecorder()
    )
    window.prompt_sent.connect(prompt_printer(settings.gpt_live_model))
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
