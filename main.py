"""Emotion Reader entry point."""

from __future__ import annotations

import sys

from config import ConfigError, load_settings


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Error de configuración en .env: {exc}", file=sys.stderr)
        return 1
    print(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
