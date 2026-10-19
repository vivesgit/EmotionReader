"""Sliding window of emotion readings summarised into an EmotionSummary."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from emotion_reader.detector import EmotionReading

MIN_SECONDARY_SCORE = 0.05
MAX_SECONDARY = 2
TREND_THRESHOLD = 0.1
MIN_FRAMES_FOR_TREND = 4


@dataclass(frozen=True)
class EmotionSummary:
    window_seconds: float
    total_frames: int
    frames_with_face: int
    dominant: str | None = None
    dominant_confidence: float = 0.0
    dominant_share: float = 0.0
    secondary: tuple[tuple[str, float], ...] = ()
    trend: str = "sin datos suficientes"
    valence_start: float = 0.0
    valence_end: float = 0.0

    @property
    def has_face(self) -> bool:
        return self.dominant is not None


class EmotionAggregator:
    def __init__(self, window_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_seconds = window_seconds
        self._clock = clock
        self._readings: deque[EmotionReading] = deque()

    def add(self, reading: EmotionReading) -> None:
        self._readings.append(reading)
        self._discard_older_than(self._clock() - self.window_seconds)

    def clear(self) -> None:
        self._readings.clear()

    def _discard_older_than(self, cutoff: float) -> None:
        while self._readings and self._readings[0].timestamp < cutoff:
            self._readings.popleft()

    def summary(self) -> EmotionSummary:
        self._discard_older_than(self._clock() - self.window_seconds)
        readings = list(self._readings)
        faces = [r for r in readings if r.face_detected]
        base = EmotionSummary(self.window_seconds, len(readings), len(faces))
        if not faces:
            return base

        labels = faces[0].scores.keys()
        means = {label: sum(r.scores[label] for r in faces) / len(faces) for label in labels}
        ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)
        dominant, dominant_confidence = ranked[0]
        share = sum(1 for r in faces if r.emotion == dominant) / len(faces)
        secondary = tuple(
            (label, score) for label, score in ranked[1:] if score >= MIN_SECONDARY_SCORE
        )[:MAX_SECONDARY]

        trend, start, end = "sin datos suficientes", 0.0, 0.0
        if len(faces) >= MIN_FRAMES_FOR_TREND:
            half = len(faces) // 2
            start = sum(r.valence for r in faces[:half]) / half
            end = sum(r.valence for r in faces[half:]) / (len(faces) - half)
            if end - start > TREND_THRESHOLD:
                trend = "mejorando"
            elif start - end > TREND_THRESHOLD:
                trend = "empeorando"
            else:
                trend = "estable"

        return EmotionSummary(
            window_seconds=self.window_seconds,
            total_frames=len(readings),
            frames_with_face=len(faces),
            dominant=dominant,
            dominant_confidence=dominant_confidence,
            dominant_share=share,
            secondary=secondary,
            trend=trend,
            valence_start=start,
            valence_end=end,
        )
