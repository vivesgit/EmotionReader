"""Structured prompt built from an EmotionSummary and the user's question."""

from __future__ import annotations

import textwrap
from datetime import datetime

from emotion_reader.aggregator import EmotionSummary

VOICE_QUESTION = "(la pregunta llega a continuación en el audio del usuario)"


def _confidence_level(value: float) -> str:
    if value >= 0.75:
        return "alta"
    if value >= 0.45:
        return "media"
    return "baja"


def build_emotion_context(summary: EmotionSummary) -> str:
    seconds = f"{summary.window_seconds:g}"
    lines = [f"[Contexto emocional estimado del usuario, últimos {seconds} s, aproximado]"]
    if not summary.has_face:
        lines.append(
            f"- No se ha detectado ninguna cara ({summary.total_frames} frames analizados)."
        )
        lines.append("Responde sin suponer ningún estado emocional.")
        return "\n".join(lines)

    lines.append(
        f"- Emoción dominante: {summary.dominant} "
        f"(confianza {_confidence_level(summary.dominant_confidence)} "
        f"{summary.dominant_confidence:.2f}, presente en el "
        f"{summary.dominant_share:.0%} de los frames)"
    )
    if summary.secondary:
        secondary = ", ".join(f"{label} ({score:.2f})" for label, score in summary.secondary)
        lines.append(f"- Secundarias: {secondary}")
    else:
        lines.append("- Secundarias: ninguna relevante")
    if summary.trend in ("mejorando", "empeorando", "estable"):
        lines.append(
            f"- Tendencia: {summary.trend} "
            f"(valencia {summary.valence_start:+.2f} -> {summary.valence_end:+.2f})"
        )
    else:
        lines.append(f"- Tendencia: {summary.trend}")
    lines.append(f"- Cara detectada en {summary.frames_with_face}/{summary.total_frames} frames")
    lines.append("Adapta tono y empatía a este contexto sin mencionarlo salvo que sea útil.")
    return "\n".join(lines)


def build_prompt(summary: EmotionSummary, question: str | None) -> str:
    """Emotion context plus the question; `None` means it is sent as audio afterwards."""
    question_text = question.strip() if question and question.strip() else VOICE_QUESTION
    return f"{build_emotion_context(summary)}\n[Pregunta del usuario]: {question_text}"


_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"
_CYAN, _YELLOW, _GREEN = "\033[36m", "\033[33m", "\033[32m"


def format_prompt_log(
    prompt: str,
    number: int,
    sent_at: datetime,
    model: str,
    audio_seconds: float = 0.0,
    color: bool = False,
    width: int = 80,
) -> str:
    """Framed, wrapped view of a sent prompt for the terminal running main.py."""

    def paint(text: str, *codes: str) -> str:
        return f"{''.join(codes)}{text}{_RESET}" if color else text

    width = max(40, width)
    via = f"voz, {audio_seconds:.1f} s de audio" if audio_seconds else "texto"
    title = f" Prompt #{number} · {sent_at:%H:%M:%S} · {via} → {model} "
    top = "╭─" + title + "─" * max(0, width - len(title) - 2)
    bar = paint("│", _DIM)
    lines = [paint(top, _DIM)]
    for raw in prompt.splitlines():
        if raw.startswith("[Pregunta del usuario]"):
            label, _, question = raw.partition(":")
            lines.append(bar)
            lines.append(f"{bar} {paint(label, _BOLD, _YELLOW)}")
            for part in textwrap.wrap(question.strip(), width - 4) or [""]:
                lines.append(f"{bar}   {paint(part, _BOLD)}")
        elif raw.startswith("["):
            for part in textwrap.wrap(raw, width - 3):
                lines.append(f"{bar} {paint(part, _BOLD, _CYAN)}")
        else:
            bullet = raw.startswith("- ")
            wrapped = textwrap.wrap(raw, width - 3, subsequent_indent="  " if bullet else "")
            for part in wrapped or [""]:
                lines.append(f"{bar} {paint(part, _GREEN if bullet else _DIM)}")
    lines.append(paint("╰" + "─" * (width - 1), _DIM))
    return "\n".join(lines)
