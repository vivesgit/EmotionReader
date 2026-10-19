"""Structured prompt built from an EmotionSummary and the user's question."""

from __future__ import annotations

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
