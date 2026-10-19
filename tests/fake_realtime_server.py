"""Local stand-in for the OpenAI Realtime WebSocket API, used by tests and offline demos.

Run `python tests/fake_realtime_server.py` and start the app with
`OPENAI_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_API_KEY=sk-fake python main.py` to try the
full flow without an OpenAI account. Answers are a short synthetic tone, not real speech.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import struct
from dataclasses import dataclass, field

from websockets.asyncio.server import ServerConnection, serve

SAMPLE_RATE = 24_000


def tone(seconds: float = 0.6, frequency: float = 440.0) -> bytes:
    samples = int(SAMPLE_RATE * seconds)
    return b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE)))
        for i in range(samples)
    )


@dataclass
class FakeRealtimeServer:
    """`behaviour` is persistent: ok, invalid_key, hang, failed_response, rate_limit_always.
    `fail_next` affects only the next response: rate_limit, drop, drop_after_audio."""

    behaviour: str = "ok"
    fail_next: str | None = None
    audio_chunks: int = 3
    answer_seconds: float = 0.6
    answer: str = "Hola, te noto contento. ¿En qué te ayudo?"
    connections: list[list[dict]] = field(default_factory=list)
    responses: int = 0
    port: int = 0
    _server: object = None

    async def start(self, port: int = 0) -> FakeRealtimeServer:
        self._server = await serve(self._handle, "127.0.0.1", port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def _send(self, ws: ServerConnection, event: dict) -> None:
        event.setdefault("event_id", f"evt_{id(event)}")
        await ws.send(json.dumps(event))

    async def _handle(self, ws: ServerConnection) -> None:
        received: list[dict] = []
        self.connections.append(received)
        if self.behaviour == "invalid_key":
            await self._send(
                ws,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                        "message": "Incorrect API key provided.",
                    },
                },
            )
            await ws.close()
            return
        await self._send(
            ws, {"type": "session.created", "session": {"type": "realtime", "id": "sess_1"}}
        )
        committed = False
        async for message in ws:
            event = json.loads(message)
            received.append(event)
            kind = event["type"]
            if kind == "session.update":
                await self._send(ws, {"type": "session.updated", "session": event["session"]})
            elif kind == "input_audio_buffer.commit":
                committed = True
            elif kind == "response.create":
                self.responses += 1
                if await self._respond(ws, committed):
                    return
                committed = False

    async def _respond(self, ws: ServerConnection, committed: bool) -> bool:
        """Send one response; returns True when the connection was closed on purpose."""
        fail, self.fail_next = self.fail_next, None
        if self.behaviour == "hang":
            return False
        if fail == "drop":
            await ws.close(code=1011, reason="simulated outage")
            return True
        if self.behaviour == "rate_limit_always" or fail == "rate_limit":
            await self._send(
                ws,
                {
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                        "message": "Rate limit reached.",
                    },
                },
            )
            return False
        ids = {"response_id": "resp_1", "item_id": "item_1", "output_index": 0}
        await self._send(
            ws, {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}
        )
        if self.behaviour == "failed_response":
            await self._send(
                ws,
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_1",
                        "status": "failed",
                        "status_details": {
                            "type": "failed",
                            "error": {"type": "insufficient_quota", "code": "insufficient_quota"},
                        },
                    },
                },
            )
            return False
        if committed:
            await self._send(
                ws,
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item_user",
                    "content_index": 0,
                    "transcript": "¿Qué tal estoy?",
                },
            )
        pcm = tone(self.answer_seconds)
        size = len(pcm) // self.audio_chunks // 2 * 2
        words = self.answer.split(" ")
        per_chunk = math.ceil(len(words) / self.audio_chunks)
        for i in range(self.audio_chunks):
            chunk = pcm[i * size : (i + 1) * size]
            await self._send(
                ws,
                {
                    "type": "response.output_audio.delta",
                    "content_index": 0,
                    "delta": base64.b64encode(chunk).decode("ascii"),
                    **ids,
                },
            )
            await self._send(
                ws,
                {
                    "type": "response.output_audio_transcript.delta",
                    "content_index": 0,
                    "delta": " ".join(words[i * per_chunk : (i + 1) * per_chunk]) + " ",
                    **ids,
                },
            )
            if fail == "drop_after_audio":
                await ws.close(code=1011, reason="simulated outage")
                return True
        await self._send(
            ws,
            {
                "type": "response.output_audio_transcript.done",
                "content_index": 0,
                "transcript": self.answer,
                **ids,
            },
        )
        await self._send(
            ws, {"type": "response.done", "response": {"id": "resp_1", "status": "completed"}}
        )
        return False


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--behaviour", default="ok")
    parser.add_argument("--seconds", type=float, default=3.0, help="duración del audio")
    return parser


async def _serve_forever(port: int, behaviour: str, seconds: float) -> None:
    server = await FakeRealtimeServer(behaviour=behaviour, answer_seconds=seconds).start(port)
    print(f"Fake Realtime API listening on ws://127.0.0.1:{server.port}/v1/realtime", flush=True)
    await asyncio.Future()


if __name__ == "__main__":
    args = build_server_parser().parse_args()
    asyncio.run(_serve_forever(args.port, args.behaviour, args.seconds))
