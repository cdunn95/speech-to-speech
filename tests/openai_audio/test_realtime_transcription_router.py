"""Tests for the realtime streaming transcription endpoint.

Exercises ``POST /realtime/transcription`` (see
``speech_to_speech.api.openai_audio.realtime_router``) with a mock engine:
chunked PCM in, Server-Sent Events out.

Two transports are used:

- ``httpx.AsyncClient`` + ``ASGITransport`` for *live-feed* tests: the
  request body is pulled from an async generator on demand, so the server
  genuinely sees the audio arrive over time.
- Starlette's synchronous ``TestClient`` for one-shot bodies and error
  cases (it reads the whole body before the app runs).
"""

import asyncio
import contextlib
import json
import time
from urllib.parse import urlsplit

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from starlette.routing import Route
from starlette.testclient import TestClient

from speech_to_speech.api.openai_audio.audio_engine import TranscriptionResult
from speech_to_speech.api.openai_audio.realtime_router import (
    RealtimeTranscriptionEndpoint,
    create_realtime_transcription_router,
)

BASE = "/realtime/transcription"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEngine:
    """Stand-in for ``AudioAPIEngine``: records calls, returns canned text."""

    def __init__(self, sample_rate: int = 16_000, has_stt: bool = True, delay: float = 0.0) -> None:
        self.sample_rate = sample_rate
        self._has_stt = has_stt
        self.delay = delay
        self.calls: list[tuple[int, str | None, str | None]] = []

    @property
    def has_stt(self) -> bool:
        return self._has_stt

    def transcribe(self, audio, language=None, prompt=None) -> TranscriptionResult:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append((len(audio), language, prompt))
        return TranscriptionResult(text=f"segment {len(self.calls)}", language="en", duration_s=0.01)


class FlakyEngine(MockEngine):
    """Fails exactly once, then behaves like ``MockEngine``."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._failed = False

    def transcribe(self, audio, language=None, prompt=None):
        if not self._failed:
            self._failed = True
            raise RuntimeError("boom")
        return super().transcribe(audio, language, prompt)


def make_app(engine: MockEngine, **endpoint_kwargs) -> FastAPI:
    app = FastAPI()
    app.include_router(create_realtime_transcription_router(engine, **endpoint_kwargs))
    return app


def pcm_bytes(seconds: float, sample_rate: int = 16_000) -> bytes:
    """Silent s16le mono PCM."""
    return np.zeros(int(seconds * sample_rate), dtype=np.int16).tobytes()


@pytest.fixture(autouse=True)
def _drain_shared_executor():
    """The endpoint shares one single-worker executor across tests.  A
    cancelled inference that already started keeps its worker until it
    finishes, so drain it between tests to keep timing-based assertions
    deterministic."""
    yield
    import speech_to_speech.api.openai_audio.realtime_router as rr

    rr._realtime_executor.submit(lambda: None).result(timeout=30)


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE payload into ``(event_name, data_dict)`` tuples."""
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue  # keep-alive comment / empty
        event = "message"
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        if data_lines:
            events.append((event, json.loads("\n".join(data_lines))))
    return events


async def afeed(seconds: float, chunk_s: float = 0.1, sample_rate: int = 16_000):
    """Async generator simulating a live microphone feed.

    Each chunk is yielded after its own audio duration has elapsed, so the
    server sees the audio arrive over the same wall-clock time it
    represents.
    """
    pcm = pcm_bytes(seconds, sample_rate)
    chunk = int(chunk_s * sample_rate * 2)
    for i in range(0, len(pcm), chunk):
        await asyncio.sleep(chunk_s)
        yield pcm[i : i + chunk]


async def async_session(engine: MockEngine, url: str, body, app: FastAPI | None = None) -> list[tuple[str, dict]]:
    transport = httpx.ASGITransport(app=app or make_app(engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", url, content=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            raw = (await resp.aread()).decode()
    return parse_sse(raw)


def sync_session(engine: MockEngine, url: str, content, app: FastAPI | None = None) -> list[tuple[str, dict]]:
    client = TestClient(app or make_app(engine))
    with client.stream("POST", url, content=content) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = resp.read().decode()
    return parse_sse(raw)


def http_scope(url: str) -> dict:
    parts = urlsplit(url)
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": parts.path,
        "query_string": parts.query.encode("ascii"),
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }


async def raw_session(
    engine: MockEngine,
    url: str,
    chunks: list[bytes],
    *,
    disconnect_after: int | None = None,
) -> list[tuple[str, dict]]:
    """Drive the endpoint with a hand-rolled ASGI harness.

    *chunks* are delivered one per ``receive()``; when they are exhausted
    ``receive()`` returns ``http.disconnect``.  *disconnect_after* forces a
    disconnect once that many chunks have been delivered.
    """
    app = make_app(engine)
    scope = http_scope(url)
    delivered = 0

    async def receive():
        nonlocal delivered
        if disconnect_after is not None and delivered >= disconnect_after:
            return {"type": "http.disconnect"}
        if delivered >= len(chunks):
            return {"type": "http.disconnect"}
        body = chunks[delivered]
        delivered += 1
        return {"type": "http.request", "body": body, "more_body": True}

    out: list[dict] = []

    async def send(message: dict) -> None:
        out.append(message)

    await app(scope, receive, send)
    raw = b"".join(m.get("body", b"") for m in out if m["type"] == "http.response.body").decode()
    return parse_sse(raw)


# ---------------------------------------------------------------------------
# Live-feed tests (async transport: body arrives over time)
# ---------------------------------------------------------------------------


async def test_live_session_emits_started_transcript_and_done():
    engine = MockEngine()
    # chunk_seconds=0.3 with 0.1 s feed chunks: by the 0.6 s tick the
    # buffer always holds >= 0.5 s of audio (the transcribe minimum), so
    # at least one live transcription is guaranteed regardless of
    # tick/chunk scheduling races.
    events = await async_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=0.3&window_seconds=3",
        afeed(1.2),
    )
    names = [name for name, _ in events]

    assert names[0] == "started"
    assert names[-1] == "done"
    assert names.count("transcript") >= 2  # live ticks + final flush

    started = events[0][1]
    assert started["sample_rate"] == 16000
    assert started["chunk_seconds"] == 0.3
    assert started["request_id"]

    done = events[-1][1]
    assert done["chunks"] == names.count("transcript")
    assert done["total_seconds"] == pytest.approx(1.2, abs=0.05)

    # Every inference ran on non-empty audio.
    assert engine.calls
    for samples, _lang, _prompt in engine.calls:
        assert samples > 0

    # Transcript events carry a sane, bounded window.
    for _name, data in events:
        if "text" in data:
            assert data["duration"] > 0
            assert data["duration"] <= 3.0 + 0.1
            assert data["end"] >= data["start"]


async def test_rolling_window_trims_old_audio():
    """The buffer never grows past the window: 5 s streamed with a 3 s
    window must transcribe at most 3 s of audio."""
    engine = MockEngine()
    events = await async_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=1&window_seconds=3",
        afeed(5.0, chunk_s=0.1),
    )
    assert [name for name, _ in events][-1] == "done"
    assert engine.calls
    samples = engine.calls[-1][0]
    assert 40_000 <= samples <= 48_000  # between 2.5 s and 3.0 s @ 16 kHz


async def test_slow_engine_does_not_pile_up_inferences():
    """With a slow engine and fast ticks, at most one inference is in
    flight at a time (later ticks are skipped, not queued)."""
    engine = MockEngine(delay=0.3)
    events = await async_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=0.1&window_seconds=3",
        afeed(1.0, chunk_s=0.05),
    )
    names = [name for name, _ in events]
    assert names[-1] == "done"
    # ~10 ticks over 1 s, each inference takes 0.3 s → far fewer than 10
    # inferences must have run if the in-flight guard works.
    assert 1 <= len(engine.calls) <= 6


# ---------------------------------------------------------------------------
# One-shot / error tests (sync TestClient: whole body up front)
# ---------------------------------------------------------------------------


def test_one_shot_body_produces_final_transcript():
    """A body sent all at once ends immediately and gets exactly one
    final-flush transcription."""
    engine = MockEngine()
    events = sync_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3",
        iter([pcm_bytes(1.0)]),
    )
    names = [name for name, _ in events]
    assert names == ["started", "transcript", "done"]
    assert engine.calls
    assert engine.calls[0][0] == 16_000  # 1.0 s @ 16 kHz


def test_resamples_client_rate_to_engine_rate():
    engine = MockEngine(sample_rate=8_000)
    events = sync_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3",
        iter([pcm_bytes(1.0, sample_rate=16_000)]),  # 32 000 samples @ 16 kHz
    )
    assert [name for name, _ in events][-1] == "done"
    assert engine.calls
    assert engine.calls[0][0] == pytest.approx(8_000, rel=0.05)  # 1 s @ 8 kHz


def test_language_and_prompt_forwarded():
    engine = MockEngine()
    sync_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3&language=fr&prompt=hello",
        iter([pcm_bytes(0.6)]),
    )
    assert engine.calls
    _samples, language, prompt = engine.calls[0]
    assert language == "fr"
    assert prompt == "hello"


def test_inference_failure_emits_error_and_session_continues():
    engine = FlakyEngine()
    events = sync_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3",
        iter([pcm_bytes(1.0)]),
    )
    names = [name for name, _ in events]
    assert "error" in names
    assert names[-1] == "done"  # session survived the failure
    error_data = next(data for name, data in events if name == "error")
    assert "boom" not in error_data["message"]  # internal details stay server-side
    assert error_data["request_id"]


def test_stt_unavailable_returns_501():
    engine = MockEngine(has_stt=False)
    client = TestClient(make_app(engine))

    resp = client.post(BASE, content=b"")
    assert resp.status_code == 501
    body = resp.json()
    assert body["error"]["code"] == "stt_not_available"


def test_invalid_sample_rate_rejected():
    engine = MockEngine()
    client = TestClient(make_app(engine))

    resp = client.post(f"{BASE}?sample_rate=100", content=b"")
    assert resp.status_code == 400

    resp = client.post(f"{BASE}?sample_rate=999000", content=b"")
    assert resp.status_code == 400

    resp = client.post(f"{BASE}?chunk_seconds=99", content=b"")
    assert resp.status_code == 400


def test_language_and_prompt_length_rejected():
    engine = MockEngine()
    client = TestClient(make_app(engine))

    assert client.post(f"{BASE}?language={'x' * 65}", content=b"").status_code == 400
    assert client.post(f"{BASE}?prompt={'x' * 2049}", content=b"").status_code == 400


def test_blank_language_and_prompt_treated_as_unset():
    engine = MockEngine()
    sync_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3&language=&prompt=",
        iter([pcm_bytes(0.6)]),
    )
    assert engine.calls
    _samples, language, prompt = engine.calls[0]
    assert language is None
    assert prompt is None


def test_window_clamped_to_chunk_seconds():
    """A window shorter than one tick would drop audio from coverage, so
    the server clamps it up and reports the effective value."""
    engine = MockEngine()
    events = sync_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3",
        iter([pcm_bytes(1.0)]),
    )
    assert events[0][1]["window_seconds"] == 5.0


def test_empty_body_emits_started_and_done():
    engine = MockEngine()
    events = sync_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3",
        iter([b""]),
    )
    names = [name for name, _ in events]
    assert names == ["started", "done"]
    assert events[-1][1]["chunks"] == 0
    assert events[-1][1]["reason"] == "completed"
    assert engine.calls == []


# ---------------------------------------------------------------------------
# Lifecycle tests (raw ASGI harness: disconnect / stall / concurrency)
# ---------------------------------------------------------------------------


async def test_no_inference_starts_after_disconnect():
    """On client disconnect the session ends without a final flush: no
    inference may start once the client is gone."""
    engine = MockEngine(delay=0.2)
    events = await raw_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3",
        [pcm_bytes(0.15), pcm_bytes(0.15)],
        disconnect_after=1,
    )
    names = [name for name, _ in events]
    assert names[0] == "started"
    assert names[-1] == "done"
    assert "transcript" not in names  # 0.15 s < 0.5 s minimum; flush skipped
    assert events[-1][1]["reason"] == "disconnected"
    assert engine.calls == []


async def test_no_repeated_inference_on_stalled_feed():
    """Ticks keep firing while the client is silent, but no new inference
    is submitted without new audio (compute-amplification guard)."""
    engine = MockEngine()
    app = make_app(engine)
    scope = http_scope(f"{BASE}?sample_rate=16000&chunk_seconds=0.2&window_seconds=3")
    chunks = [pcm_bytes(0.1)] * 6  # 0.6 s total, then silence
    delivered = 0

    async def receive():
        nonlocal delivered
        if delivered < len(chunks):
            body = chunks[delivered]
            delivered += 1
            return {"type": "http.request", "body": body, "more_body": True}
        await asyncio.Event().wait()  # stall: connected, no audio

    out: list[dict] = []

    async def send(message: dict) -> None:
        out.append(message)

    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.sleep(1.5)  # ~7 ticks at 0.2 s
    assert len(engine.calls) == 1  # one window, no re-runs on the same audio
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_idle_timeout_ends_session():
    engine = MockEngine()
    app = make_app(engine, idle_timeout_s=0.4)
    transport = httpx.ASGITransport(app=app)

    async def feed():
        yield pcm_bytes(0.2)
        await asyncio.sleep(3.0)  # goes silent

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        started_at = time.monotonic()
        async with client.stream(
            "POST", f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3", content=feed()
        ) as resp:
            assert resp.status_code == 200
            raw = (await resp.aread()).decode()
        elapsed = time.monotonic() - started_at

    events = parse_sse(raw)
    names = [name for name, _ in events]
    assert names[-1] == "done"
    assert events[-1][1]["reason"] == "idle_timeout"
    assert elapsed < 3.0  # ended by the idle guard, not by the feed


async def test_max_sessions_rejects_overflow():
    engine = MockEngine()
    endpoint = RealtimeTranscriptionEndpoint(engine, max_sessions=1)
    app = FastAPI()
    app.router.routes.append(Route(BASE, endpoint=endpoint, methods=["POST"]))
    scope = http_scope(f"{BASE}?sample_rate=16000")
    release = asyncio.Event()

    async def receive():
        await release.wait()
        return {"type": "http.disconnect"}

    out: list[dict] = []

    async def send(message: dict) -> None:
        out.append(message)

    first = asyncio.create_task(app(scope, receive, send))
    await asyncio.sleep(0.1)  # let the first session claim its slot

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(BASE, content=b"")
        resp2 = await client.post(BASE, content=b"")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "too_many_sessions"
    assert resp2.status_code == 503  # the cap survives rejections

    assert endpoint._active_sessions == 1  # no double-release

    release.set()
    await first

    # The slot is released: a new session is accepted now.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", BASE, content=pcm_bytes(0.6)) as resp:
            assert resp.status_code == 200
            await resp.aread()


async def test_unexpected_message_type_does_not_stall_session():
    """A non-request/disconnect message must not stall the session: the
    handler re-arms receive() and keeps going."""
    engine = MockEngine()
    app = make_app(engine)
    scope = http_scope(f"{BASE}?sample_rate=16000&chunk_seconds=5&window_seconds=3")
    messages = [
        {"type": "http.request", "body": pcm_bytes(0.6), "more_body": True},
        {"type": "http.something-unknown"},
        {"type": "http.disconnect"},
    ]
    idx = 0

    async def receive():
        nonlocal idx
        message = messages[idx]
        idx += 1
        return message

    out: list[dict] = []

    async def send(message: dict) -> None:
        out.append(message)

    await app(scope, receive, send)
    raw = b"".join(m.get("body", b"") for m in out if m["type"] == "http.response.body").decode()
    names = [name for name, _ in parse_sse(raw)]
    assert names[0] == "started"
    assert names[-1] == "done"


async def test_final_flush_timeout_emits_error_and_done():
    """The body ends while an inference is in flight; the flush gives up
    after final_flush_timeout_s and reports an error."""
    engine = MockEngine(delay=1.0)
    # First submission happens at the 0.6 s tick (0.6 s buffered); the
    # body ends at ~0.8 s, well inside the 1.0 s inference.
    events = await async_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=0.2&window_seconds=3",
        afeed(0.8),
        app=make_app(engine, final_flush_timeout_s=0.2),
    )
    names = [name for name, _ in events]
    assert "error" in names
    assert names[-1] == "done"
    error_data = next(data for name, data in events if name == "error")
    assert "timed out" in error_data["message"]


async def test_tick_inference_failure_emits_error_and_session_continues():
    """A failed periodic inference emits an error event and the session
    keeps transcribing on the next tick."""

    engine = FlakyEngine()
    events = await async_session(
        engine,
        f"{BASE}?sample_rate=16000&chunk_seconds=0.3&window_seconds=3",
        afeed(1.2),
    )
    names = [name for name, _ in events]
    assert "error" in names
    assert names[-1] == "done"
    assert names.count("transcript") >= 1  # session recovered after the failure
