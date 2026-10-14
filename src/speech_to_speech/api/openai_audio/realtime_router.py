"""FastAPI router exposing a realtime (streaming) transcription endpoint.

``POST /realtime/transcription`` accepts a *live* audio feed: the client
opens a chunked HTTP request and streams raw 16-bit little-endian PCM mono
samples for as long as it likes (minutes, hours, ...).  The server replies
with a Server-Sent Events (SSE) stream that emits a ``transcript`` event
every ``chunk_seconds`` containing the transcription of a rolling window of
the most recent audio, so a session can run indefinitely without ever
buffering the whole stream in memory.

The original request/response endpoints (``/v1/audio/transcriptions`` etc.)
are unchanged; this router is an additional, independent API.

Wire protocol
-------------
Request::

    POST /realtime/transcription?sample_rate=16000&chunk_seconds=3
         &window_seconds=30&language=en&prompt=...
    Transfer-Encoding: chunked
    <raw s16le mono PCM, streamed indefinitely>

Response (``text/event-stream``)::

    event: started
    data: {"request_id": "...", "sample_rate": 16000, "chunk_seconds": 3.0,
           "window_seconds": 30.0}

    event: transcript            (repeated, ~every chunk_seconds)
    data: {"chunk_index": 1, "text": "...", "language": "en",
           "start": 27.0, "end": 30.0, "duration": 30.0}

    event: done                  (after the body ends, the client
                                  disconnects, or the session goes idle)
    data: {"request_id": "...", "chunks": 41, "total_seconds": 123.4,
           "reason": "completed" | "disconnected" | "idle_timeout"}

    event: error                 (non-fatal; the session continues)
    data: {"message": "...", "request_id": "..."}

Consecutive ``transcript`` events cover *overlapping* windows (each window
ends where the audio had arrived when it was transcribed), so clients should
treat each event as the latest live caption rather than appending every
event verbatim.

A pre-recorded file sent all at once (with ``Content-Length``) also works:
the body ends immediately and a single final transcription is emitted.

Session lifecycle
-----------------
- If the client disconnects, the session ends immediately: no final
  transcription is started and any in-flight inference is abandoned.
- A session that receives no audio for ``idle_timeout_s`` (default 120 s)
  is ended with ``reason: idle_timeout``.
- No new inference is submitted while no new audio has arrived since the
  previous one, so a stalled feed cannot burn compute re-transcribing the
  same window.
- Audio that arrives after the last periodic snapshot is transcribed in a
  final flush when the body ends, so a "stop" button does not drop the tail
  (unless the in-flight inference failed or timed out).
- At most ``max_sessions`` (default 8) sessions run concurrently per
  endpoint instance; extra requests are rejected with 503.

Implementation notes
--------------------
The endpoint is a *raw ASGI* app (not a ``StreamingResponse``) on purpose:
Starlette's ``StreamingResponse`` runs a ``listen_for_disconnect`` task
that calls ``receive()`` in a loop, which races with (and in test
transports deadlocks against) a handler that reads the request body
stream.  Owning the ``receive()``/``send()`` loop directly lets the handler
interleave body reads with periodic transcription ticks and guarantees
``receive()`` is never called after the body is complete.

All realtime inference runs on a dedicated single-worker executor: the
engine serializes STT on an internal lock anyway, so one worker is enough
and it keeps the loop's default executor free for the request/response
endpoints.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs

import numpy as np
from fastapi import APIRouter
from starlette.routing import Route
from starlette.types import Message, Receive, Send

from speech_to_speech.api.openai_audio.audio_engine import AudioAPIEngine
from speech_to_speech.api.openai_audio.logging_config import get_loggers
from speech_to_speech.api.openai_audio.router import _openai_error, _safe

# ── constants ─────────────────────────────────────────────────────────

DEFAULT_SAMPLE_RATE = 16_000
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 96_000

DEFAULT_CHUNK_SECONDS = 3.0
MIN_CHUNK_SECONDS = 0.1
MAX_CHUNK_SECONDS = 15.0

DEFAULT_WINDOW_SECONDS = 30.0
MIN_WINDOW_SECONDS = 3.0
MAX_WINDOW_SECONDS = 120.0

MIN_TRANSCRIBE_SECONDS = 0.5  # don't run STT on less than this much audio
HEARTBEAT_SECONDS = 15.0  # SSE keep-alive comment interval
FINAL_FLUSH_TIMEOUT_S = 60.0  # max wait for an in-flight inference at EOF
IDLE_TIMEOUT_S = 120.0  # end the session after this much silence
MAX_CONCURRENT_SESSIONS = 8
# Defense in depth: with the validated parameter ranges the window term
# below is always the smaller one (120 s * 96 kHz * 2 = 23 MB < 32 MB).
MAX_BUFFER_BYTES = 32 * 1024 * 1024

# All realtime inference is serialized on the engine's STT lock, so a single
# dedicated worker is sufficient and keeps the loop's default executor free
# for the request/response endpoints.
_realtime_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rt-transcribe")


# ── helpers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Params:
    sample_rate: int
    language: str | None
    prompt: str | None
    chunk_seconds: float
    window_seconds: float


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    """Format one Server-Sent Event (json.dumps escapes newlines)."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _transcribe_window(
    engine: AudioAPIEngine,
    pcm: bytes,
    sample_rate: int,
    language: str | None,
    prompt: str | None,
) -> tuple[str, str | None]:
    """Decode one PCM snapshot and run STT on it (blocking; run in executor).

    Returns ``(text, language)``.
    """
    if len(pcm) % 2:  # drop an unpaired trailing byte, if any
        pcm = pcm[:-1]
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate != engine.sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=engine.sample_rate)
    result = engine.transcribe(audio, language, prompt)
    return result.text, result.language


def _parse_params(
    scope: Mapping[str, Any],
) -> tuple[_Params | None, tuple[int, bytes, list[tuple[bytes, bytes]]] | None]:
    """Parse and validate the query string.

    Returns ``(params, None)`` on success, or ``(None, (status, body,
    headers))`` describing an error response.
    """
    # Spec-compliant clients percent-encode query strings; latin-1 keeps
    # percent-encoded UTF-8 round-tripping through parse_qs intact.
    qs = parse_qs(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)

    def get(name: str) -> str | None:
        values = qs.get(name)
        return values[0] if values else None

    try:
        sample_rate = int(get("sample_rate") or str(DEFAULT_SAMPLE_RATE))
        if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
            raise ValueError("sample_rate out of range")
        chunk_seconds = float(get("chunk_seconds") or str(DEFAULT_CHUNK_SECONDS))
        if not MIN_CHUNK_SECONDS <= chunk_seconds <= MAX_CHUNK_SECONDS:
            raise ValueError("chunk_seconds out of range")
        window_seconds = float(get("window_seconds") or str(DEFAULT_WINDOW_SECONDS))
        if not MIN_WINDOW_SECONDS <= window_seconds <= MAX_WINDOW_SECONDS:
            raise ValueError("window_seconds out of range")
        # Blank values (e.g. ``?language=``) are treated as unset so they
        # cannot poison every inference with an empty language hint.
        language = get("language") or None
        if language is not None and len(language) > 64:
            raise ValueError("language too long")
        prompt = get("prompt") or None
        if prompt is not None and len(prompt) > 2048:
            raise ValueError("prompt too long")
    except (TypeError, ValueError):
        response = _openai_error(400, "Invalid query parameter", "invalid_request_error")
        return None, (response.status_code, bytes(response.body), list(response.raw_headers))

    # The rolling window must cover at least one tick, otherwise whole
    # stretches of audio would never be transcribed.
    window_seconds = max(window_seconds, chunk_seconds)
    return _Params(sample_rate, language, prompt, chunk_seconds, window_seconds), None


def _sse_headers(request_id: str) -> list[tuple[bytes, bytes]]:
    return [
        (b"content-type", b"text/event-stream; charset=utf-8"),
        (b"cache-control", b"no-cache, no-transform"),
        (b"x-accel-buffering", b"no"),
        (b"x-request-id", request_id.encode("ascii")),
    ]


def _swallow_exception(fut: asyncio.Future) -> None:
    """Retrieve an abandoned executor future's exception so asyncio does not
    log "Future exception was never retrieved" if it is cancelled mid-flight."""
    if not fut.cancelled():
        fut.exception()


# ── raw ASGI endpoint ─────────────────────────────────────────────────


class RealtimeTranscriptionEndpoint:
    """Raw ASGI app implementing ``POST /realtime/transcription``.

    Must be a non-function callable so Starlette's ``Route`` treats it as
    an ASGI app (see module docstring for why ``StreamingResponse`` is not
    used).
    """

    def __init__(
        self,
        engine: AudioAPIEngine,
        *,
        heartbeat_s: float = HEARTBEAT_SECONDS,
        final_flush_timeout_s: float = FINAL_FLUSH_TIMEOUT_S,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
        max_sessions: int = MAX_CONCURRENT_SESSIONS,
    ) -> None:
        if heartbeat_s <= 0:
            raise ValueError("heartbeat_s must be > 0")
        if final_flush_timeout_s <= 0:
            raise ValueError("final_flush_timeout_s must be > 0")
        if idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be > 0")
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.engine = engine
        self.heartbeat_s = heartbeat_s
        self.final_flush_timeout_s = final_flush_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.max_sessions = max_sessions
        self._slot_lock = threading.Lock()
        self._active_sessions = 0
        loggers = get_loggers()
        self.api_log = loggers["api"]
        self.trans_log = loggers["transcription"]
        self.error_log = loggers["errors"]

    async def __call__(self, scope: Message, receive: Receive, send: Send) -> None:
        request_id = str(uuid.uuid4())
        started = perf_counter()

        params, error = _parse_params(scope)

        if error is None and not self.engine.has_stt:
            response = _openai_error(
                501,
                "STT is not available (the stt backend is set to 'none')",
                "server_error",
                code="stt_not_available",
            )
            error = (response.status_code, bytes(response.body), list(response.raw_headers))

        if error is not None:
            status, body, headers = error
            self.api_log.warning("REALTIME-TRANSCRIPTION rejected request=%s status=%d", request_id, status)
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": False})
            return

        # _parse_params returns (params, None) as a pair, so params is set
        # whenever error is not.
        assert params is not None

        # Claim a session slot only if under the limit, so the finally
        # below releases exactly one slot per accepted session (a rejected
        # request holds no slot and returns before the try).
        with self._slot_lock:
            over_limit = self._active_sessions >= self.max_sessions
            if not over_limit:
                self._active_sessions += 1

        if over_limit:
            response = _openai_error(
                503, "Too many concurrent realtime sessions", "server_error", code="too_many_sessions"
            )
            self.api_log.warning("REALTIME-TRANSCRIPTION rejected request=%s status=503 (session limit)", request_id)
            await send({"type": "http.response.start", "status": 503, "headers": list(response.raw_headers)})
            await send({"type": "http.response.body", "body": bytes(response.body), "more_body": False})
            return

        try:
            loop = asyncio.get_running_loop()
            buffer = bytearray()
            max_bytes = min(int(params.window_seconds * params.sample_rate * 2), MAX_BUFFER_BYTES)
            min_bytes = int(MIN_TRANSCRIBE_SECONDS * params.sample_rate * 2)

            await send({"type": "http.response.start", "status": 200, "headers": _sse_headers(request_id)})

            self.api_log.info(
                "REALTIME-TRANSCRIPTION request=%s sample_rate=%s language=%s chunk=%.1fs window=%.1fs",
                request_id,
                params.sample_rate,
                _safe(params.language) if params.language else None,
                params.chunk_seconds,
                params.window_seconds,
            )
            self.trans_log.info(
                "START request=%s sample_rate=%s language=%s chunk=%.1fs window=%.1fs",
                request_id,
                params.sample_rate,
                _safe(params.language) if params.language else None,
                params.chunk_seconds,
                params.window_seconds,
            )

            in_flight: asyncio.Future | None = None
            chunk_index = 0
            total_bytes = 0
            bytes_at_last_submit = 0  # total_bytes when the in-flight snapshot was taken
            snapshot_len = 0  # PCM length covered by the in-flight snapshot
            now = loop.time()
            next_tick = now + params.chunk_seconds
            next_heartbeat = now + self.heartbeat_s
            last_audio_at = now
            reason = "completed"

            async def _emit(event: str, payload: dict[str, Any]) -> None:
                await send({"type": "http.response.body", "body": _sse(event, payload), "more_body": True})

            async def _emit_transcript(text: str, lang: str | None, end_total_bytes: int, window_bytes: int) -> None:
                nonlocal chunk_index
                chunk_index += 1
                end_s = end_total_bytes / 2 / params.sample_rate
                duration_s = window_bytes / 2 / params.sample_rate
                await _emit(
                    "transcript",
                    {
                        "chunk_index": chunk_index,
                        "text": text,
                        "language": lang,
                        "start": round(max(0.0, end_s - duration_s), 3),
                        "end": round(end_s, 3),
                        "duration": round(duration_s, 3),
                    },
                )
                self.trans_log.info(
                    "CHUNK request=%s index=%d text_len=%d position=%.1fs",
                    request_id,
                    chunk_index,
                    len(text),
                    end_s,
                )

            async def _recv() -> Message:
                # receive() returns a plain Awaitable; wrap it in a
                # coroutine so it can be passed to create_task.
                return await receive()

            recv_task: asyncio.Task[Message] = loop.create_task(_recv())
            body_complete = False
            disconnected = False

            try:
                await _emit(
                    "started",
                    {
                        "request_id": request_id,
                        "sample_rate": params.sample_rate,
                        "chunk_seconds": params.chunk_seconds,
                        "window_seconds": params.window_seconds,
                    },
                )

                while not body_complete and not disconnected:
                    # Idle guard: end the session if no audio has arrived
                    # for a while (a stalled trickle must not hold a
                    # session forever).
                    if loop.time() - last_audio_at > self.idle_timeout_s:
                        reason = "idle_timeout"
                        break

                    # Sleep until the next tick, heartbeat, idle deadline,
                    # or incoming body data.
                    idle_deadline = last_audio_at + self.idle_timeout_s
                    timeout = max(0.0, min(next_tick, next_heartbeat, idle_deadline) - loop.time())
                    if timeout > 0 and not recv_task.done():
                        await asyncio.wait({recv_task}, timeout=timeout)

                    # ── process a received message, if any ─────────────
                    if recv_task.done():
                        try:
                            message = recv_task.result()
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            self.trans_log.warning(
                                "READ-ERROR request=%s error=%s (treating as disconnect)",
                                request_id,
                                _safe(str(exc)),
                            )
                            disconnected = True
                            reason = "disconnected"
                            break

                        message_type = message.get("type")
                        if message_type == "http.disconnect":
                            disconnected = True
                            reason = "disconnected"
                            break
                        if message_type != "http.request":
                            # Unexpected message type: re-arm and keep
                            # going rather than busy-spinning on the done
                            # task.
                            recv_task = loop.create_task(_recv())
                            continue

                        body = message.get("body", b"")
                        if body:
                            buffer.extend(body)
                            total_bytes += len(body)
                            last_audio_at = loop.time()
                            if len(buffer) > max_bytes:
                                # Trim the oldest audio, keeping 2-byte sample alignment.
                                overflow = len(buffer) - max_bytes
                                overflow -= overflow % 2
                                del buffer[:overflow]
                        if not message.get("more_body", False):
                            body_complete = True
                            break
                        recv_task = loop.create_task(_recv())

                    now = loop.time()

                    # ── tick: collect finished / submit new ────────────
                    if now >= next_tick:
                        next_tick = now + params.chunk_seconds

                        if in_flight is not None and in_flight.done():
                            task, in_flight = in_flight, None
                            try:
                                text, lang = task.result()
                            except Exception:
                                self.error_log.exception(
                                    "REALTIME-TRANSCRIPTION inference failed request=%s", request_id
                                )
                                await _emit("error", {"message": "Transcription failed", "request_id": request_id})
                            else:
                                await _emit_transcript(text, lang, bytes_at_last_submit, snapshot_len)

                        # Only submit when new audio has arrived since the
                        # last snapshot: a stalled feed must not
                        # re-transcribe the same window every tick.
                        if in_flight is None and len(buffer) >= min_bytes and total_bytes > bytes_at_last_submit:
                            snapshot = bytes(buffer)
                            bytes_at_last_submit = total_bytes
                            snapshot_len = len(snapshot)
                            # run_in_executor returns a Future (awaitable,
                            # has .done()/.result()); do not wrap it in
                            # create_task.
                            in_flight = loop.run_in_executor(
                                _realtime_executor,
                                _transcribe_window,
                                self.engine,
                                snapshot,
                                params.sample_rate,
                                params.language,
                                params.prompt,
                            )
                            in_flight.add_done_callback(_swallow_exception)

                    # ── heartbeat ──────────────────────────────────────
                    if now >= next_heartbeat:
                        next_heartbeat = now + self.heartbeat_s
                        await send({"type": "http.response.body", "body": b": keep-alive\n\n", "more_body": True})

                # ── final flush (only while the client is connected) ───
                if not disconnected:
                    tail_pending = False
                    if in_flight is not None:
                        task, in_flight = in_flight, None
                        try:
                            text, lang = await asyncio.wait_for(task, timeout=self.final_flush_timeout_s)
                        except asyncio.TimeoutError:
                            task.cancel()
                            self.error_log.error("REALTIME-TRANSCRIPTION final flush timed out request=%s", request_id)
                            await _emit("error", {"message": "Final transcription timed out", "request_id": request_id})
                        except Exception:
                            self.error_log.exception("REALTIME-TRANSCRIPTION final flush failed request=%s", request_id)
                            await _emit("error", {"message": "Final transcription failed", "request_id": request_id})
                        else:
                            await _emit_transcript(text, lang, bytes_at_last_submit, snapshot_len)
                            tail_pending = total_bytes > bytes_at_last_submit
                    else:
                        # No in-flight snapshot: the buffer is the tail.
                        tail_pending = total_bytes > bytes_at_last_submit

                    # Transcribe any audio that arrived after the last
                    # snapshot, so a body that ends between ticks does not
                    # drop its tail.
                    if tail_pending and len(buffer) >= min_bytes:
                        tail_fut = loop.run_in_executor(
                            _realtime_executor,
                            _transcribe_window,
                            self.engine,
                            bytes(buffer),
                            params.sample_rate,
                            params.language,
                            params.prompt,
                        )
                        tail_fut.add_done_callback(_swallow_exception)
                        try:
                            text, lang = await asyncio.wait_for(tail_fut, timeout=self.final_flush_timeout_s)
                        except asyncio.TimeoutError:
                            self.error_log.error("REALTIME-TRANSCRIPTION final flush timed out request=%s", request_id)
                            await _emit("error", {"message": "Final transcription timed out", "request_id": request_id})
                        except Exception:
                            self.error_log.exception("REALTIME-TRANSCRIPTION final flush failed request=%s", request_id)
                            await _emit("error", {"message": "Final transcription failed", "request_id": request_id})
                        else:
                            await _emit_transcript(text, lang, total_bytes, len(buffer))

                total_s = total_bytes / 2 / params.sample_rate
                elapsed = perf_counter() - started
                if disconnected:
                    self.trans_log.warning(
                        "COMPLETE request=%s chunks=%d total_audio=%.1fs session=%.1fs (client disconnected)",
                        request_id,
                        chunk_index,
                        total_s,
                        elapsed,
                    )
                else:
                    self.trans_log.info(
                        "COMPLETE request=%s chunks=%d total_audio=%.1fs session=%.1fs reason=%s",
                        request_id,
                        chunk_index,
                        total_s,
                        elapsed,
                        reason,
                    )
                self.api_log.info(
                    "REALTIME-TRANSCRIPTION ok request=%s chunks=%d time=%.1fs",
                    request_id,
                    chunk_index,
                    elapsed,
                )
                await _emit(
                    "done",
                    {
                        "request_id": request_id,
                        "chunks": chunk_index,
                        "total_seconds": round(total_s, 3),
                        "reason": reason,
                    },
                )
                await send({"type": "http.response.body", "body": b"", "more_body": False})

            finally:
                if not recv_task.done():
                    recv_task.cancel()
                elif not recv_task.cancelled():
                    # A receive() that failed after the loop broke would
                    # otherwise log "Task exception was never retrieved".
                    recv_task.exception()
                if in_flight is not None and not in_flight.done():
                    in_flight.cancel()
        finally:
            with self._slot_lock:
                self._active_sessions -= 1


def create_realtime_transcription_router(
    engine: AudioAPIEngine,
    **endpoint_kwargs: Any,
) -> APIRouter:
    """Build the ``/realtime`` router bound to *engine*.

    *endpoint_kwargs* are forwarded to
    :class:`RealtimeTranscriptionEndpoint` (``heartbeat_s``,
    ``final_flush_timeout_s``, ``idle_timeout_s``, ``max_sessions``).
    """
    router = APIRouter(tags=["realtime"])
    router.routes.append(
        Route(
            "/realtime/transcription",
            endpoint=RealtimeTranscriptionEndpoint(engine, **endpoint_kwargs),
            methods=["POST"],
            include_in_schema=False,
        )
    )
    return router
