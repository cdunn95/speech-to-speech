"""FastAPI router exposing OpenAI-compatible audio endpoints.

Endpoints
---------
- ``POST /v1/audio/transcriptions``  – transcribe audio → text
- ``POST /v1/audio/translations``    – translate non-English audio → English text
- ``POST /v1/audio/speech``          – synthesise text → audio

The router is mounted on the same FastAPI application as the Realtime
endpoints so a single port serves both API families.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import subprocess
import uuid
import wave
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from speech_to_speech.api.openai_audio.audio_engine import AudioAPIEngine
from speech_to_speech.api.openai_audio.logging_config import get_loggers

# ── constants ─────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB (OpenAI limit)
MAX_TEXT_LENGTH = 4096

SUPPORTED_TRANSCRIPTION_FORMATS = frozenset({"json", "text", "srt", "verbose_json", "vtt"})
SUPPORTED_SPEECH_FORMATS = frozenset({"mp3", "wav", "pcm", "flac"})

_CONTENT_TYPE_BY_FORMAT: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
}


def _openai_error(status: int, message: str, err_type: str, param: str | None = None, code: str | None = None) -> JSONResponse:
    """Build an OpenAI-shaped error body."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )


# ── audio codec helpers ───────────────────────────────────────────────


def _decode_audio_file(data: bytes, sample_rate: int = 16_000) -> np.ndarray:
    """Decode an uploaded audio file to a float32 mono numpy array at *sample_rate*."""
    import librosa
    import soundfile as sf

    buf = io.BytesIO(data)

    # Fast path: soundfile handles wav / flac / ogg natively.
    try:
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        buf.seek(0)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio
    except Exception:
        buf.seek(0)

    # Slow path: librosa + audioread (mp3, m4a, webm, … via ffmpeg).
    audio, _ = librosa.load(buf, sr=sample_rate, mono=True)
    return audio


def _encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode int16 mono *audio* into the requested container format."""
    audio = audio.astype(np.int16, copy=False)

    if fmt == "pcm":
        return audio.tobytes()

    if fmt == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    if fmt == "flac":
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="FLAC", subtype="PCM_16")
        return buf.getvalue()

    if fmt == "mp3":
        return _encode_mp3(audio, sample_rate)

    raise HTTPException(status_code=501, detail=f"Unsupported output format: {fmt}")


def _encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode to MP3 via *lameenc* (preferred) or *ffmpeg* (fallback)."""
    pcm = audio.tobytes()

    try:
        import lameenc

        enc = lameenc.Encoder()
        enc.set_bit_rate(128)
        enc.set_in_sample_rate(sample_rate)
        enc.set_channels(1)
        enc.set_quality(2)
        return enc.encode(pcm) + enc.flush()
    except ImportError:
        pass

    # Fallback: shell out to ffmpeg.
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
                "-i", "pipe:0",
                "-c:a", "libmp3lame", "-b:a", "128k",
                "pipe:1",
            ],
            input=pcm,
            capture_output=True,
            check=True,
            timeout=60,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(
            status_code=501,
            detail="MP3 encoding unavailable: install 'lameenc' (pip install lameenc) or ensure ffmpeg is on PATH",
        ) from exc


# ── subtitle / format helpers ─────────────────────────────────────────


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_srt(text: str, duration: float) -> str:
    return f"1\n{_srt_timestamp(0.0)} --> {_srt_timestamp(duration)}\n{text}\n"


def _format_vtt(text: str, duration: float) -> str:
    return f"WEBVTT\n\n00:00.000 --> {_vtt_timestamp(duration)}\n{text}\n"


# ── thread-pool helper ────────────────────────────────────────────────


async def _run_sync(func, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable in the default thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)


# ── router factory ────────────────────────────────────────────────────


def create_audio_api_router(engine: AudioAPIEngine) -> APIRouter:
    """Build the ``/v1/audio`` router bound to *engine*."""
    loggers = get_loggers()
    api_log = loggers["api"]
    trans_log = loggers["transcription"]
    translat_log = loggers["translation"]
    speech_log = loggers["speech"]
    error_log = loggers["errors"]

    router = APIRouter(prefix="/v1/audio", tags=["audio"])

    # ── POST /v1/audio/transcriptions ────────────────────────────────

    @router.post("/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        stream: bool = Form(False),
        timestamp_granularities: list[str] = Form(default=[]),
        keywords: list[str] = Form(default=[]),
        languages: list[str] = Form(default=[]),
        chunking_strategy: str | None = Form(None),
        known_speaker_names: list[str] = Form(default=[]),
    ) -> Response:
        request_id = str(uuid.uuid4())
        started = perf_counter()

        # Resolve language: prefer explicit `language`, fall back to first `languages[]`.
        effective_language = language
        if effective_language is None and languages:
            effective_language = languages[0]

        api_log.info(
            "TRANSCRIPTION request=%s model=%s language=%s format=%s file=%s",
            request_id, model, effective_language, response_format, file.filename,
        )
        trans_log.info(
            "START request=%s file=%s model=%s language=%s format=%s stream=%s",
            request_id, file.filename, model, effective_language, response_format, stream,
        )

        try:
            if stream:
                trans_log.warning("Streaming transcription not supported; returning non-streamed result")

            if response_format not in SUPPORTED_TRANSCRIPTION_FORMATS:
                trans_log.error("Unsupported response_format=%s", response_format)
                return _openai_error(
                    400,
                    f"Unsupported response_format: {response_format!r}. "
                    f"Supported: {sorted(SUPPORTED_TRANSCRIPTION_FORMATS)}",
                    "invalid_request_error",
                    param="response_format",
                )

            data = await file.read()
            file_size = len(data)
            trans_log.info("RECEIVED request=%s file=%s size=%d bytes", request_id, file.filename, file_size)

            if file_size == 0:
                return _openai_error(400, "Uploaded file is empty", "invalid_request_error", param="file", code="empty_file")
            if file_size > MAX_FILE_SIZE_BYTES:
                return _openai_error(
                    400,
                    f"File too large: {file_size} bytes (max {MAX_FILE_SIZE_BYTES})",
                    "invalid_request_error",
                    param="file",
                    code="file_too_large",
                )

            if not engine.has_stt:
                return _openai_error(
                    501,
                    "STT is not available (the stt backend is set to 'none')",
                    "server_error",
                    code="stt_not_available",
                )

            # Decode
            decode_start = perf_counter()
            audio = await _run_sync(_decode_audio_file, data, engine.sample_rate)
            decode_s = perf_counter() - decode_start
            duration_s = len(audio) / engine.sample_rate
            trans_log.info(
                "DECODED request=%s samples=%d duration=%.2fs decode_time=%.3fs",
                request_id, len(audio), duration_s, decode_s,
            )

            # Infer
            result = await _run_sync(engine.transcribe, audio, effective_language, prompt)
            total_s = perf_counter() - started

            trans_log.info(
                "COMPLETE request=%s text_len=%d language=%s total_time=%.3fs",
                request_id, len(result.text), result.language, total_s,
            )
            api_log.info("TRANSCRIPTION ok request=%s time=%.3fs", request_id, total_s)

            return _transcription_response(result.text, result.language, duration_s, response_format, temperature)

        except HTTPException:
            raise
        except Exception as exc:
            total_s = perf_counter() - started
            error_log.exception("TRANSCRIPTION failed request=%s time=%.3fs", request_id, total_s)
            api_log.error("TRANSCRIPTION failed request=%s error=%s time=%.3fs", request_id, exc, total_s)
            return _openai_error(500, f"Transcription failed: {exc}", "server_error")

    # ── POST /v1/audio/translations ──────────────────────────────────

    @router.post("/translations")
    async def create_translation(
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
    ) -> Response:
        request_id = str(uuid.uuid4())
        started = perf_counter()

        api_log.info(
            "TRANSLATION request=%s model=%s format=%s file=%s",
            request_id, model, response_format, file.filename,
        )
        translat_log.info(
            "START request=%s file=%s model=%s format=%s",
            request_id, file.filename, model, response_format,
        )

        try:
            if response_format not in SUPPORTED_TRANSCRIPTION_FORMATS - {"diarized_json"}:
                return _openai_error(
                    400,
                    f"Unsupported response_format: {response_format!r}",
                    "invalid_request_error",
                    param="response_format",
                )

            data = await file.read()
            file_size = len(data)
            translat_log.info("RECEIVED request=%s file=%s size=%d bytes", request_id, file.filename, file_size)

            if file_size == 0:
                return _openai_error(400, "Uploaded file is empty", "invalid_request_error", param="file", code="empty_file")
            if file_size > MAX_FILE_SIZE_BYTES:
                return _openai_error(
                    400,
                    f"File too large: {file_size} bytes (max {MAX_FILE_SIZE_BYTES})",
                    "invalid_request_error",
                    param="file",
                    code="file_too_large",
                )

            if not engine.has_stt:
                return _openai_error(
                    501,
                    "STT is not available (the stt backend is set to 'none')",
                    "server_error",
                    code="stt_not_available",
                )

            decode_start = perf_counter()
            audio = await _run_sync(_decode_audio_file, data, engine.sample_rate)
            decode_s = perf_counter() - decode_start
            duration_s = len(audio) / engine.sample_rate
            translat_log.info(
                "DECODED request=%s samples=%d duration=%.2fs decode_time=%.3fs",
                request_id, len(audio), duration_s, decode_s,
            )

            # Translation = transcription forced to English.
            result = await _run_sync(engine.transcribe, audio, "en", prompt)
            total_s = perf_counter() - started

            translat_log.info(
                "COMPLETE request=%s text_len=%d total_time=%.3fs",
                request_id, len(result.text), total_s,
            )
            api_log.info("TRANSLATION ok request=%s time=%.3fs", request_id, total_s)

            return _transcription_response(result.text, "en", duration_s, response_format, 0.0)

        except HTTPException:
            raise
        except Exception as exc:
            total_s = perf_counter() - started
            error_log.exception("TRANSLATION failed request=%s time=%.3fs", request_id, total_s)
            api_log.error("TRANSLATION failed request=%s error=%s time=%.3fs", request_id, exc, total_s)
            return _openai_error(500, f"Translation failed: {exc}", "server_error")

    # ── POST /v1/audio/speech ────────────────────────────────────────

    @router.post("/speech")
    async def create_speech(request_body: dict[str, Any]) -> Response:
        request_id = str(uuid.uuid4())
        started = perf_counter()

        model = request_body.get("model", "")
        text = request_body.get("input", "")
        voice = request_body.get("voice", "alloy")
        response_format = request_body.get("response_format", "mp3")
        speed = request_body.get("speed", 1.0)
        instructions = request_body.get("instructions")
        stream_format = request_body.get("stream_format")

        # `voice` may be a string or a custom-voice object {"id": "..."}.
        if isinstance(voice, dict):
            voice = voice.get("id", "alloy")

        api_log.info(
            "SPEECH request=%s model=%s voice=%s format=%s text_len=%d speed=%s",
            request_id, model, voice, response_format, len(text), speed,
        )
        speech_log.info(
            "START request=%s model=%s voice=%s format=%s text_len=%d speed=%s instructions=%s",
            request_id, model, voice, response_format, len(text), speed,
            "(set)" if instructions else None,
        )

        try:
            if not text or not text.strip():
                return _openai_error(400, "Input text is empty", "invalid_request_error", param="input")
            if len(text) > MAX_TEXT_LENGTH:
                return _openai_error(
                    400,
                    f"Input text too long: {len(text)} chars (max {MAX_TEXT_LENGTH})",
                    "invalid_request_error",
                    param="input",
                    code="text_too_long",
                )
            if response_format not in SUPPORTED_SPEECH_FORMATS:
                return _openai_error(
                    400,
                    f"Unsupported response_format: {response_format!r}. "
                    f"Supported: {sorted(SUPPORTED_SPEECH_FORMATS)}",
                    "invalid_request_error",
                    param="response_format",
                )

            if not engine.has_tts:
                return _openai_error(
                    501,
                    "TTS is not available",
                    "server_error",
                    code="tts_not_available",
                )

            if speed != 1.0:
                speech_log.warning("Speed adjustment (%s) is not supported by the local TTS backend; using 1.0", speed)

            audio, sr = await _run_sync(engine.synthesize, text.strip(), voice, None)
            total_s = perf_counter() - started

            if audio.size == 0:
                return _openai_error(500, "TTS engine produced no audio", "server_error")

            encoded = await _run_sync(_encode_audio, audio, sr, response_format)
            total_s = perf_counter() - started

            speech_log.info(
                "COMPLETE request=%s audio_bytes=%d format=%s total_time=%.3fs",
                request_id, len(encoded), response_format, total_s,
            )
            api_log.info("SPEECH ok request=%s bytes=%d time=%.3fs", request_id, len(encoded), total_s)

            return Response(
                content=encoded,
                media_type=_CONTENT_TYPE_BY_FORMAT.get(response_format, "application/octet-stream"),
                headers={"X-Request-Id": request_id},
            )

        except HTTPException:
            raise
        except Exception as exc:
            total_s = perf_counter() - started
            error_log.exception("SPEECH failed request=%s time=%.3fs", request_id, total_s)
            api_log.error("SPEECH failed request=%s error=%s time=%.3fs", request_id, exc, total_s)
            return _openai_error(500, f"Speech synthesis failed: {exc}", "server_error")

    return router


# ── response builder ──────────────────────────────────────────────────


def _transcription_response(
    text: str,
    language: str | None,
    duration: float,
    response_format: str,
    temperature: float,
) -> Response:
    """Build the HTTP response for a transcription / translation result."""
    if response_format == "json":
        return JSONResponse({"text": text})

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format == "srt":
        return PlainTextResponse(_format_srt(text, duration), media_type="application/x-subrip")

    if response_format == "vtt":
        return PlainTextResponse(_format_vtt(text, duration), media_type="text/vtt")

    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": language or "unknown",
                "duration": round(duration, 3),
                "text": text,
                "segments": [
                    {
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": round(duration, 3),
                        "text": text,
                        "temperature": temperature,
                    }
                ],
            }
        )

    # Fallback (should not reach here due to earlier validation).
    return JSONResponse({"text": text})
