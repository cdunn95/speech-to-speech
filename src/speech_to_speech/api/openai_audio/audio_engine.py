"""Standalone STT / TTS inference engine for the OpenAI-compatible audio API.

Wraps the pipeline handler instances so the HTTP endpoints can call
``process()`` directly (bypassing the queue-based pipeline) while sharing
the same loaded models.  A per-engine lock serialises concurrent API
requests against the (typically GPU-resident) model.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    AudioOutput,
    Transcription,
    TTSInput,
    VADAudio,
)

stt_log = logging.getLogger("speech_to_speech.api.audio.stt")
tts_log = logging.getLogger("speech_to_speech.api.audio.tts")


@dataclass
class TranscriptionResult:
    """Normalised output of one STT inference call."""

    text: str
    language: str | None = None
    duration_s: float | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)


class AudioAPIEngine:
    """Direct (non-pipeline) inference over already-loaded STT / TTS handlers.

    Parameters
    ----------
    stt_handler:
        A ``BaseSTTHandler`` instance (e.g. ``WhisperSTTHandler``).
        ``None`` when the STT backend is ``"none"``.
    tts_handler:
        The TTS ``BaseHandler`` instance (last handler in the pipeline chain).
        ``None`` when no TTS backend is configured.
    sample_rate:
        Pipeline sample rate (default 16 000 Hz).
    """

    def __init__(
        self,
        stt_handler: Any | None = None,
        tts_handler: Any | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        self._stt_handler = stt_handler
        self._tts_handler = tts_handler
        self._sample_rate = sample_rate
        self._stt_lock = threading.Lock()
        self._tts_lock = threading.Lock()

    # ── capability queries ────────────────────────────────────────────

    @property
    def has_stt(self) -> bool:
        """True when a real STT inference handler is available.

        Returns False for ``None`` and for the ``AudioInputNotifier``
        passthrough used when ``--stt none`` is configured.
        """
        if self._stt_handler is None:
            return False
        # AudioInputNotifier is the passthrough for --stt none; it doesn't
        # perform inference, it just forwards audio to the LLM queue.
        from speech_to_speech.LLM.audio_input_notifier import AudioInputNotifier

        return not isinstance(self._stt_handler, AudioInputNotifier)

    @property
    def has_tts(self) -> bool:
        return self._tts_handler is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # ── STT ───────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Run one STT inference on *audio* (float32, mono, 16 kHz)."""
        stt_handler = self._stt_handler
        if stt_handler is None:
            raise RuntimeError("No STT handler available (stt backend is 'none')")

        started = perf_counter()
        stt_log.info(
            "STT inference start: samples=%d (%.2fs) language=%s prompt=%s",
            len(audio),
            len(audio) / self._sample_rate,
            language,
            "(set)" if prompt else None,
        )

        with self._stt_lock:
            vad_audio = VADAudio(audio=audio, mode="final")
            results = list(stt_handler.process(vad_audio))

        elapsed = perf_counter() - started

        for item in results:
            if isinstance(item, Transcription):
                stt_log.info(
                    "STT inference complete: text_len=%d language=%s time=%.3fs",
                    len(item.text),
                    item.language_code,
                    elapsed,
                )
                return TranscriptionResult(
                    text=item.text,
                    language=item.language_code,
                    duration_s=elapsed,
                )

        stt_log.error("STT handler produced no Transcription result (got %s)", results)
        raise RuntimeError("STT handler did not produce a transcription")

    # ── TTS ───────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Run one TTS inference.  Returns ``(int16_audio, sample_rate)``."""
        tts_handler = self._tts_handler
        if tts_handler is None:
            raise RuntimeError("No TTS handler available")

        started = perf_counter()
        tts_log.info(
            "TTS inference start: text_len=%d voice=%s language=%s",
            len(text),
            voice,
            language,
        )

        with self._tts_lock:
            tts_input = TTSInput(text=text, language_code=language)
            chunks: list[np.ndarray] = []
            for chunk in tts_handler.process(tts_input):
                arr = _extract_audio(chunk)
                if arr is not None and arr.size:
                    chunks.append(arr)

        elapsed = perf_counter() - started
        audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)
        tts_log.info(
            "TTS inference complete: samples=%d (%.2fs) time=%.3fs",
            len(audio),
            len(audio) / self._sample_rate,
            elapsed,
        )
        return audio, self._sample_rate


def _extract_audio(chunk: Any) -> np.ndarray | None:
    """Pull a raw int16 ``np.ndarray`` out of a TTS handler yield."""
    if isinstance(chunk, np.ndarray):
        return chunk.astype(np.int16, copy=False)
    if isinstance(chunk, AudioOutput):
        payload = chunk.audio
        if isinstance(payload, np.ndarray):
            return payload.astype(np.int16, copy=False)
        if isinstance(payload, bytes):
            return np.frombuffer(payload, dtype=np.int16)
    if isinstance(chunk, bytes):
        if chunk == AUDIO_RESPONSE_DONE:
            return None
        return np.frombuffer(chunk, dtype=np.int16)
    return None
