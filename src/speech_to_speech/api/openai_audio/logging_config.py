"""File-based logging configuration for the OpenAI-compatible audio API.

All log output is written to ``~/api-logs/`` with one file per context:

- ``api.log``            – every HTTP request (method, path, status, duration)
- ``transcription.log``  – /v1/audio/transcriptions details
- ``translation.log``    – /v1/audio/translations details
- ``speech.log``         – /v1/audio/speech details
- ``stt.log``            – STT model inference details
- ``tts.log``            – TTS model inference details
- ``errors.log``         – all errors and exceptions
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOG_DIR = Path(os.path.expanduser("~/api-logs"))

_LOG_FILES: dict[str, str] = {
    "api": "api.log",
    "transcription": "transcription.log",
    "translation": "translation.log",
    "speech": "speech.log",
    "stt": "stt.log",
    "tts": "tts.log",
    "errors": "errors.log",
}

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_configured = False
_loggers: dict[str, logging.Logger] = {}


def setup_audio_api_logging(log_level: str = "info") -> dict[str, logging.Logger]:
    """Create ``~/api-logs/`` and attach a ``FileHandler`` to each context logger.

    Safe to call more than once; subsequent calls are no-ops.
    Returns a mapping of context name → logger.
    """
    global _configured, _loggers
    if _configured:
        return _loggers

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    for name, filename in _LOG_FILES.items():
        logger = logging.getLogger(f"speech_to_speech.api.audio.{name}")
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.FileHandler(_LOG_DIR / filename, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_FORMAT))
            handler.setLevel(level)
            logger.addHandler(handler)

        # Let records also propagate to the root logger (console output).
        logger.propagate = True
        _loggers[name] = logger

    _configured = True
    _loggers["api"].info("Audio API logging initialised → %s", _LOG_DIR)
    return _loggers


def get_loggers() -> dict[str, logging.Logger]:
    """Return the configured loggers, setting them up with defaults if needed."""
    if not _configured:
        return setup_audio_api_logging()
    return _loggers
