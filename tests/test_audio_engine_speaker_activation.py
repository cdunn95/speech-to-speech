"""Regression tests for AudioAPIEngine._activate_speaker voice-list handling."""

from __future__ import annotations

import pytest

from speech_to_speech.api.openai_audio.audio_engine import AudioAPIEngine


class _FakeTTSHandler:
    def __init__(self, voices: list[str]) -> None:
        self._voices = voices
        self.speaker: str | None = None

    def get_supported_speakers(self) -> list[str]:
        return list(self._voices)


def test_activate_speaker_skips_validation_with_empty_voice_list() -> None:
    handler = _FakeTTSHandler(voices=[])
    engine = AudioAPIEngine(tts_handler=handler)

    # An empty voice list means "unknown", not "no valid voices": the
    # activation must succeed unvalidated instead of hard-failing every
    # built-in voice request.
    engine._activate_speaker(handler, "luna")

    assert handler.speaker == "luna"


def test_activate_speaker_resolves_case_insensitively() -> None:
    handler = _FakeTTSHandler(voices=["Luna", "Bruno"])
    engine = AudioAPIEngine(tts_handler=handler)

    engine._activate_speaker(handler, "luna")

    assert handler.speaker == "Luna"


def test_activate_speaker_rejects_unknown_speaker() -> None:
    handler = _FakeTTSHandler(voices=["Luna", "Bruno"])
    engine = AudioAPIEngine(tts_handler=handler)

    with pytest.raises(ValueError, match="Unknown speaker"):
        engine._activate_speaker(handler, "alloy")
