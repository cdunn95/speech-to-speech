"""Audio normalization for voice-clone reference audio.

Uploaded reference audio (an mp3) is decoded and re-encoded to a 16-bit PCM
mono WAV at the Qwen3-TTS speech-tokenizer sample rate. Both the GGML
(``faster-qwen3-tts``) and MLX (``mlx-audio``) backends read plain WAV files
reliably, so normalizing once at upload time keeps the clone store simple and
the synthesis path free of any mp3 decoding.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Qwen3-TTS operates on 24 kHz audio; store reference clips at that rate.
CLONE_AUDIO_SAMPLE_RATE = 24000
# Qwen3-TTS ICL mode prepends reference codec tokens to every generation.
# A 20 s reference is ~480 extra prefill tokens per utterance; beyond that
# the TTFA cost outweighs any cloning-quality benefit. Two seconds is the
# minimum for a meaningful ICL clone.
MIN_CLONE_AUDIO_SECONDS = 2.0
MAX_CLONE_AUDIO_SECONDS = 20.0


class CloneAudioError(Exception):
    """Raised when reference audio cannot be decoded or normalized."""


def convert_to_wav(
    input_bytes: bytes,
    output_path: Path,
    target_sr: int = CLONE_AUDIO_SAMPLE_RATE,
) -> tuple[float, bool]:
    """Decode *input_bytes* and write a 16-bit PCM mono WAV to *output_path*.

    Returns a tuple of (duration_in_seconds, was_truncated). Raises
    :class:`CloneAudioError` when the audio is empty, undecodable, or shorter
    than the minimum.
    """
    if not input_bytes:
        raise CloneAudioError("Audio file is empty.")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(input_bytes)
            tmp_path = tmp.name

        try:
            import librosa

            # Cap decoding at MAX+1 s so a pathological file cannot allocate
            # unbounded memory before the length check below truncates it.
            audio, _ = librosa.load(
                tmp_path,
                sr=target_sr,
                mono=True,
                duration=MAX_CLONE_AUDIO_SECONDS + 1.0,
            )
        except Exception as e:  # noqa: BLE001 - surface a uniform clone error
            logger.warning("Failed to decode uploaded audio: %s", e, exc_info=True)
            raise CloneAudioError(
                "The uploaded audio could not be decoded. Please upload a valid MP3 file (1–60 s)."
            ) from e

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise CloneAudioError("The uploaded audio contains no samples.")

        duration = len(audio) / float(target_sr)
        if duration < MIN_CLONE_AUDIO_SECONDS:
            raise CloneAudioError(
                f"Reference audio is too short ({duration:.2f}s); "
                f"at least {MIN_CLONE_AUDIO_SECONDS:.0f}s is required for a good clone."
            )

        truncated = False
        if duration > MAX_CLONE_AUDIO_SECONDS:
            # Keep only the leading window; longer clips add no cloning benefit
            # and increase per-utterance prefill latency.
            audio = audio[: int(MAX_CLONE_AUDIO_SECONDS * target_sr)]
            duration = len(audio) / float(target_sr)
            truncated = True
            logger.info("Reference audio truncated from >%.0fs to %.0fs", MAX_CLONE_AUDIO_SECONDS + 1, duration)

        import soundfile as sf

        sf.write(str(output_path), audio, target_sr, subtype="PCM_16")
        return duration, truncated
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
