"""Voice-clone management for Qwen3-TTS.

This package owns the on-disk store of cloned voices (reference audio plus its
transcription) and the small amount of state needed to route synthesis to a
"default" clone. It is backend-agnostic: the store only persists data, and the
routing is applied to TTS handlers that opt in by exposing the
``set_clone_reference`` / ``clear_clone_reference`` methods (Qwen3-TTS does).
"""

from speech_to_speech.clones.store import (
    Clone,
    CloneError,
    CloneExistsError,
    CloneNotFoundError,
    CloneStore,
)

__all__ = [
    "Clone",
    "CloneError",
    "CloneExistsError",
    "CloneNotFoundError",
    "CloneStore",
]
