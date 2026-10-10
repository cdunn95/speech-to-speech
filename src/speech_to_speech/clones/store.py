"""Persistent store for Qwen3-TTS voice clones.

A clone is a named reference voice: an mp3 upload normalized to a WAV file, its
transcription, and an optional free-form description. Clones are identified by a
simple incrementing integer ID (never reused) and can also be referenced by
name. The store persists a small JSON metadata file plus one WAV per clone under
a storage directory (``S2S_CLONE_DIR``, default ``~/.speech_to_speech/clones``).

All public methods are safe to call from multiple threads; a single lock guards
the read-modify-write cycle on the metadata file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from speech_to_speech.clones.audio import CLONE_AUDIO_SAMPLE_RATE, convert_to_wav

logger = logging.getLogger(__name__)

DEFAULT_CLONE_DIR = os.path.join("~", ".speech_to_speech", "clones")
CLONE_DIR_ENV = "S2S_CLONE_DIR"
_METADATA_FILE = "clones.json"

# A clone can be addressed by its integer ID or by its (unique) name.
CloneIdentifier = Union[int, str]


class CloneError(Exception):
    """Base error for clone operations."""


class CloneNotFoundError(CloneError):
    """Raised when a clone cannot be found by name or ID."""


class CloneExistsError(CloneError):
    """Raised when a clone with the same name already exists."""


@dataclass
class Clone:
    """An in-memory view of a stored voice clone."""

    id: int
    name: str
    description: str | None
    transcription: str
    audio_path: Path
    created_at: str

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for the ``/clone/list`` endpoint.

        The description is included only when present, per the API contract.
        """
        result: dict[str, Any] = {"id": self.id, "name": self.name}
        if self.description is not None:
            result["description"] = self.description
        return result

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "transcription": self.transcription,
            "audio_file": self.audio_path.name,
            "created_at": self.created_at,
        }


class CloneStore:
    """Manages voice clones on disk with a simple incrementing integer ID scheme."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        resolved = storage_dir or os.environ.get(CLONE_DIR_ENV) or DEFAULT_CLONE_DIR
        self.storage_dir = Path(resolved).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.storage_dir / _METADATA_FILE
        self._lock = threading.Lock()
        self._state = self._load_state()

    # -- persistence -------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if self._meta_path.exists():
            try:
                with open(self._meta_path, encoding="utf-8") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read clone metadata %s: %s; starting fresh", self._meta_path, e)
                state = {}
            if not isinstance(state, dict):
                state = {}
            state.setdefault("next_id", 1)
            state.setdefault("default_id", None)
            state.setdefault("clones", [])
            # Reconcile next_id against existing IDs to prevent WAV clobbering
            # if the metadata was rolled back or hand-edited.
            existing_ids = [m["id"] for m in state["clones"] if isinstance(m, dict) and "id" in m]
            if existing_ids:
                state["next_id"] = max(state["next_id"], max(existing_ids) + 1)
            return state
        return {"next_id": 1, "default_id": None, "clones": []}

    def _save_state(self) -> None:
        tmp = self._meta_path.with_name(self._meta_path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
            os.replace(tmp, self._meta_path)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise CloneError(f"Failed to save clone metadata: {e}") from e

    # -- helpers -----------------------------------------------------------

    def _audio_path_for(self, clone_id: int) -> Path:
        return self.storage_dir / f"{clone_id}.wav"

    def _meta_to_clone(self, meta: dict[str, Any]) -> Clone:
        return Clone(
            id=meta["id"],
            name=meta["name"],
            description=meta.get("description"),
            transcription=meta.get("transcription", ""),
            audio_path=self._audio_path_for(meta["id"]),
            created_at=meta.get("created_at", ""),
        )

    def _find(self, identifier: CloneIdentifier) -> dict[str, Any] | None:
        """Resolve an ID or name to a metadata record (caller must hold the lock)."""
        if isinstance(identifier, bool):
            return None
        if isinstance(identifier, int):
            for meta in self._state["clones"]:
                if meta["id"] == identifier:
                    return meta
            return None
        value = str(identifier).strip()
        if not value:
            return None
        # A purely numeric name is ambiguous: prefer the ID, then fall back to a name.
        if value.isdigit():
            for meta in self._state["clones"]:
                if meta["id"] == int(value):
                    return meta
        for meta in self._state["clones"]:
            if str(meta["name"]).lower() == value.lower():
                return meta
        return None

    # -- public API --------------------------------------------------------

    def create(
        self,
        name: str,
        description: str | None,
        transcription: str,
        audio_bytes: bytes,
    ) -> Clone:
        """Normalize *audio_bytes* and register a new clone. Returns the created clone."""
        name = (name or "").strip()
        transcription = (transcription or "").strip()
        description = (description or "").strip() or None
        if not name:
            raise CloneError("A non-empty name is required.")
        if not transcription:
            raise CloneError("A non-empty transcription is required.")

        with self._lock:
            for meta in self._state["clones"]:
                if str(meta["name"]).lower() == name.lower():
                    raise CloneExistsError(f"A clone named '{name}' already exists.")

            clone_id = int(self._state["next_id"])
            audio_path = self._audio_path_for(clone_id)
            try:
                convert_to_wav(audio_bytes, audio_path, CLONE_AUDIO_SAMPLE_RATE)
            except Exception as e:  # noqa: BLE001 - normalize any decode failure
                audio_path.unlink(missing_ok=True)
                raise CloneError(f"Failed to process the reference audio: {e}") from e

            clone = Clone(
                id=clone_id,
                name=name,
                description=description,
                transcription=transcription,
                audio_path=audio_path,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._state["clones"].append(clone.to_meta_dict())
            self._state["next_id"] = clone_id + 1
            self._save_state()
            return clone

    def list(self) -> list[Clone]:
        with self._lock:
            return [self._meta_to_clone(meta) for meta in self._state["clones"]]

    def get(self, identifier: CloneIdentifier) -> Clone | None:
        with self._lock:
            meta = self._find(identifier)
            return self._meta_to_clone(meta) if meta is not None else None

    def get_default(self) -> Clone | None:
        with self._lock:
            default_id = self._state["default_id"]
            if default_id is None:
                return None
            for meta in self._state["clones"]:
                if meta["id"] == default_id:
                    return self._meta_to_clone(meta)
            return None

    def set_default(self, identifier: CloneIdentifier) -> Clone:
        """Mark the clone addressed by *identifier* as the default voice."""
        with self._lock:
            meta = self._find(identifier)
            if meta is None:
                raise CloneNotFoundError(f"Clone not found: {identifier!r}")
            self._state["default_id"] = meta["id"]
            self._save_state()
            return self._meta_to_clone(meta)

    def delete(self, identifier: CloneIdentifier) -> tuple[Clone, bool]:
        """Delete the clone addressed by *identifier*.

        Returns a tuple of (deleted_clone, was_default) so the caller can
        decide whether to clear routing without a racy pre-read.
        """
        with self._lock:
            meta = self._find(identifier)
            if meta is None:
                raise CloneNotFoundError(f"Clone not found: {identifier!r}")
            clone_id = meta["id"]
            was_default = self._state["default_id"] == clone_id
            self._state["clones"] = [m for m in self._state["clones"] if m["id"] != clone_id]
            if was_default:
                self._state["default_id"] = None
            self._save_state()
            self._audio_path_for(clone_id).unlink(missing_ok=True)
            return self._meta_to_clone(meta), was_default
