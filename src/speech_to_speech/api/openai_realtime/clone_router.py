"""REST endpoints for managing Qwen3-TTS voice clones.

Endpoints
---------
- ``POST /clone/new``       upload an mp3 + name (+ optional description) + transcription
- ``POST /clone/default``   designate the default clone voice to route synthesis to
- ``POST /clone/delete/``   delete a clone by name or ID
- ``GET  /clone/list``      list all available clones (id, name, description if present)

The router is backend-agnostic: it persists clones through :class:`CloneStore`
and, when a default is set (or cleared), applies it to any TTS handler in the
pipeline pool that exposes ``set_clone_reference`` / ``clear_clone_reference``
(Qwen3-TTS does). Non-Qwen3 TTS backends are simply left untouched.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Union

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from speech_to_speech.clones.store import Clone, CloneError, CloneNotFoundError, CloneStore

logger = logging.getLogger(__name__)

# Cap the upload to 25 MB to prevent memory exhaustion from unauthenticated clients.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class CloneRef(BaseModel):
    """Body for endpoints that address a clone by name or integer ID."""

    clone: Union[int, str]


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(content={"status": "error", "error": message}, status_code=status_code)


def _is_mp3(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    return filename.endswith(".mp3") or content_type in {"audio/mpeg", "audio/mp3"}


def create_clone_router(clone_store: CloneStore, pool: list[Any]) -> APIRouter:
    """Build the ``/clone`` router bound to *clone_store* and the pipeline *pool*."""
    router = APIRouter(tags=["clones"])

    def _tts_handlers() -> list[Any]:
        handlers: list[Any] = []
        for unit in pool:
            tts = unit.handlers[-1] if unit.handlers else None
            if tts is not None:
                handlers.append(tts)
        return handlers

    def _apply_default(clone: Clone | None) -> tuple[int, list[str]]:
        """Route synthesis to *clone* (or clear it) across every TTS handler.

        Returns the number of handlers the change was applied to plus the
        per-handler failure reasons, so callers can report *why* nothing was
        applied (e.g. a backend that rejects raw-audio cloning).
        """
        applied = 0
        failures: list[str] = []
        for tts in _tts_handlers():
            try:
                if clone is not None and hasattr(tts, "set_clone_reference"):
                    # Validate the WAV exists before routing to it.
                    if not clone.audio_path.is_file():
                        logger.warning("Clone %d audio file missing: %s; skipping", clone.id, clone.audio_path)
                        continue
                    tts.set_clone_reference(clone.audio_path, clone.transcription)
                    applied += 1
                elif clone is None and hasattr(tts, "clear_clone_reference"):
                    tts.clear_clone_reference()
                    applied += 1
            except Exception as e:  # noqa: BLE001 - a bad handler must not break the API
                failures.append(str(e))
                logger.warning("Failed to apply clone to a TTS handler: %s", e)
        return applied, failures

    @router.post("/clone/new")
    async def clone_new(
        file: UploadFile | None = File(None),
        name: str | None = Form(None),
        description: str | None = Form(None),
        transcription: str | None = Form(None),
    ) -> Any:
        # All fields are optional at the framework level so that every failure is
        # reported in the same {"status": "error", "error": ...} shape rather than
        # a framework 422 validation body.
        name = (name or "").strip()
        transcription = (transcription or "").strip()
        description = (description or "").strip() or None
        if not name:
            return _error("A non-empty name is required.")
        if not transcription:
            return _error("A non-empty transcription is required.")
        if file is None or not file.filename:
            return _error("An mp3 file is required.")
        if not _is_mp3(file):
            return _error("The uploaded file must be an mp3.")

        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if not audio_bytes:
            return _error("The uploaded file is empty.")
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            return _error(f"Uploaded file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

        try:
            clone = await asyncio.to_thread(clone_store.create, name, description, transcription, audio_bytes)
        except CloneError as e:
            return _error(str(e))

        logger.info("Created voice clone id=%d name=%r", clone.id, clone.name)
        return {"status": "success", "id": clone.id, "name": clone.name}

    @router.post("/clone/default")
    async def clone_default(body: CloneRef) -> Any:
        try:
            clone = await asyncio.to_thread(clone_store.set_default, body.clone)
        except CloneError as e:
            status_code = 404 if isinstance(e, CloneNotFoundError) else 400
            return _error(str(e), status_code)

        applied, failures = _apply_default(clone)
        logger.info(
            "Set default voice clone id=%d name=%r (applied to %d TTS handler(s))",
            clone.id,
            clone.name,
            applied,
        )
        result: dict[str, Any] = {"status": "success", "id": clone.id, "name": clone.name}
        if applied == 0:
            if failures:
                # A handler exists but rejected the clone (e.g. the kitten C++
                # backend cannot do raw-audio cloning) — say so explicitly.
                result["warning"] = (
                    f"The default clone was persisted but could not be applied to any TTS handler: {failures[0]}"
                )
            else:
                result["warning"] = (
                    "No TTS handlers support voice cloning. The default is persisted "
                    "but synthesis will use the factory voice."
                )
        return result

    @router.post("/clone/delete/")
    async def clone_delete(body: CloneRef) -> Any:
        # Clear TTS handler references BEFORE deleting the file so that no
        # in-flight generation can read a deleted WAV. We peek at the default
        # first; if it matches the clone being deleted, we clear routing before
        # the store delete unlinks the WAV. The store's `was_default` return is
        # the authoritative signal — if our pre-read was stale (concurrent
        # set_default), we clear after the delete as a safety net.
        target = await asyncio.to_thread(clone_store.get, body.clone)
        if target is None:
            return _error(f"Clone not found: {body.clone!r}", 404)

        current_default = await asyncio.to_thread(clone_store.get_default)
        is_default_pre = current_default is not None and current_default.id == target.id

        if is_default_pre:
            _apply_default(None)

        try:
            clone, was_default = await asyncio.to_thread(clone_store.delete, body.clone)
        except CloneError as e:
            status_code = 404 if isinstance(e, CloneNotFoundError) else 400
            return _error(str(e), status_code)

        # Safety net: if the pre-read missed it (concurrent set_default between
        # our get_default and delete), clear now.
        if was_default and not is_default_pre:
            _apply_default(None)

        logger.info("Deleted voice clone id=%d name=%r", clone.id, clone.name)
        return {"status": "success", "id": clone.id, "name": clone.name}

    @router.get("/clone/list")
    async def clone_list() -> Any:
        clones = await asyncio.to_thread(clone_store.list)
        return [c.to_public_dict() for c in clones]

    # Apply any persisted default clone to the TTS handlers at startup so the
    # routed voice survives a server restart.
    _startup_default = clone_store.get_default()
    if _startup_default is not None:
        _apply_default(_startup_default)

    return router
