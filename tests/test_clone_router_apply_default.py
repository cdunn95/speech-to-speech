"""Regression tests: /clone/default must report why a clone was not applied."""

from __future__ import annotations

import io
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from speech_to_speech.api.openai_realtime.clone_router import create_clone_router
from speech_to_speech.clones.store import CloneStore


class _RejectingTTS:
    """A TTS handler that rejects raw-audio cloning (e.g. the kitten C++ backend)."""

    def set_clone_reference(self, audio_path, transcription=None) -> None:
        raise ValueError("Raw-audio voice cloning is not supported by the kitten_tts_cpp backend.")

    def clear_clone_reference(self) -> None:
        pass


class _PipelineUnit:
    def __init__(self, handlers: list) -> None:
        self.handlers = handlers


def _wav_bytes(seconds: float = 5.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path) -> TestClient:
    store = CloneStore(tmp_path / "clones")
    store.create("test-voice", None, "a transcript", _wav_bytes())
    router = create_clone_router(store, [_PipelineUnit([_RejectingTTS()])])
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_clone_default_reports_backend_rejection(client: TestClient) -> None:
    response = client.post("/clone/default", json={"clone": "test-voice"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    # The generic "no handlers support cloning" warning would be misleading
    # here: a handler exists but rejected the clone. The reason must be named.
    assert "kitten_tts_cpp" in body["warning"]


def test_clone_default_keeps_generic_warning_without_handlers(tmp_path) -> None:
    store = CloneStore(tmp_path / "clones")
    store.create("test-voice", None, "a transcript", _wav_bytes())
    router = create_clone_router(store, [])
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.post("/clone/default", json={"clone": "test-voice"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "No TTS handlers support voice cloning" in body["warning"]
