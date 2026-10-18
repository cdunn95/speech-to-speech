from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS import kitten_tts_handler as kitten_module
from speech_to_speech.TTS.kitten_tts_handler import KittenTTSHandler
from tests.turns import reopen


class FakeTTS:
    sample_rate = 24000
    available_voices = ["Bruno", "Bella", "Luna", "Kiki", "German"]

    def __init__(self, generate: Callable[..., np.ndarray]) -> None:
        self.generate = generate


def make_handler(
    generate: Callable[..., np.ndarray],
    *,
    blocksize: int = 4,
    cancel_scope: CancelScope | None = None,
    speaker: str = "Bruno",
) -> KittenTTSHandler:
    handler = KittenTTSHandler.__new__(KittenTTSHandler)
    handler.tts = FakeTTS(generate)
    handler.backend = "kittenml"
    # Unpinned, like the CLI-configured default voice.
    handler._speaker = speaker
    handler._speaker_pinned = False
    handler._initial_voice = speaker
    handler.preset = "stable"
    handler.blocksize = blocksize
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    handler._gen_kwargs = {}
    handler._default_language = None
    handler._initial_clone_ref = None
    handler._clone_ref = None
    return handler


def _install_fake_kitten_modules(monkeypatch: pytest.MonkeyPatch, *, with_cpp: bool) -> list[dict]:
    """Fake the kittenml / kitten_tts_cpp imports and record constructor calls."""
    calls: list[dict] = []

    class FakeKittenTTS:
        def __init__(self, model_name, **kwargs) -> None:
            calls.append({"model_name": model_name, **kwargs})
            self.device = kwargs.get("device") or "cuda"
            self.sample_rate = 24000
            self.available_voices = ["Bruno", "Bella"]

        def generate(self, text, **kwargs) -> np.ndarray:
            return np.zeros(8, dtype=np.float32)

    kittenml_module = types.ModuleType("kittenml")
    kittenml_module.KittenTTS = FakeKittenTTS
    monkeypatch.setitem(sys.modules, "kittenml", kittenml_module)
    if with_cpp:
        cpp_module = types.ModuleType("kitten_tts_cpp")
        cpp_module.KittenTTS = FakeKittenTTS
        monkeypatch.setitem(sys.modules, "kitten_tts_cpp", cpp_module)
    else:
        monkeypatch.setitem(sys.modules, "kitten_tts_cpp", None)
    return calls


def test_setup_rejects_invalid_blocksize() -> None:
    handler = KittenTTSHandler.__new__(KittenTTSHandler)

    with pytest.raises(ValueError, match="blocksize must be positive"):
        handler.setup(Event(), blocksize=0)


def test_regular_mode_uses_kittenml_with_auto_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_kitten_modules(monkeypatch, with_cpp=True)
    handler = KittenTTSHandler(Event(), Queue(), Queue(), setup_args=(Event(),))

    assert handler.backend == "kittenml"
    assert handler.device == "cuda"
    assert calls == [{"model_name": "KittenML/kitten-tts-2", "device": None, "decoder": "default", "weights": "packed"}]


def test_use_cpu_prefers_the_native_cpp_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_kitten_modules(monkeypatch, with_cpp=True)
    handler = KittenTTSHandler(Event(), Queue(), Queue(), setup_args=(Event(),), setup_kwargs={"use_cpu": True})

    assert handler.backend == "kitten_tts_cpp"
    assert handler.device == "cpu"
    assert calls == [{"model_name": "KittenML/kitten-tts-2", "decoder": "default", "weights": "packed"}]


def test_use_cpu_falls_back_to_kittenml_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_kitten_modules(monkeypatch, with_cpp=False)
    handler = KittenTTSHandler(Event(), Queue(), Queue(), setup_args=(Event(),), setup_kwargs={"use_cpu": True})

    assert handler.backend == "kittenml"
    assert handler.device == "cpu"
    assert calls == [
        {"model_name": "KittenML/kitten-tts-2", "device": "cpu", "decoder": "default", "weights": "packed"}
    ]


def test_use_cpu_with_ref_audio_uses_kittenml_for_cloning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fake_kitten_modules(monkeypatch, with_cpp=True)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    handler = KittenTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"use_cpu": True, "ref_audio": str(ref)},
    )

    assert handler.backend == "kittenml"
    assert handler.device == "cpu"
    assert handler._clone_ref == (ref, None)
    assert calls == [
        {"model_name": "KittenML/kitten-tts-2", "device": "cpu", "decoder": "default", "weights": "packed"}
    ]


def test_use_cpu_with_non_packed_weights_uses_kittenml(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_kitten_modules(monkeypatch, with_cpp=True)
    handler = KittenTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"use_cpu": True, "weights": "emb4"},
    )

    assert handler.backend == "kittenml"
    assert handler.device == "cpu"
    assert calls == [{"model_name": "KittenML/kitten-tts-2", "device": "cpu", "decoder": "default", "weights": "emb4"}]


@pytest.mark.parametrize(
    ("pipeline_code", "expected_normalize"),
    [
        ("en-auto", True),
        ("pt-BR", False),
        ("no", False),
        ("zh", False),
        (None, True),
    ],
)
def test_process_normalizes_or_falls_back_from_pipeline_language_codes(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_code: str | None,
    expected_normalize: bool,
) -> None:
    normalizes = []

    def generate(text, **kwargs):
        normalizes.append(kwargs["normalize"])
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate)

    list(handler.process(TTSInput(text="Hello", language_code=pipeline_code)))

    assert normalizes == [expected_normalize]


@pytest.mark.parametrize(("detected", "expected_normalize"), [("fr", False), ("no", True)])
def test_process_keeps_session_language_for_unsupported_detected_language(
    monkeypatch: pytest.MonkeyPatch, detected: str, expected_normalize: bool
) -> None:
    normalizes = []

    def generate(text, **kwargs):
        normalizes.append(kwargs["normalize"])
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate)

    list(handler.process(TTSInput(text="Hello", selected_language="en", tts_language_code=detected)))

    assert normalizes == [expected_normalize]


def test_process_clips_int16_audio_and_pads_the_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(text, **_kwargs):
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(
        kitten_module.scipy.signal,
        "resample_poly",
        lambda *_args: np.array([2.0, -2.0, 0.5], dtype=np.float32),
    )
    handler = make_handler(generate)

    chunks = list(handler.process(TTSInput(text="Hello", language_code="en")))

    assert len(chunks) == 1
    assert chunks[0].dtype == np.int16
    np.testing.assert_array_equal(chunks[0], np.array([32767, -32768, 16384, 0], dtype=np.int16))


def test_process_drops_audio_when_cancelled_during_synthesis() -> None:
    cancel_scope = CancelScope()

    def generate(text, **_kwargs):
        cancel_scope.cancel()
        return np.zeros(8, dtype=np.float32)

    handler = make_handler(generate, cancel_scope=cancel_scope)

    assert list(handler.process(TTSInput(text="Hello", language_code="en"))) == []


def test_stale_keyed_terminal_becomes_cleanup_after_lm_tts_handoff() -> None:
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    handler = KittenTTSHandler.__new__(KittenTTSHandler)
    handler.speculative_turns = tracker
    terminal = EndOfResponse(
        response_key="response_1",
        turn_id="turn_1",
        turn_revision=0,
        cancel_generation=7,
    )
    reopen(tracker)

    outputs = list(handler.process(terminal))
    queued = handler.output_for_queue(outputs[0], terminal)

    assert outputs == [AUDIO_RESPONSE_DONE]
    assert terminal.cleanup_only is True
    assert isinstance(queued, AudioOutput)
    assert queued.response_key == "response_1"
    assert queued.cancel_generation == 7
    assert queued.cleanup_only is True


def test_process_checks_cancellation_before_padded_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    cancel_scope = CancelScope()

    def generate(text, **_kwargs):
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(5))
    handler = make_handler(generate, cancel_scope=cancel_scope)
    chunks = handler.process(TTSInput(text="Hello", language_code="en"))

    assert len(next(chunks)) == 4
    cancel_scope.cancel()
    with pytest.raises(StopIteration):
        next(chunks)


def test_process_prefers_active_clone_over_pinned_response_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    handler._clone_ref = (Path("ref.wav"), "reference transcript")

    list(handler.process(TTSInput(text="Hello", response={"audio": {"output": {"voice": "Bella"}}})))

    assert seen[0]["reference"] == "ref.wav"
    assert seen[0]["reference_text"] == "reference transcript"
    assert "voice" not in seen[0]


def test_process_prefers_active_clone_over_pinned_session_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    handler._clone_ref = (Path("ref.wav"), "reference transcript")
    runtime_config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(
            type="realtime",
            audio={"output": {"voice": "Luna"}},
        )
    )

    list(handler.process(TTSInput(text="Hello", runtime_config=runtime_config)))

    assert seen[0]["reference"] == "ref.wav"
    assert "voice" not in seen[0]


def test_process_prefers_pinned_response_voice_over_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")

    list(handler.process(TTSInput(text="Hello", response={"audio": {"output": {"voice": "Bella"}}})))

    assert seen[0]["voice"] == "Bella"
    assert "reference" not in seen[0]


def test_process_prefers_pinned_session_voice_over_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    runtime_config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(
            type="realtime",
            audio={"output": {"voice": "Luna"}},
        )
    )

    list(handler.process(TTSInput(text="Hello", runtime_config=runtime_config)))

    assert seen[0]["voice"] == "Luna"
    assert "reference" not in seen[0]


def test_process_uses_the_language_voice_for_dedicated_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")

    list(handler.process(TTSInput(text="Hallo", language_code="de")))

    assert seen[0]["voice"] == "German"
    assert seen[0]["normalize"] is False
    assert "reference" not in seen[0]


def test_process_language_voice_yields_to_pinned_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")

    list(
        handler.process(TTSInput(text="Hallo", language_code="de", response={"audio": {"output": {"voice": "Bella"}}}))
    )

    assert seen[0]["voice"] == "Bella"
    assert "reference" not in seen[0]


def test_process_language_voice_yields_to_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    handler._clone_ref = (Path("ref.wav"), "reference transcript")

    list(handler.process(TTSInput(text="Hallo", language_code="de")))

    assert seen[0]["reference"] == "ref.wav"
    assert "voice" not in seen[0]


def test_process_falls_back_to_the_current_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Kiki")

    list(handler.process(TTSInput(text="Hello")))

    assert seen[0]["voice"] == "Kiki"
    assert "reference" not in seen[0]


def test_default_language_applies_when_no_pipeline_language(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    handler._default_language = "de"

    list(handler.process(TTSInput(text="Hallo")))

    assert seen[0]["voice"] == "German"
    assert seen[0]["normalize"] is False


def test_preset_reaches_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    handler.preset = "expressive"

    list(handler.process(TTSInput(text="Hello", language_code="en")))

    assert seen[0]["preset"] == "expressive"


def test_gen_kwargs_cannot_override_speaker_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Kiki")
    handler._gen_kwargs = {"voice": "Luna", "reference": "/other.wav", "max_new_tokens": 42}

    list(handler.process(TTSInput(text="Hello")))

    assert seen[0]["voice"] == "Kiki"
    assert "reference" not in seen[0]
    assert seen[0]["max_new_tokens"] == 42


def test_clone_reference_set_and_clear(tmp_path: Path) -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")

    handler.set_clone_reference(ref, "a transcript")
    assert handler._clone_ref == (ref, "a transcript")

    handler.clear_clone_reference()
    assert handler._clone_ref is None


def test_clear_clone_reference_restores_the_setup_reference(tmp_path: Path) -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    handler._initial_clone_ref = (ref, "setup transcript")
    handler._clone_ref = handler._initial_clone_ref

    handler.set_clone_reference(ref, "routed transcript")
    assert handler._clone_ref == (ref, "routed transcript")

    handler.clear_clone_reference()
    assert handler._clone_ref == (ref, "setup transcript")


def test_set_clone_reference_rejects_missing_file() -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))

    with pytest.raises(ValueError, match="Clone reference audio not found"):
        handler.set_clone_reference("/nonexistent/ref.wav", "text")


def test_set_clone_reference_rejects_the_cpp_backend(tmp_path: Path) -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))
    handler.backend = "kitten_tts_cpp"
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="not supported by the kitten_tts_cpp backend"):
        handler.set_clone_reference(ref, "text")


def test_get_supported_speakers() -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))

    assert handler.get_supported_speakers() == ["Bruno", "Bella", "Luna", "Kiki", "German"]


def test_on_session_end_restores_the_initial_voice() -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32), speaker="Bruno")
    handler.speaker = "Bella"

    handler.on_session_end()

    assert handler.speaker == "Bruno"


def test_on_session_end_keeps_active_clone() -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32), speaker="Bruno")
    handler.speaker = "Bella"
    handler._clone_ref = (Path("ref.wav"), "transcript")

    handler.on_session_end()

    assert handler.speaker == "Bruno"
    assert handler._clone_ref == (Path("ref.wav"), "transcript")


def test_api_activated_speaker_wins_over_language_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    # Simulate the audio engine activating a built-in speaker at runtime.
    handler.speaker = "Bella"

    list(handler.process(TTSInput(text="Hallo", language_code="de")))

    assert seen[0]["voice"] == "Bella"
    assert "reference" not in seen[0]


def test_normalize_disabled_for_non_dedicated_language(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")

    list(handler.process(TTSInput(text="こんにちは", language_code="ja")))

    assert seen[0]["normalize"] is False
    assert seen[0]["voice"] == "Bruno"


def test_warmup_generates_with_the_initial_speaker() -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append((text, kwargs))
        return np.zeros(8, dtype=np.float32)

    handler = make_handler(generate, speaker="Kiki")

    handler.warmup()

    assert seen == [("Warmup", {"voice": "Kiki", "preset": "stable", "normalize": True})]


def test_warmup_swallows_generation_errors() -> None:
    def generate(text, **_kwargs):
        raise RuntimeError("backend exploded during warmup")

    handler = make_handler(generate, speaker="Kiki")

    # A warmup failure must not propagate out of setup() and crash the server.
    handler.warmup()


def test_build_kittenml_rejects_non_kitten2_repos(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_kitten_tts(model_name, **kwargs):
        raise TypeError("Onnx.__init__() got an unexpected keyword argument 'decoder'")

    kittenml_module = types.ModuleType("kittenml")
    kittenml_module.KittenTTS = fake_kitten_tts
    monkeypatch.setitem(sys.modules, "kittenml", kittenml_module)

    with pytest.raises(ValueError, match="not a Kitten-TTS-2 checkpoint"):
        KittenTTSHandler(
            Event(),
            Queue(),
            Queue(),
            setup_args=(Event(),),
            setup_kwargs={"model_name": "KittenML/kitten-tts-nano-0.8"},
        )


def test_setup_normalizes_default_language(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_kitten_modules(monkeypatch, with_cpp=False)

    handler = KittenTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"default_language": "de-DE"},
    )
    assert handler._default_language == "de"

    handler = KittenTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"default_language": "auto"},
    )
    assert handler._default_language is None
    assert len(calls) == 2


def test_get_supported_speakers_returns_empty_list_when_unavailable() -> None:
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))
    handler.tts.available_voices = None

    assert handler.get_supported_speakers() == []


def test_cleanup_closes_the_model() -> None:
    closed = []

    class ClosableTTS(FakeTTS):
        def close(self) -> None:
            closed.append(True)

    handler = KittenTTSHandler.__new__(KittenTTSHandler)
    handler.tts = ClosableTTS(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))

    handler.cleanup()

    assert closed == [True]


def test_cleanup_swallows_close_errors() -> None:
    class FailingTTS(FakeTTS):
        def close(self) -> None:
            raise RuntimeError("engine already gone")

    handler = KittenTTSHandler.__new__(KittenTTSHandler)
    handler.tts = FailingTTS(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))

    handler.cleanup()  # must not raise


def test_pinned_voice_object_form_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")

    list(handler.process(TTSInput(text="Hello", response={"audio": {"output": {"voice": {"id": "Bella"}}}})))

    assert seen[0]["voice"] == "Bella"
    assert "reference" not in seen[0]


def test_unknown_pinned_voice_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")

    list(handler.process(TTSInput(text="Hello", response={"audio": {"output": {"voice": "alloy"}}})))

    assert seen[0]["voice"] == "Bruno"
    assert "reference" not in seen[0]


def test_on_session_end_unpins_the_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def generate(text, **kwargs):
        seen.append(kwargs)
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))
    handler = make_handler(generate, speaker="Bruno")
    handler.speaker = "Bella"
    handler.on_session_end()

    list(handler.process(TTSInput(text="Hallo", language_code="de")))

    assert seen[0]["voice"] == "German"


def test_set_clone_reference_expands_user_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    handler = make_handler(lambda text, **_kwargs: np.zeros(8, dtype=np.float32))

    handler.set_clone_reference("~/ref.wav", "text")

    assert handler._clone_ref == (ref, "text")


def test_setup_then_process_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    class RecordingKittenTTS:
        def __init__(self, model_name, **kwargs) -> None:
            self.device = "cpu"
            self.sample_rate = 24000
            self.available_voices = ["Bruno", "Bella", "German"]

        def generate(self, text, **kwargs) -> np.ndarray:
            seen.append(kwargs)
            return np.zeros(8, dtype=np.float32)

    kittenml_module = types.ModuleType("kittenml")
    kittenml_module.KittenTTS = RecordingKittenTTS
    monkeypatch.setitem(sys.modules, "kittenml", kittenml_module)
    monkeypatch.setitem(sys.modules, "kitten_tts_cpp", None)
    monkeypatch.setattr(kitten_module.scipy.signal, "resample_poly", lambda *_args: np.zeros(4))

    handler = KittenTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"voice": "Bruno", "blocksize": 4},
    )

    list(handler.process(TTSInput(text="Hallo", language_code="de")))

    # warmup + the German utterance; the language voice is auto-selected.
    assert [call["voice"] for call in seen] == ["Bruno", "German"]
    assert seen[1]["normalize"] is False
