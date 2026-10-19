"""
Kitten-TTS 2 Handler

Supports KittenML's Kitten-TTS-2 (1.7B ternary speech language model plus a
flow-matching vocoder) for high-quality multilingual speech synthesis with
in-context voice cloning.

- Regular mode: the ``kittenml`` PyTorch backend, CUDA when available, CPU
  otherwise.
- CPU mode (``--use-cpu``): the native ``kitten_tts_cpp`` (llama.cpp) backend
  when installed (and no feature forces the PyTorch path), otherwise the
  ``kittenml`` PyTorch CPU path.

The model outputs 24 kHz mono float32 audio, which is resampled to the
pipeline's 16 kHz int16 format.
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from threading import Event
from typing import Any, Iterator

import numpy as np
import scipy.signal
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.clones.audio import MIN_CLONE_AUDIO_SECONDS
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SAMPLE_RATE = 16000
DEFAULT_MODEL = "KittenML/kitten-tts-2"
KITTEN_SAMPLE_RATE = 24000
# The clone store's upload floor, reused as kitten's hard rejection floor so
# the API never accepts a clip the backend then refuses.
KITTEN_MIN_CLONE_SECONDS = MIN_CLONE_AUDIO_SECONDS

# Languages with a dedicated Kitten-TTS-2 voice. The model can speak more,
# but these are the ones with native voice presets.
KITTEN_LANGUAGE_CODES = frozenset({"ar", "de", "en", "es", "fr", "hi", "it", "pt", "ru", "zh"})

# The nine language-named voices are how Kitten-TTS-2 reaches those languages:
# the voice is what carries the accent.
KITTEN_LANGUAGE_VOICES = {
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}


def _voice_name(voice: object) -> str | None:
    """Extract a voice name from a protocol value (string or voice-ID object)."""
    if voice is None:
        return None
    if not isinstance(voice, str):
        # The realtime protocol also allows a voice-ID object.
        voice = getattr(voice, "id", None)
    if voice is None:
        return None
    name = str(voice).strip()
    return name or None


class KittenTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech using KittenML's Kitten-TTS-2 model.

    47 built-in voices, in-context voice cloning from a reference clip
    (5-30 seconds recommended, 2 seconds minimum), and expression control.
    Output is 24 kHz mono float32.
    """

    def setup(
        self,
        should_listen: Event,
        model_name: str = DEFAULT_MODEL,
        use_cpu: bool = False,
        voice: str = "Bruno",
        preset: str = "stable",
        weights: str = "packed",
        decoder: str = "default",
        default_language: str = "auto",
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_kwargs: object,
    ) -> None:
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}")

        self.should_listen = should_listen
        # Bypass the speaker property: the CLI voice is the unpinned default,
        # so the language-voice auto-selection may still apply.
        self._speaker = voice
        self._speaker_pinned = False
        self._initial_voice = voice
        self.preset = preset
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self._gen_kwargs = dict(gen_kwargs or {})
        normalized_default_language = self._normalize_language_code(default_language)
        self._default_language: str | None = (
            None if normalized_default_language in ("", "auto") else normalized_default_language
        )
        if self._default_language is not None and self._default_language not in KITTEN_LANGUAGE_CODES:
            logger.warning(
                "Kitten TTS default language %r has no dedicated voice; the default voice will "
                "be used with English text normalization disabled",
                self._default_language,
            )
        self._initial_clone_ref: tuple[Path, str | None] | None = (
            (Path(ref_audio).expanduser(), ref_text) if ref_audio else None
        )
        self._clone_ref = self._initial_clone_ref
        if ref_text and not ref_audio:
            logger.warning("Kitten TTS: ref_text has no effect without ref_audio; ignoring it")
        if self._initial_clone_ref is not None:
            # The CLI path cannot reject the process for a marginal clip, so it
            # reports the same quality guidance as the runtime path (which
            # raises) instead of staying silent.
            self._log_reference_duration_guidance(self._initial_clone_ref[0])

        self._load_model(model_name, use_cpu, weights, decoder)
        self._validate_model_config(voice, preset, weights, decoder)
        self.warmup()

    @property
    def speaker(self) -> str:
        return self._speaker

    @speaker.setter
    def speaker(self, value: str) -> None:
        # External activation (e.g. the audio engine's _activate_speaker) is an
        # explicit voice choice that wins over the language-voice auto-selection.
        # Resolve case-insensitively when the voice list is known: the engine
        # passes client strings through unvalidated when the model exposes no
        # list, and the model's own voice lookup is case-sensitive.
        supported = getattr(self, "_supported_voices", None)
        if supported:
            value = supported.get(value.lower(), value)
        self._speaker = value
        self._speaker_pinned = True

    def _load_model(self, model_name: str, use_cpu: bool, weights: str, decoder: str) -> None:
        if use_cpu:
            self._setup_cpu(model_name, weights, decoder)
        else:
            self._setup_regular(model_name, weights, decoder)

    def _setup_regular(self, model_name: str, weights: str, decoder: str) -> None:
        try:
            from kittenml import KittenTTS
        except ImportError as exc:
            raise ImportError(
                "kittenml is required for the Kitten TTS backend. "
                'Install it with `pip install "speech-to-speech[kitten]"`.'
            ) from exc
        self.backend = "kittenml"
        # device=None lets kittenml auto-select CUDA when available, CPU otherwise.
        self.tts = self._build_kittenml(KittenTTS, model_name, None, weights, decoder)
        self.device = str(getattr(self.tts, "device", "auto"))
        logger.info("Loaded Kitten TTS (%s) via kittenml on %s", model_name, self.device)

    def _setup_cpu(self, model_name: str, weights: str, decoder: str) -> None:
        # Prefer the native C++ backend: fastest CPU inference. It only supports
        # packed weights and cannot do raw-audio voice cloning, so either
        # condition forces the PyTorch path.
        if self._clone_ref is None and weights == "packed":
            try:
                from kitten_tts_cpp import KittenTTS as CppKittenTTS
            except ImportError as exc:
                logger.warning("kitten_tts_cpp is not importable (%s); using the kittenml CPU path", exc)
                CppKittenTTS = None
            if CppKittenTTS is not None:
                self.backend = "kitten_tts_cpp"
                # The C++ engine is CPU-only, so no device argument is needed.
                # `preset`/`normalize` are accepted by kitten-tts-cpp 0.1.0's
                # generate() even though they are not in its published docs.
                self.tts = CppKittenTTS(model_name, decoder=decoder, weights=weights)
                self.device = "cpu"
                logger.info("Loaded Kitten TTS (%s) via the native C++ CPU backend", model_name)
                return
        try:
            from kittenml import KittenTTS
        except ImportError as exc:
            raise ImportError(
                "CPU mode requires kitten_tts_cpp (preferred) or kittenml. "
                'Install it with `pip install "speech-to-speech[kitten]"`.'
            ) from exc
        self.backend = "kittenml"
        self.tts = self._build_kittenml(KittenTTS, model_name, "cpu", weights, decoder)
        self.device = "cpu"
        if self._clone_ref is not None:
            logger.info(
                "Loaded Kitten TTS (%s) via kittenml on CPU (raw-audio cloning needs the PyTorch backend)",
                model_name,
            )
        elif weights != "packed":
            logger.info(
                "Loaded Kitten TTS (%s) via kittenml on CPU (the C++ backend only supports packed weights)",
                model_name,
            )
        else:
            logger.info("Loaded Kitten TTS (%s) via kittenml on CPU", model_name)

    @staticmethod
    def _build_kittenml(KittenTTS: Any, model_name: str, device: str | None, weights: str, decoder: str) -> Any:
        try:
            return KittenTTS(model_name, device=device, decoder=decoder, weights=weights)
        except TypeError as exc:
            message = str(exc)
            if "onnx" in message.lower():
                # A legacy KITTEN1/ONNX repo (e.g. kitten-tts-nano) routed through
                # the ONNX model class, which rejects the KITTEN2 kwargs.
                raise ValueError(
                    f"Model {model_name!r} is not a Kitten-TTS-2 checkpoint; the kitten backend "
                    f"only supports KITTEN2 repos such as KittenML/kitten-tts-2. Original error: {message}"
                ) from exc
            if "decoder" in message or "weights" in message:
                # A kittenml older than 0.9.2 whose constructor predates the
                # decoder/weights API.
                raise ValueError(
                    "This kittenml version is too old for the Kitten-TTS-2 decoder/weights API. "
                    "Upgrade with `pip install -U 'kittenml>=0.9.4'`. Original error: " + message
                ) from exc
            raise ValueError(f"Failed to load Kitten-TTS-2 model {model_name!r}. Original error: {message}") from exc

    def _validate_model_config(self, voice: str, preset: str, weights: str, decoder: str) -> None:
        """Validate the CLI config against what the loaded model declares.

        Static misconfiguration (a typo in voice/preset/weights/decoder) would
        otherwise only surface per-utterance at runtime, so fail fast here. A
        missing capability list is treated as "unknown" (warn, don't fail) so
        exotic repos that do not publish one can still load.
        """
        voices = self.get_supported_speakers()
        if voices:
            # Cache the lowercase lookup once; it is immutable model metadata.
            self._supported_voices = {str(v).lower(): str(v) for v in voices}
            if voice.lower() not in self._supported_voices:
                raise ValueError(
                    f"Kitten TTS voice {voice!r} is not available in this model "
                    f"(--kitten_tts_voice). Choose one of: {', '.join(sorted(voices, key=str.lower))}"
                )
            missing = sorted(v for v in KITTEN_LANGUAGE_VOICES.values() if v.lower() not in self._supported_voices)
            if missing:
                logger.warning(
                    "Kitten TTS model is missing language voices %s; those languages will use the default voice",
                    ", ".join(missing),
                )
        else:
            self._supported_voices = {}
            logger.warning("Kitten TTS model exposes no voice list; runtime voice selection will be unvalidated")
        presets = getattr(self.tts, "decode_presets", None)
        if presets is None:
            config = getattr(self.tts, "config", None)
            if isinstance(config, dict):
                presets = config.get("decode_presets")
        if presets and preset not in presets:
            raise ValueError(
                f"Kitten TTS preset {preset!r} is not available (--kitten_tts_preset). "
                f"Choose from: {', '.join(sorted(str(p) for p in presets))}"
            )
        available_weights = getattr(self.tts, "available_weights", None)
        if available_weights and weights not in available_weights:
            raise ValueError(
                f"Kitten TTS weights variant {weights!r} is not available (--kitten_tts_weights). "
                f"Choose from: {', '.join(sorted(str(w) for w in available_weights))}"
            )
        decoders = getattr(self.tts, "available_decoders", None)
        if callable(decoders):
            decoders = decoders()
        if decoders and decoder not in decoders:
            raise ValueError(
                f"Kitten TTS decoder variant {decoder!r} is not available (--kitten_tts_decoder). "
                f"Choose from: {', '.join(sorted(str(d) for d in decoders))}"
            )

    def warmup(self) -> None:
        logger.info("Warming up Kitten TTS...")
        # A warmup failure should not take the whole server down at startup;
        # real-utterance generation failures are already handled by the handler
        # run loop, so degrade to a warning here, mirroring qwen3.
        try:
            self._generate("Warmup", self.speaker, None, None, True)
        except Exception as exc:
            logger.warning("Kitten TTS warmup generation failed: %s", exc)

    def _generate(
        self,
        text: str,
        voice: str | None,
        reference: Path | None,
        reference_text: str | None,
        normalize: bool,
    ) -> np.ndarray:
        kwargs: dict[str, Any] = dict(self._gen_kwargs)
        # Speaker selection is owned by this handler; never let raw gen_kwargs
        # sneak a second speaker in (kittenml rejects voice+reference together).
        kwargs.pop("voice", None)
        kwargs.pop("reference", None)
        kwargs.pop("reference_text", None)
        if reference is not None:
            kwargs["reference"] = str(reference)
            if reference_text:
                kwargs["reference_text"] = reference_text
        elif voice:
            kwargs["voice"] = voice
        kwargs.setdefault("preset", self.preset)
        # Text normalization is tuned for English; disable it for other languages.
        kwargs.setdefault("normalize", normalize)
        audio = self.tts.generate(text, **kwargs)
        return np.asarray(audio, dtype=np.float32).squeeze()

    def _resolve_speaker(self, tts_input: TTSInput, language: str | None) -> tuple[str | None, Path | None, str | None]:
        """Pick the speaker for this request: active clone > pinned voice > language voice > current speaker."""
        # A routed voice clone takes precedence over per-request voice pins,
        # mirroring qwen3: without this guard, any client that sets `voice` in
        # session.update or response.create would bypass the clone.
        # Bind once so a concurrent clear cannot land between the check and the unpack.
        clone_ref = self._clone_ref
        if clone_ref is not None:
            reference, reference_text = clone_ref
            return None, reference, reference_text
        pinned: str | None = None
        response = tts_input.response
        if response and response.audio and response.audio.output:
            pinned = _voice_name(response.audio.output.voice)
        if pinned is None and tts_input.runtime_config is not None:
            audio_cfg = tts_input.runtime_config.session.audio
            audio_output = audio_cfg.output if audio_cfg is not None else None
            if audio_output is not None:
                pinned = _voice_name(audio_output.voice)
        if pinned is not None:
            # Validate: an unknown name (e.g. an OpenAI default like "alloy" in a
            # session.update) would hard-fail every utterance, so degrade to the
            # language/default voice with a warning, mirroring qwen3. When the
            # model exposes no voice list at all, any pin degrades the same way
            # (the engine boundary, by contrast, activates unvalidated).
            supported = getattr(self, "_supported_voices", None)
            if supported is None:
                supported = {s.lower(): s for s in self.get_supported_speakers()}
            resolved = supported.get(pinned.lower())
            if resolved is None:
                logger.warning(
                    "Kitten TTS: ignoring pinned voice %r because it is not a supported Kitten voice",
                    pinned,
                )
            else:
                return resolved, None, None
        # The language-named voices carry the accent for their language, but a
        # speaker activated at runtime (e.g. via /v1/audio/speech) wins.
        language_voice = KITTEN_LANGUAGE_VOICES.get(language or "")
        if language_voice and not self._speaker_pinned:
            return language_voice, None, None
        return self.speaker, None, None

    @staticmethod
    def _normalize_language_code(language_code: str) -> str:
        normalized = language_code.strip().lower()
        if normalized.endswith("-auto"):
            normalized = normalized.removesuffix("-auto")
        return normalized.split("-", 1)[0]

    def _resolve_language(self, language_code: str | None) -> str | None:
        if not language_code:
            return None
        normalized = self._normalize_language_code(language_code)
        if normalized in KITTEN_LANGUAGE_CODES:
            return normalized
        logger.debug(
            "Kitten TTS has no dedicated voice for language code %r; using the default voice",
            language_code,
        )
        return None

    def _is_cancelled(self, generation: int | None) -> bool:
        return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)

        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.wait_for_gate(tts_input.turn_id, tts_input.turn_revision):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.wait_for_gate(
            tts_input.turn_id,
            tts_input.turn_revision,
            commit=True,
        ):
            logger.debug(
                "Dropping stale TTS input for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return

        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None

        text = tts_input.text
        if not text.strip():
            return

        language_code = tts_input.tts_language_code
        if not language_code and self._default_language is not None:
            # The OpenAI-compatible audio endpoints do not carry a language;
            # fall back to the operator's default.
            language_code = self._default_language
        normalized_code = self._normalize_language_code(language_code or "")
        selected = tts_input.selected_language
        if selected not in (None, "auto") and normalized_code not in KITTEN_LANGUAGE_CODES:
            # A detected language Kitten has no dedicated voice for keeps the session language.
            language_code = selected
            normalized_code = self._normalize_language_code(language_code)
        language = self._resolve_language(language_code)
        # Normalization is decided from the raw code: the model speaks languages
        # without a dedicated voice, and the English normalizer mangles them.
        normalize = normalized_code in ("", "en")

        voice, reference, reference_text = self._resolve_speaker(tts_input, language)

        console.print(f"[green]ASSISTANT: {text}")

        audio = self._generate(text, voice, reference, reference_text, normalize)

        # If the user interrupted during the blocking generate(), drop the buffer.
        if self._is_cancelled(cancel_gen):
            logger.info("Kitten TTS output cancelled (interruption)")
            return

        source_sample_rate = int(getattr(self.tts, "sample_rate", KITTEN_SAMPLE_RATE))
        divisor = gcd(source_sample_rate, PIPELINE_SAMPLE_RATE)
        audio_16k_float = scipy.signal.resample_poly(
            audio,
            PIPELINE_SAMPLE_RATE // divisor,
            source_sample_rate // divisor,
        )

        # Convert to int16 format expected by the audio pipeline
        audio_int16 = np.clip(audio_16k_float * 32768, -32768, 32767).astype(np.int16)

        # Yield in block-aligned chunks so the streamer can handle it smoothly
        n = (len(audio_int16) // self.blocksize) * self.blocksize
        for i in range(0, n, self.blocksize):
            if self._is_cancelled(cancel_gen):
                logger.info("Kitten TTS output cancelled (interruption)")
                return
            yield audio_int16[i : i + self.blocksize]

        # Pad the tail so the audio streamer's fixed-blocksize callback doesn't crash
        if n < len(audio_int16):
            if self._is_cancelled(cancel_gen):
                logger.info("Kitten TTS output cancelled (interruption)")
                return
            tail = audio_int16[n:]
            yield np.pad(tail, (0, self.blocksize - len(tail)))

    def get_supported_speakers(self) -> list[str]:
        cached = getattr(self, "_supported_voices", None)
        if cached is not None:
            return list(cached.values())
        voices = getattr(self.tts, "available_voices", None)
        return [str(v) for v in voices] if voices else []

    def set_clone_reference(self, audio_path: str | Path, transcription: str | None = None) -> None:
        if self.backend == "kitten_tts_cpp":
            raise ValueError(
                "Raw-audio voice cloning is not supported by the kitten_tts_cpp backend. "
                "Restart without --use-cpu to clone from a reference clip."
            )
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Clone reference audio not found: {path}")
        duration = self._reference_duration(path)
        # Gate on the platform clone contract (2s floor, set by the clone
        # store), not the model's 5s quality guidance: clips the API accepted
        # at upload must keep working here.
        if duration is not None and not (KITTEN_MIN_CLONE_SECONDS <= duration <= 30.0):
            raise ValueError(
                f"Clone reference {path.name} is {duration:.1f}s long; Kitten-TTS-2 expects a "
                f"{KITTEN_MIN_CLONE_SECONDS:.0f}-30 second reference clip"
            )
        self._log_reference_duration_guidance(path, duration)
        # Publish atomically so process() never sees a torn (path, text) pair.
        self._clone_ref = (path, transcription)
        logger.info("Kitten TTS clone reference set: %s", path.name)

    @classmethod
    def _log_reference_duration_guidance(cls, path: Path, duration: float | None = None) -> None:
        """Warn (never raise) when a reference clip is outside the 5-30s sweet spot.

        Callers have different contracts: setup() uses this for the CLI
        reference (which it cannot reject), while set_clone_reference() calls
        it after its hard gate, so the sub-floor branch is only reachable from
        setup().
        """
        if duration is None:
            duration = cls._reference_duration(path)
        if duration is None:
            return
        if duration < KITTEN_MIN_CLONE_SECONDS or duration > 30.0:
            logger.warning(
                "Kitten TTS reference %s is %.1fs long; %.0f-30s clips give the best clone quality",
                path.name,
                duration,
                KITTEN_MIN_CLONE_SECONDS,
            )
        elif duration < 5.0:
            logger.warning(
                "Kitten TTS reference %s is %.1fs; 5-30s gives the best clone quality",
                path.name,
                duration,
            )

    @staticmethod
    def _reference_duration(path: Path) -> float | None:
        """Probe the reference clip's duration, or None when it cannot be read."""
        try:
            import soundfile

            return float(soundfile.info(str(path)).duration)
        except Exception:  # noqa: BLE001 - unreadable formats fall through to wave
            pass
        try:
            import wave

            with wave.open(str(path), "rb") as wav:
                rate = wav.getframerate()
                if rate:
                    return wav.getnframes() / rate
        except Exception:  # noqa: BLE001 - give up on probing, keep the file check only
            pass
        return None

    def clear_clone_reference(self) -> None:
        # Restore the CLI-provided reference (if any) instead of clearing it
        # permanently: the audio engine clears the clone whenever a built-in
        # voice is selected.
        self._clone_ref = self._initial_clone_ref
        logger.debug("Kitten TTS clone reference cleared")

    def on_session_end(self) -> None:
        # Restore the CLI voice (unpinned, so the language-voice auto-selection
        # applies again). The active clone (if any) is router-managed and
        # survives session boundaries.
        self._speaker = self._initial_voice
        self._speaker_pinned = False
        logger.debug("Kitten TTS session state reset")

    def cleanup(self) -> None:
        close = getattr(getattr(self, "tts", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("Kitten TTS cleanup failed: %s", exc)
