from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KittenTTSHandlerArguments:
    kitten_tts_model_name: str = field(
        default="KittenML/kitten-tts-2",
        metadata={"help": "Hugging Face repo id for the Kitten-TTS model. Default is 'KittenML/kitten-tts-2'."},
    )
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": "Force CPU inference for Kitten-TTS. Uses the native C++ backend (kitten_tts_cpp) when "
            "installed, otherwise the PyTorch CPU path. Default is False (auto: CUDA when available, "
            "CPU otherwise)."
        },
    )
    kitten_tts_voice: str = field(
        default="Bruno",
        metadata={"help": "Built-in Kitten-TTS voice name (e.g. Bruno, Bella, Luna, Kiki). Default is 'Bruno'."},
    )
    kitten_tts_preset: str = field(
        default="stable",
        metadata={"help": "Kitten-TTS decoding preset: 'stable' or 'expressive'. Default is 'stable'."},
    )
    kitten_tts_weights: str = field(
        default="packed",
        metadata={"help": "Kitten-TTS weight variant: 'packed', 'emb4', or 'full'. Default is 'packed'."},
    )
    kitten_tts_decoder: str = field(
        default="default",
        metadata={"help": "Kitten-TTS decoder variant: 'default' or 'student_w4'. Default is 'default'."},
    )
    kitten_tts_default_language: str = field(
        default="auto",
        metadata={
            "help": "Default language code used when a request carries none (e.g. /v1/audio/speech). "
            "Selects the matching language voice and disables English text normalization. "
            "Default is 'auto' (English normalization on)."
        },
    )
    kitten_tts_ref_audio: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a 5-30s reference audio clip for in-context voice cloning."},
    )
    kitten_tts_ref_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional transcript of the reference clip. When omitted, the clip is transcribed "
            "with Whisper on every synthesis call, adding per-utterance latency."
        },
    )
    kitten_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in samples for streaming output. Default is 512."},
    )
