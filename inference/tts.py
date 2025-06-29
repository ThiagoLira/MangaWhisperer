from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from transformers import AutoTokenizer

# Parler imports
from parler_tts import ParlerTTSForConditionalGeneration

# Kokoro imports will be optional
try:
    from kokoro import KPipeline  # type: ignore
except Exception:  # pragma: no cover - kokoro may not be installed
    KPipeline = None


class BaseTTS(ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    def tts_from_text(self, text: str, description: Optional[str] = None) -> tuple[npt.NDArray, int]:
        """Generate audio from text.

        Returns a tuple ``(audio, sample_rate)``.
        """
        raise NotImplementedError


class ParlerTTS(BaseTTS):
    """Wrapper around the ParlerTTS model."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            "2121-8/japanese-parler-tts-mini"
        ).to(self.device)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(
            "2121-8/japanese-parler-tts-mini", subfolder="prompt_tokenizer"
        )
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            "2121-8/japanese-parler-tts-mini", subfolder="description_tokenizer"
        )

    def tts_from_text(self, text: str, description: Optional[str] = None) -> tuple[npt.NDArray, int]:
        prompt = text
        description = description or ""
        input_ids = self.description_tokenizer(description, return_tensors="pt").input_ids.to(
            self.device
        )
        prompt_input_ids = self.prompt_tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        return generation.cpu().numpy().squeeze(), 44100


class KokoroTTS(BaseTTS):
    """Wrapper around the Kokoro TTS pipeline."""

    def __init__(self, voice: str = "jf_tebukuro", device: Optional[str] = None) -> None:
        if KPipeline is None:
            raise ImportError("kokoro library is required for KokoroTTS")
        self.voice = voice
        self.pipeline = KPipeline(lang_code="j", device=device)

    def tts_from_text(self, text: str, description: Optional[str] = None) -> tuple[npt.NDArray, int]:
        voice = self.voice
        audio_segments = []
        for result in self.pipeline(text, voice=voice, split_pattern=r"\n+"):
            if result.audio is not None:
                audio_segments.append(result.audio.numpy())
        audio = np.concatenate(audio_segments) if audio_segments else np.array([], dtype=np.float32)
        return audio, 24000


class TTS(BaseTTS):
    """Main entry point for TTS with pluggable providers."""

    providers = {
        "parler": ParlerTTS,
        "kokoro": KokoroTTS,
    }

    def __init__(self, provider: str = "parler", **kwargs) -> None:
        provider = provider.lower()
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        self.backend = self.providers[provider](**kwargs)

    def tts_from_text(self, text: str, description: Optional[str] = None) -> tuple[npt.NDArray, int]:
        return self.backend.tts_from_text(text, description)
