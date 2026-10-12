"""Data models and protocols for ASR (Speech Recognition) systems."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from narezka.config import AsrConfig


class Word(BaseModel):
    """Word with precise start and end timestamps."""

    text: str
    start: float
    end: float
    prob: float | None = None


class Segment(BaseModel):
    """Speech segment (phrase) with word-level timestamps and speaker/language tags."""

    id: int
    start: float
    end: float
    text: str
    speaker: str = "streamer"  # "streamer" | "friends" | "game" | "S1" ...
    lang: str = "ru"  # "ru" | "en" | ...
    words: list[Word] = Field(default_factory=list)
    source_track: str = "voice_sep"  # "mic" | "voice_sep" | "discord" | "mix"
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class SpeechSegment(BaseModel):
    """Input interval for ASR detected by VAD and classified by LID."""

    id: int
    start: float
    end: float
    lang: str = "ru"
    lang_confidence: float = 1.0
    speaker: str = "streamer"
    source_track: str = "voice_sep"
    is_game_voice_line: bool = False


class ASRBackend(Protocol):
    """Protocol implemented by all ASR backends (Faster-Whisper, GigaAM, NeMo)."""

    name: str

    def transcribe(
        self,
        audio_path: Path,
        segments: list[SpeechSegment],
        config: AsrConfig,
        device: str = "cuda",
    ) -> list[Segment]:
        """Transcribe speech segments from audio file."""
        ...
