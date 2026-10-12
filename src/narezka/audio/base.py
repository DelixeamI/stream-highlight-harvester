"""Protocols and interfaces for audio processing (separation, VAD, LID)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class AudioSeparationBackend(Protocol):
    """Protocol for audio source separation backends (RoFormer, Demucs)."""

    def separate(
        self,
        input_audio: Path,
        output_voice: Path,
        output_accompaniment: Path,
        chunk_seconds: int = 600,
        device: str = "cuda",
    ) -> None:
        """Separate input audio into vocals/voice and accompaniment/background tracks."""
        ...


class VADBackend(Protocol):
    """Protocol for Voice Activity Detection backends."""

    def detect_speech(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        pad_ms: int = 150,
        device: str = "cpu",
    ) -> list[tuple[float, float]]:
        """Return list of speech intervals as (start_seconds, end_seconds)."""
        ...


class LIDBackend(Protocol):
    """Protocol for Language Identification backends."""

    def identify_language(
        self,
        audio_path: Path,
        intervals: list[tuple[float, float]],
        min_segment_s: float = 1.5,
        device: str = "cuda",
    ) -> list[dict[str, Any]]:
        """Identify language for speech intervals.

        Returns:
            List of dicts: {"start": float, "end": float, "lang": str, "confidence": float}
        """
        ...
