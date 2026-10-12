"""Factory for instantiating ASR backends."""

from __future__ import annotations

from narezka.asr.base import ASRBackend
from narezka.asr.faster_whisper_backend import FasterWhisperBackend
from narezka.asr.gigaam_backend import GigaAMBackend


def get_asr_backend(name: str) -> ASRBackend:
    """Return an instantiated ASRBackend by name."""
    if name == "faster_whisper":
        return FasterWhisperBackend()
    elif name == "gigaam":
        return GigaAMBackend()
    raise ValueError(f"Неизвестный бэкенд ASR: '{name}'. Доступны: 'faster_whisper', 'gigaam'")
