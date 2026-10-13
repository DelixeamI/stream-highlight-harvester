"""Unit tests for Voice Activity Detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from narezka.audio.vad import SileroVADBackend, get_vad_backend, load_audio_tensor


def test_get_vad_backend() -> None:
    backend = get_vad_backend("silero")
    assert isinstance(backend, SileroVADBackend)


def test_load_audio_tensor(synthetic_audio_wav: Path) -> None:
    tensor = load_audio_tensor(synthetic_audio_wav, target_sr=16000)
    assert tensor.ndim == 1
    assert tensor.dtype.is_floating_point
    assert len(tensor) > 0


def test_silero_vad_mocked(synthetic_audio_wav: Path) -> None:
    mock_timestamps = [
        {"start": 0.2, "end": 1.1},
        {"start": 1.3, "end": 1.8},
    ]

    with patch("silero_vad.load_silero_vad") as mock_load:
        with patch("silero_vad.get_speech_timestamps", return_value=mock_timestamps):
            mock_load.return_value = MagicMock()

            backend = SileroVADBackend()
            intervals = backend.detect_speech(synthetic_audio_wav)

            assert len(intervals) == 2
            assert intervals[0] == (0.2, 1.1)
            assert intervals[1] == (1.3, 1.8)
