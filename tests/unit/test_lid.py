"""Unit tests for Language Identification (LID)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from narezka.audio.lid import SpeechBrainLidBackend, WhisperLidBackend, get_lid_backend


def test_get_lid_backend() -> None:
    sb = get_lid_backend("speechbrain")
    assert isinstance(sb, SpeechBrainLidBackend)
    wh = get_lid_backend("whisper")
    assert isinstance(wh, WhisperLidBackend)


def test_speechbrain_lid_mocked(synthetic_audio_wav: Path) -> None:
    backend = SpeechBrainLidBackend()

    mock_classifier = MagicMock()
    # out_prob, score, index, text_lab
    mock_classifier.classify_batch.return_value = (
        None,
        torch.tensor([0.95]),
        None,
        ["ru: Russian"],
    )

    with patch.object(backend, "_load_model", return_value=mock_classifier):
        intervals = [(0.0, 2.0)]
        results = backend.identify_language(synthetic_audio_wav, intervals, min_segment_s=1.0)

        assert len(results) == 1
        assert results[0]["lang"] == "ru"
        assert results[0]["confidence"] == 0.95


def test_whisper_lid_short_segment_skips(synthetic_audio_wav: Path) -> None:
    backend = WhisperLidBackend()
    intervals = [(0.0, 0.5)]  # less than min_segment_s=1.5
    results = backend.identify_language(synthetic_audio_wav, intervals, min_segment_s=1.5)

    assert len(results) == 1
    assert results[0]["lang"] == "ru"
    assert results[0]["confidence"] == 1.0


def test_whisper_lid_mocked_long_segment(synthetic_audio_wav: Path) -> None:
    backend = WhisperLidBackend()
    mock_model = MagicMock()
    mock_model.detect_language.return_value = ("ru", 0.92, [("ru", 0.92)])

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        intervals = [(0.0, 2.0)]
        results = backend.identify_language(synthetic_audio_wav, intervals, min_segment_s=1.5)

        assert len(results) == 1
        assert results[0]["lang"] == "ru"
        assert results[0]["confidence"] == 0.92
