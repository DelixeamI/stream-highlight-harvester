"""Unit tests for Stage S06: Audio features (loudness, pitch, tagger, speech rate, lexicon)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from narezka.asr.base import Segment, Word
from narezka.audio.features import (
    compute_lexicon_features,
    compute_speech_rate,
)
from narezka.audio.loudness import (
    compute_robust_zscore,
    compute_short_term_loudness,
    extract_loudness_features,
)
from narezka.audio.pitch import compute_pitch_track
from narezka.audio.tagger import HeuristicAudioTagger


def test_robust_zscore_basic():
    # Array with constant baseline 1.0 and a spike at index 50
    x = np.ones(100, dtype=np.float32)
    x[50] = 10.0

    z = compute_robust_zscore(x, window_frames=21)
    assert len(z) == 100
    # Baseline should have z close to 0
    assert abs(z[10]) < 0.1
    # Spike should have high z
    assert z[50] > 5.0


def test_short_term_loudness_silence_and_signal():
    sr = 16000
    # 2 seconds of silence, then 2 seconds of sine wave
    t = np.linspace(0, 4, 4 * sr, dtype=np.float32)
    audio = np.zeros(4 * sr, dtype=np.float32)
    audio[2 * sr :] = 0.5 * np.sin(2 * np.pi * 440 * t[2 * sr :])

    loudness = compute_short_term_loudness(audio, sample_rate=sr, resolution_s=0.5, window_s=0.4)
    assert len(loudness) == 8  # 4 seconds / 0.5s = 8 frames
    # Silence frames should be clamped near -70 dB
    assert loudness[0] <= -60.0
    # Active frames should be significantly louder
    assert loudness[5] > -30.0


def test_loudness_feature_extraction_and_events():
    sr = 16000
    audio = np.random.normal(0, 0.01, 10 * sr).astype(np.float32)
    # Add a loud burst between 4s and 6s
    audio[4 * sr : 6 * sr] *= 15.0

    raw_loud, loud_z, events = extract_loudness_features(
        audio_path_or_data=audio,
        sample_rate=sr,
        resolution_s=0.5,
        rolling_window_s=5.0,
        cap=5.0,
    )
    assert len(loud_z) == 20
    assert np.max(loud_z) > 3.0
    # Should detect at least one loud_burst event
    assert len(events) >= 1
    assert events[0].type == "loud_burst"
    assert events[0].score > 0.5


def test_pitch_tracking():
    sr = 16000
    # 3 seconds of 220 Hz sine wave
    t = np.linspace(0, 3, 3 * sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 220 * t)

    f0_grid, voiced_mask = compute_pitch_track(audio, sample_rate=sr, resolution_s=0.5)
    assert len(f0_grid) == 6
    assert np.all(voiced_mask)
    # Detected pitch should be very close to 220 Hz
    assert abs(np.median(f0_grid) - 220.0) < 5.0


def test_heuristic_tagger():
    sr = 32000
    # 2 seconds of synthetic high-energy scream (burst + high freq)
    t = np.linspace(0, 2, 2 * sr, dtype=np.float32)
    audio = 0.4 * np.sin(2 * np.pi * 2500 * t)

    tagger = HeuristicAudioTagger()
    laugh_p, scream_p = tagger.predict_probs(audio, resolution_s=0.5)
    assert len(scream_p) == 4
    # High frequency loud tone should trigger scream probability
    assert np.max(scream_p) > 0.3


def test_speech_rate_sliding_window():
    # 10 second stream with 6 words placed between 4s and 6s
    words = [
        Word(text="раз", start=4.1, end=4.4),
        Word(text="два", start=4.5, end=4.8),
        Word(text="три", start=4.9, end=5.2),
        Word(text="четыре", start=5.3, end=5.6),
        Word(text="пять", start=5.7, end=6.0),
        Word(text="шесть", start=6.1, end=6.4),
    ]
    seg = Segment(id=1, start=4.0, end=6.5, text="раз два три четыре пять шесть", words=words)

    rate = compute_speech_rate([seg], duration_s=10.0, resolution_s=0.5, window_s=3.0)
    assert len(rate) == 20
    # Rate at 0s should be 0
    assert rate[0] == 0.0
    # Rate around 5.0s (frame 10) should be ~6 words / 3s = 2.0 words/sec
    assert rate[10] >= 1.5


def test_reaction_lexicon_matching(tmp_path: Path):
    lex_file = tmp_path / "test_lexicon.yaml"
    lex_file.write_text(
        """
groups:
  delight:
    weight: 0.85
    phrases:
      - вот это да
      - имба
  laughter:
    weight: 0.90
    phrases:
      - ахах
      - ору
""",
        encoding="utf-8",
    )

    segs = [
        Segment(id=1, start=2.0, end=4.0, text="Офигеть, вот это да, какая имба!"),
        Segment(id=2, start=6.0, end=8.0, text="Ору с этого момента, ахаха"),
    ]

    hype_lex, events = compute_lexicon_features(
        segments=segs,
        duration_s=10.0,
        lexicon_path=lex_file,
        resolution_s=0.5,
    )

    assert len(hype_lex) == 20
    # Matching segment 1
    assert hype_lex[5] >= 0.8
    # Matching segment 2
    assert hype_lex[13] >= 0.9
    assert len(events) == 2
    assert events[0].type == "hype_phrase"
