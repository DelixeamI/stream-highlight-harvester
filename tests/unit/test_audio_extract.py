"""Unit tests for audio extraction from video files."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from narezka.media.audio_extract import extract_audio_stream


def test_extract_audio_stream_synthetic(synthetic_video_mp4: Path, tmp_path: Path) -> None:
    out_16k = tmp_path / "out_16k.wav"
    extracted = extract_audio_stream(
        vod_path=synthetic_video_mp4,
        output_wav=out_16k,
        stream_index=0,
        sample_rate=16000,
        channels=1,
    )

    assert extracted.is_file()
    assert extracted.stat().st_size > 0

    info = sf.info(str(extracted))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.duration > 0.0


def test_extract_audio_stream_stereo_44k(synthetic_video_mp4: Path, tmp_path: Path) -> None:
    out_44k = tmp_path / "out_44k.wav"
    extracted = extract_audio_stream(
        vod_path=synthetic_video_mp4,
        output_wav=out_44k,
        stream_index=0,
        sample_rate=44100,
        channels=2,
    )

    assert extracted.is_file()
    info = sf.info(str(extracted))
    assert info.samplerate == 44100
    assert info.channels == 2
