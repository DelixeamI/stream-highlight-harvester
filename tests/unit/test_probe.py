"""Unit tests for media probe utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from narezka.media.probe import parse_fps, probe_vod


def test_parse_fps() -> None:
    assert parse_fps("60/1") == 60.0
    assert parse_fps("30/1") == 30.0
    assert parse_fps("30000/1001") == 29.97002997002997
    assert parse_fps("59.94") == 59.94
    assert parse_fps("0/0") == 30.0
    assert parse_fps("") == 30.0


def test_probe_vod_synthetic(synthetic_video_mp4: Path) -> None:
    result = probe_vod(synthetic_video_mp4)
    assert result.duration_s > 0.0
    assert result.video.width > 0
    assert result.video.height > 0
    assert result.video.fps > 0.0
    assert len(result.audio_streams) >= 1
    assert result.file_size_bytes > 0


def test_probe_vod_missing_video_stream(tmp_path: Path) -> None:
    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_bytes(b"dummy")

    mock_output = '{"streams": [], "format": {"duration": "10.0"}}'
    with patch("shutil.which", return_value="/usr/bin/ffprobe"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output, returncode=0)
            with pytest.raises(ValueError, match="не обнаружено видеопотоков"):
                probe_vod(dummy_file)
