"""Pytest fixtures and synthetic media generation using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from narezka.config import AppConfig, load_config
from narezka.project import Project


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Temporary workspace directory for tests."""
    ws = tmp_path / "workspace_test"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def sample_config(tmp_workspace: Path) -> AppConfig:
    """Fixture providing a valid AppConfig instance pointing to temporary workspace."""
    cfg = load_config(base_dir=tmp_workspace)
    cfg.device = "cpu"  # unit tests run on CPU by default
    return cfg


@pytest.fixture
def synthetic_video_mp4(tmp_workspace: Path) -> Path:
    """Generate a 2-second synthetic MP4 video file with audio via ffmpeg."""
    out_file = tmp_workspace / "synthetic_test.mp4"
    if out_file.exists():
        return out_file

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # Fallback dummy file if ffmpeg is missing in test environment
        out_file.write_bytes(b"\x00" * 1024)
        return out_file

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=320x240:rate=25",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out_file),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_file


@pytest.fixture
def synthetic_audio_wav(tmp_workspace: Path) -> Path:
    """Generate a 2-second 16kHz mono WAV file via ffmpeg."""
    out_file = tmp_workspace / "synthetic_audio.wav"
    if out_file.exists():
        return out_file

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        out_file.write_bytes(b"\x00" * 1024)
        return out_file

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_file),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_file


@pytest.fixture
def test_project(synthetic_video_mp4: Path, sample_config: AppConfig, tmp_workspace: Path) -> Project:
    """Fixture providing an initialized Project instance with synthetic VOD."""
    projects_dir = tmp_workspace / "projects"
    return Project.create(
        vod_path=synthetic_video_mp4,
        config=sample_config,
        projects_base_dir=projects_dir,
    )
