"""Unit tests for diagnostic checks in narezka doctor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from narezka.config import AppConfig
from narezka.doctor import (
    check_api_keys,
    check_disk_space,
    check_ffmpeg,
    check_nvenc,
    check_python,
    check_torch_cuda,
)


def test_check_python_passes() -> None:
    res = check_python()
    assert res.status == "PASS"
    assert "Python" in res.details


def test_check_disk_space(tmp_path: Path) -> None:
    res = check_disk_space(tmp_path)
    assert res.status in ("PASS", "WARN")
    assert "GB" in res.details


def test_check_api_keys_local_profile() -> None:
    cfg = AppConfig()
    cfg.judge.profile = "local"
    res = check_api_keys(cfg)
    assert res.status == "PASS"
    assert "локальный профиль" in res.details


def test_check_api_keys_cloud_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig()
    cfg.judge.profile = "cloud"

    # Missing key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res_missing = check_api_keys(cfg)
    assert res_missing.status == "WARN"

    # Key provided
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_123456789")
    res_present = check_api_keys(cfg)
    assert res_present.status == "PASS"


def test_check_ffmpeg_mocked() -> None:
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="ffmpeg version 7.1.1\n", returncode=0)
            res = check_ffmpeg()
            assert res.status == "PASS"
            assert "ffmpeg version 7.1.1" in res.details


def test_check_nvenc_mocked() -> None:
    with patch("shutil.which", return_value="ffmpeg"):
        with patch("subprocess.run") as mock_run:
            # First call for -encoders, second for test encoding
            mock_run.side_effect = [
                MagicMock(stdout=" V..... h264_nvenc \n V..... hevc_nvenc \n", returncode=0),
                MagicMock(returncode=0),
            ]
            res = check_nvenc()
            assert res.status == "PASS"
            assert "h264_nvenc" in res.details


def test_check_torch_cuda_unavailable() -> None:
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.7.0"
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        res = check_torch_cuda()
        assert res.status == "FAIL"
        assert "недоступна" in res.details
