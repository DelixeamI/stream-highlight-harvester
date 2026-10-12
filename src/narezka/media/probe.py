"""Media probing utilities using ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AudioStreamInfo(BaseModel):
    index: int
    codec: str
    channels: int
    sample_rate: int
    bitrate: int | None = None


class VideoStreamInfo(BaseModel):
    index: int
    codec: str
    width: int
    height: int
    fps: float
    duration_s: float
    keyframe_interval_s: float | None = None


class VodProbeResult(BaseModel):
    path: str
    duration_s: float
    file_size_bytes: int
    video: VideoStreamInfo
    audio_streams: list[AudioStreamInfo] = Field(default_factory=list)
    raw_format: dict[str, Any] = Field(default_factory=dict)


def parse_fps(fps_str: str) -> float:
    """Safely parse fractional fps string from ffprobe (e.g. '60/1', '59.94', '30000/1001')."""
    if not fps_str or fps_str == "0/0":
        return 30.0
    if "/" in fps_str:
        num, den = fps_str.split("/", 1)
        den_f = float(den)
        if den_f == 0.0:
            return 30.0
        return float(num) / den_f
    try:
        return float(fps_str)
    except ValueError:
        return 30.0


def estimate_keyframe_interval(vod_path: Path, max_scan_seconds: int = 60) -> float | None:
    """Estimate median keyframe interval in seconds over the first 60s of the video."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pict_type,pts_time",
        "-read_intervals",
        f"%+{max_scan_seconds}",
        "-of",
        "json",
        str(vod_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        frames = data.get("frames", [])
        keyframe_times = [
            float(f["pts_time"])
            for f in frames
            if f.get("pict_type") == "I" and "pts_time" in f
        ]
        if len(keyframe_times) < 2:
            return None

        diffs = [
            round(keyframe_times[i] - keyframe_times[i - 1], 3)
            for i in range(1, len(keyframe_times))
        ]
        diffs.sort()
        median_interval = diffs[len(diffs) // 2]
        return median_interval
    except Exception as exc:
        logger.debug("Failed to estimate keyframe interval: %s", exc)
        return None


def probe_vod(vod_path: Path) -> VodProbeResult:
    """Probe video and audio streams of VOD using ffprobe."""
    path = Path(vod_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл VOD не найден: {path}")

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe не найден в системном PATH. Установите FFmpeg.")

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr_sample = exc.stderr.strip()[-500:] if exc.stderr else "Неизвестная ошибка ffprobe"
        raise RuntimeError(f"Ошибка ffprobe при анализе файла {path.name}:\n{stderr_sample}") from exc

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Не удалось распарсить JSON вывод ffprobe: {exc}") from exc

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream: dict[str, Any] | None = None
    audio_streams: list[AudioStreamInfo] = []

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = s
        elif codec_type == "audio":
            audio_streams.append(
                AudioStreamInfo(
                    index=int(s.get("index", len(audio_streams))),
                    codec=s.get("codec_name", "unknown"),
                    channels=int(s.get("channels", 2)),
                    sample_rate=int(s.get("sample_rate", 44100)),
                    bitrate=int(s.get("bit_rate")) if s.get("bit_rate") else None,
                )
            )

    if not video_stream:
        raise ValueError(f"В файле {path.name} не обнаружено видеопотоков.")

    duration_s = float(
        format_info.get("duration")
        or video_stream.get("duration")
        or 0.0
    )
    fps = parse_fps(video_stream.get("r_frame_rate", "30/1"))
    keyframe_interval = estimate_keyframe_interval(path)

    video_info = VideoStreamInfo(
        index=int(video_stream.get("index", 0)),
        codec=video_stream.get("codec_name", "unknown"),
        width=int(video_stream.get("width", 1920)),
        height=int(video_stream.get("height", 1080)),
        fps=round(fps, 2),
        duration_s=round(duration_s, 2),
        keyframe_interval_s=keyframe_interval,
    )

    return VodProbeResult(
        path=str(path),
        duration_s=round(duration_s, 2),
        file_size_bytes=path.stat().st_size,
        video=video_info,
        audio_streams=audio_streams,
        raw_format=format_info,
    )
