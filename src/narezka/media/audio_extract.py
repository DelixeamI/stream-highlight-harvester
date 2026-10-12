"""Audio extraction from video files using ffmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio_stream(
    vod_path: Path,
    output_wav: Path,
    stream_index: int = 0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Extract a specific audio stream from VOD to WAV format via ffmpeg.

    Args:
        vod_path: Path to the input video file.
        output_wav: Destination path for the extracted WAV file.
        stream_index: Relative index of the audio stream (e.g. 0 for first audio track).
        sample_rate: Target sample rate (16000 for ASR/VAD, 44100 for separation).
        channels: Target channels (1 for mono, 2 for stereo).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg не найден в системном PATH.")

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(vod_path.resolve()),
        "-map",
        f"0:a:{stream_index}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_wav.resolve()),
    ]

    logger.debug("Running ffmpeg extract: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr_tail = "\n".join(exc.stderr.strip().splitlines()[-10:]) if exc.stderr else ""
        raise RuntimeError(
            f"Ошибка ffmpeg при извлечении аудио (дорожка {stream_index}) из {vod_path.name}:\n{stderr_tail}"
        ) from exc

    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"Файл аудио {output_wav.name} не был создан или имеет нулевой размер.")

    return output_wav
