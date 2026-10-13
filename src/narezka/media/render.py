"""Hardware NVENC and stream-copy video clip rendering using FFmpeg."""

from __future__ import annotations

import concurrent.futures
import logging
import re
import shutil
import subprocess
from pathlib import Path

from narezka.config import RenderConfig
from narezka.models import Moment

logger = logging.getLogger(__name__)

TRANSLIT_TABLE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify_title(text: str) -> str:
    """Convert Russian or English text into safe alphanumeric slug for filenames."""
    text = text.lower()
    res = []
    for ch in text:
        if ch in TRANSLIT_TABLE:
            res.append(TRANSLIT_TABLE[ch])
        elif ch.isalnum():
            res.append(ch)
        else:
            res.append("-")
    slug = "".join(res)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:40] if slug else "clip"


def format_clip_filename(index: int, moment: Moment) -> str:
    """Generate standardized filename: {NNN}_{HHhMMmSSs}_{category}_{slug(title)}.mp4."""
    total_sec = int(moment.start)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    time_part = f"{h:02d}h{m:02d}m{s:02d}s"
    slug = slugify_title(moment.title)
    return f"{index:03d}_{time_part}_{moment.category}_{slug}.mp4"


def render_single_clip(
    vod_path: Path,
    moment: Moment,
    output_path: Path,
    render_cfg: RenderConfig,
    fast: bool = False,
) -> Path:
    """Render a single clip from VOD using FFmpeg NVENC or stream copy."""
    ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_str = f"{moment.start:.3f}"
    duration_str = f"{moment.end - moment.start:.3f}"

    if fast:
        # Fast mode: stream copy without re-encoding (cut on keyframes)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", start_str,
            "-t", duration_str,
            "-i", str(vod_path),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]
    else:
        # Frame-accurate NVENC or libx264 re-encoding
        codec = render_cfg.codec
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", start_str,
            "-t", duration_str,
            "-i", str(vod_path),
            "-c:v", codec,
        ]

        if "nvenc" in codec:
            cmd.extend([
                "-preset", render_cfg.preset,
                "-cq", str(render_cfg.cq),
            ])
        else:
            cmd.extend([
                "-preset", "veryfast",
                "-crf", "22",
            ])

        cmd.extend([
            "-c:a", "aac",
            "-b:a", render_cfg.audio_bitrate,
            "-pix_fmt", "yuv420p",
            str(output_path),
        ])

    logger.debug("Executing ffmpeg: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if "nvenc" in render_cfg.codec and not fast:
            logger.warning("NVENC encoding failed (%s), falling back to libx264", res.stderr[-200:])
            # Fallback to libx264
            cmd_fallback = [
                ffmpeg_exe,
                "-y",
                "-ss", start_str,
                "-t", duration_str,
                "-i", str(vod_path),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", render_cfg.audio_bitrate,
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]
            res_fb = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if res_fb.returncode != 0:
                raise RuntimeError(f"FFmpeg fallback failed: {res_fb.stderr[-300:]}")
        else:
            raise RuntimeError(f"FFmpeg failed with code {res.returncode}: {res.stderr[-300:]}")

    logger.info("Rendered clip %s (%.1fs)", output_path.name, moment.end - moment.start)
    return output_path


def render_clips_batch(
    vod_path: Path,
    moments: list[Moment],
    clips_dir: Path,
    render_cfg: RenderConfig,
    fast: bool = False,
    selected_ids: set[str] | None = None,
    max_workers: int | None = None,
) -> list[Moment]:
    """Render batch of moments in parallel up to max_workers NVENC sessions."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    workers = max_workers or render_cfg.max_parallel

    target_moments = moments
    if selected_ids:
        target_moments = [m for m in moments if m.id in selected_ids]

    tasks: list[tuple[int, Moment, Path]] = []
    for idx, m in enumerate(target_moments, start=1):
        fname = format_clip_filename(idx, m)
        out_p = clips_dir / fname
        tasks.append((idx, m, out_p))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_moment = {
            executor.submit(
                render_single_clip,
                vod_path=vod_path,
                moment=m,
                output_path=out_p,
                render_cfg=render_cfg,
                fast=fast,
            ): (m, out_p)
            for _, m, out_p in tasks
        }

        for future in concurrent.futures.as_completed(future_to_moment):
            m, out_p = future_to_moment[future]
            try:
                future.result()
                # Update relative clip path in moment
                rel_path = f"clips/{out_p.name}"
                m.clip_path = rel_path
            except Exception as e:
                logger.error("Failed to render moment %s: %s", m.id, e)

    return moments
