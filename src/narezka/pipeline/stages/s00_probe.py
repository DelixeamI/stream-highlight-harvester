"""Stage S0: Media probe using ffprobe."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from narezka.config import AppConfig
from narezka.media.probe import probe_vod
from narezka.pipeline.stage import Stage
from narezka.project import Project


class S00ProbeStage(Stage):
    """Probes VOD streams, duration, fps, resolution, codecs, and keyframe intervals."""

    name = "s00_probe"
    code_version = "1.0.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        # Input is the VOD file itself, identified quickly by sha1_head
        return project.metadata.vod.sha1_head

    def compute_params_hash(self, config: AppConfig) -> str:
        # No specific parameters affect ffprobe
        return hashlib.sha256(b"s00_probe_default_params").hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        probe_result = probe_vod(project.vod_path)

        out_json = output_dir / "probe_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            f.write(probe_result.model_dump_json(indent=2))

        # Update project.json with probed metadata
        meta = project.metadata
        meta.vod.duration_s = probe_result.duration_s
        meta.vod.fps = probe_result.video.fps
        meta.vod.width = probe_result.video.width
        meta.vod.height = probe_result.video.height
        meta.vod.video_codec = probe_result.video.codec
        meta.vod.audio_streams_count = len(probe_result.audio_streams)
        project.save_metadata(meta)

        return {
            "duration_s": probe_result.duration_s,
            "fps": probe_result.video.fps,
            "width": probe_result.video.width,
            "height": probe_result.video.height,
            "audio_streams": len(probe_result.audio_streams),
            "probe_json": str(out_json),
        }
