"""Stage S07: Computer Vision analysis of video stream (killfeed, HUD, multikills)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from narezka.config import AppConfig
from narezka.models import Event
from narezka.pipeline.stage import Stage
from narezka.project import Project
from narezka.vision.registry import get_vision_plugin
from narezka.vision.sampler import FrameSampler

logger = logging.getLogger(__name__)


class S07VisionStage(Stage):
    """Extract visual highlights, personal kills, multikills, and HUD events from video frames."""

    name = "s07_vision"
    code_version = "1.2.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        s00_meta = project.stages_dir / "s00_probe" / "meta.json"
        h0 = ""
        if s00_meta.is_file():
            try:
                with open(s00_meta, encoding="utf-8") as f:
                    h0 = json.load(f).get("cache_key", "")
            except Exception:
                pass
        vod_sha1 = project.metadata.vod.sha1_head or "default_vod"
        return hashlib.sha256(f"{vod_sha1}::{h0}".encode()).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "game": config.game,
            "vision": config.vision.model_dump(),
            "timeline_resolution": config.timeline.resolution_s,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        probe_json = project.stages_dir / "s00_probe" / "probe_result.json"
        duration_s = project.metadata.vod.duration_s
        if probe_json.is_file():
            try:
                with open(probe_json, encoding="utf-8") as f:
                    pdata = json.load(f)
                    duration_s = float(pdata.get("duration_s", duration_s))
            except Exception:
                pass

        game_name = project.config.game or "generic"
        plugin = get_vision_plugin(game_name)
        fps = project.config.vision.sampling.fps

        logger.info(
            "Running vision analysis for game '%s' (plugin: %s, fps: %.1f, duration: %.1fs)...",
            game_name,
            plugin.__class__.__name__,
            fps,
            duration_s,
        )

        events: list[Event] = []
        try:
            with FrameSampler(project.vod_path, fps=fps) as sampler:
                events = plugin.detect_events(
                    sampler=sampler,
                    duration_s=duration_s,
                    config=project.config,
                )
        except Exception as exc:
            logger.warning("Vision frame sampling failed (%s). Continuing with empty vision events.", exc)

        # Generate feature curves (e.g. continuous 1D 'kill' track)
        feature_curves = plugin.get_feature_curves(
            events=events,
            duration_s=duration_s,
            resolution_s=project.config.timeline.resolution_s,
        )

        # Save vision_events.json
        events_file = output_dir / "vision_events.json"
        with open(events_file, "w", encoding="utf-8") as f:
            json.dump(
                {"events": [e.model_dump() for e in events]},
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save vision_features.npz
        features_npz = output_dir / "vision_features.npz"
        np.savez_compressed(str(features_npz), **feature_curves)

        num_kills = sum(1 for e in events if e.type == "kill")
        num_multikills = sum(1 for e in events if e.type == "multikill")

        return {
            "game": game_name,
            "plugin": plugin.__class__.__name__,
            "total_events": len(events),
            "num_kills": num_kills,
            "num_multikills": num_multikills,
            "events_path": str(events_file),
            "features_path": str(features_npz),
        }
