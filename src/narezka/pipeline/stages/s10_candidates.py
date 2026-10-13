"""Stage S10: Timeline fusion, integral interest curve S(t), and candidate generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from narezka.config import AppConfig
from narezka.fusion.candidates import generate_candidates
from narezka.fusion.timeline import build_unified_timeline
from narezka.models import Event
from narezka.pipeline.stage import Stage
from narezka.project import Project


class S10CandidatesStage(Stage):
    """Fuse all signals onto unified timeline and generate candidate highlight windows."""

    name = "s10_candidates"
    code_version = "1.0.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        s06_meta = project.stages_dir / "s06_audio_features" / "meta.json"
        s07_meta = project.stages_dir / "s07_vision" / "meta.json"
        s08_meta = project.stages_dir / "s08_chat_features" / "meta.json"
        s09_meta = project.stages_dir / "s09_semantic_scan" / "meta.json"
        h6 = ""
        h7 = ""
        h8 = ""
        h9 = ""
        if s06_meta.is_file():
            try:
                with open(s06_meta, encoding="utf-8") as f:
                    h6 = json.load(f).get("cache_key", "")
            except Exception:
                pass
        if s07_meta.is_file():
            try:
                with open(s07_meta, encoding="utf-8") as f:
                    h7 = json.load(f).get("cache_key", "")
            except Exception:
                pass
        if s08_meta.is_file():
            try:
                with open(s08_meta, encoding="utf-8") as f:
                    h8 = json.load(f).get("cache_key", "")
            except Exception:
                pass
        if s09_meta.is_file():
            try:
                with open(s09_meta, encoding="utf-8") as f:
                    h9 = json.load(f).get("cache_key", "")
            except Exception:
                pass

        raw = f"{h6}::{h7}::{h8}::{h9}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "timeline": config.timeline.model_dump(),
            "features": config.features.model_dump(),
            "output": config.output.model_dump(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        # 1. Load Audio features and events from S06
        s06_dir = project.stages_dir / "s06_audio_features"
        s06_npz = s06_dir / "audio_features.npz"
        s06_events_file = s06_dir / "audio_events.json"

        features_dict: dict[str, np.ndarray] = {}
        events: list[Event] = []
        duration_s = 0.0

        if s06_npz.is_file():
            data = np.load(str(s06_npz))
            timestamps = data["timestamps"]
            duration_s = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
            for k in ["loud_z", "pitch_z", "laugh_p", "scream_p", "speech_rate", "hype_lex"]:
                if k in data:
                    features_dict[k] = data[k]

        if s06_events_file.is_file():
            with open(s06_events_file, encoding="utf-8") as f:
                evt_data = json.load(f)
                events.extend([Event.model_validate(e) for e in evt_data.get("events", [])])

        # 2. Load Vision features and events from S07 (if present)
        s07_dir = project.stages_dir / "s07_vision"
        s07_npz = s07_dir / "vision_features.npz"
        s07_events_file = s07_dir / "vision_events.json"

        if s07_npz.is_file():
            vdata = np.load(str(s07_npz))
            for k in ["kill", "visual_action"]:
                if k in vdata:
                    features_dict[k] = vdata[k]

        if s07_events_file.is_file():
            try:
                with open(s07_events_file, encoding="utf-8") as f:
                    vevt_data = json.load(f)
                    events.extend([Event.model_validate(e) for e in vevt_data.get("events", [])])
            except Exception:
                pass

        # 3. Load Chat features and events from S08 (if present)
        s08_dir = project.stages_dir / "s08_chat_features"
        s08_npz = s08_dir / "chat_features.npz"
        s08_events_file = s08_dir / "chat_events.json"

        if s08_npz.is_file():
            cdata = np.load(str(s08_npz))
            for k in ["chat_rate_z", "chat_laugh", "chat_hype"]:
                if k in cdata:
                    features_dict[k] = cdata[k]

        if s08_events_file.is_file():
            with open(s08_events_file, encoding="utf-8") as f:
                cevt_data = json.load(f)
                events.extend([Event.model_validate(e) for e in cevt_data.get("events", [])])

        # 3. Load Semantic events from S09 (if present)
        s09_events_file = project.stages_dir / "s09_semantic_scan" / "semantic_events.json"
        if s09_events_file.is_file():
            try:
                with open(s09_events_file, encoding="utf-8") as f:
                    sem_data = json.load(f)
                    events.extend([Event.model_validate(e) for e in sem_data.get("events", [])])
            except Exception:
                pass

        if duration_s <= 0.0:
            probe_json = project.stages_dir / "s00_probe" / "probe_result.json"
            if probe_json.is_file():
                with open(probe_json, encoding="utf-8") as f:
                    duration_s = float(json.load(f).get("duration_s", 0.0))

        # 3. Build unified timeline and compute S(t)
        timeline = build_unified_timeline(
            features_dict=features_dict,
            duration_s=duration_s,
            timeline_cfg=project.config.timeline,
            features_cfg=project.config.features,
        )

        # 4. Generate candidate intervals
        candidates = generate_candidates(
            timeline=timeline,
            events=events,
            config=project.config,
            duration_s=duration_s,
        )

        # 5. Save timeline NPZ
        npz_path = output_dir / "timeline.npz"
        save_dict = {
            "timestamps": timeline.timestamps,
            "s_raw": timeline.s_raw,
            "s_smooth": timeline.s_smooth,
        }
        save_dict.update(timeline.features)
        np.savez_compressed(str(npz_path), **save_dict)

        # 6. Save candidates JSONL
        candidates_path = output_dir / "candidates.jsonl"
        with open(candidates_path, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(c.model_dump_json() + "\n")

        # 7. Save all events JSON
        events_path = output_dir / "events.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump({"events": [e.model_dump() for e in events]}, f, indent=2, ensure_ascii=False)

        return {
            "duration_s": round(duration_s, 2),
            "num_candidates": len(candidates),
            "max_s_smooth": float(np.max(timeline.s_smooth)) if len(timeline.s_smooth) > 0 else 0.0,
            "num_events": len(events),
            "timeline_path": str(npz_path),
            "candidates_path": str(candidates_path),
        }
