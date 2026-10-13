"""Stage S8: Twitch chat feature extraction, delay alignment and event detection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from narezka.chat.features import extract_and_align_chat
from narezka.chat.twitch_downloader import TwitchDownloaderJSON
from narezka.config import AppConfig
from narezka.pipeline.stage import Stage
from narezka.project import Project


class S08ChatFeaturesStage(Stage):
    """Extract and align Twitch chat features and events."""

    name = "s08_chat_features"
    code_version = "1.0.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        s06_meta = project.stages_dir / "s06_audio_features" / "meta.json"
        s06_hash = ""
        if s06_meta.is_file():
            try:
                with open(s06_meta, encoding="utf-8") as f:
                    s06_hash = json.load(f).get("cache_key", "")
            except Exception:
                pass

        chat_stat = "none"
        chat_path = None
        if project.config.chat.path:
            p = Path(project.config.chat.path)
            if p.is_file():
                chat_path = p
        if chat_path is None:
            candidate_p = project.project_dir / "chat.json"
            if candidate_p.is_file():
                chat_path = candidate_p

        if chat_path and chat_path.is_file():
            st = chat_path.stat()
            chat_stat = f"{st.st_size}_{st.st_mtime}"

        raw = f"{s06_hash}::{chat_stat}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "chat": config.chat.model_dump(),
            "timeline": config.timeline.model_dump(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        # 1. Determine timeline duration and audio excitation from S06
        s06_npz = project.stages_dir / "s06_audio_features" / "audio_features.npz"
        duration_s = 0.0
        audio_excitation: np.ndarray | None = None

        if s06_npz.is_file():
            data = np.load(str(s06_npz))
            timestamps = data["timestamps"]
            duration_s = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
            loud_z = data["loud_z"]
            laugh_p = data["laugh_p"]
            audio_excitation = loud_z + laugh_p
        else:
            # Fallback to probe info if S06 not run
            probe_json = project.stages_dir / "s00_probe" / "probe_result.json"
            if probe_json.is_file():
                with open(probe_json, encoding="utf-8") as f:
                    duration_s = float(json.load(f).get("duration_s", 0.0))

        # 2. Check chat file existence
        chat_path = None
        if project.config.chat.path:
            p = Path(project.config.chat.path)
            if p.is_file():
                chat_path = p
        if chat_path is None:
            candidate_p = project.project_dir / "chat.json"
            if candidate_p.is_file():
                chat_path = candidate_p

        markers_path = Path("plugins/common/chat_markers.yaml")
        if not markers_path.is_file():
            markers_path = Path(__file__).resolve().parents[4] / "plugins" / "common" / "chat_markers.yaml"

        messages = []
        if chat_path and chat_path.is_file():
            parser = TwitchDownloaderJSON()
            messages = parser.load_messages(chat_path)

        res = extract_and_align_chat(
            messages=messages,
            duration_s=duration_s,
            audio_excitation=audio_excitation,
            chat_cfg=project.config.chat,
            features_cfg=project.config.features,
            timeline_cfg=project.config.timeline,
            markers_path=markers_path,
        )

        # 3. Save chat features NPZ
        npz_path = output_dir / "chat_features.npz"
        np.savez_compressed(
            str(npz_path),
            timestamps=res.timestamps,
            chat_rate=res.chat_rate,
            chat_rate_z=res.chat_rate_z,
            chat_laugh=res.chat_laugh,
            chat_hype=res.chat_hype,
            estimated_delay_s=res.estimated_delay_s,
        )

        # 4. Save chat events JSON
        events_path = output_dir / "chat_events.json"
        events_dict = [e.model_dump() for e in res.events]
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "estimated_delay_s": res.estimated_delay_s,
                    "num_messages": len(messages),
                    "events": events_dict,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "chat_file_present": chat_path is not None,
            "num_messages": len(messages),
            "estimated_delay_s": round(res.estimated_delay_s, 2),
            "num_events": len(res.events),
            "max_chat_rate_z": float(np.max(res.chat_rate_z)) if len(res.chat_rate_z) > 0 else 0.0,
            "max_chat_laugh": float(np.max(res.chat_laugh)) if len(res.chat_laugh) > 0 else 0.0,
            "features_path": str(npz_path),
            "events_path": str(events_path),
        }
