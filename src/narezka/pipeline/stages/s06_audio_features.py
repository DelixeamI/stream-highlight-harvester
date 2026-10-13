"""Stage S6: Audio feature extraction (loudness, pitch, laughter, scream, speech rate, lexicon)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from narezka.asr.base import Segment
from narezka.audio.features import extract_all_audio_features
from narezka.config import AppConfig
from narezka.pipeline.stage import Stage
from narezka.project import Project


class S06AudioFeaturesStage(Stage):
    """Extract audio features and events on unified timeline grid."""

    name = "s06_audio_features"
    code_version = "1.0.0"
    is_gpu_stage = True

    def compute_input_hash(self, project: Project) -> str:
        s04_meta = project.stages_dir / "s04_asr" / "meta.json"
        s04_hash = ""
        if s04_meta.is_file():
            try:
                with open(s04_meta, encoding="utf-8") as f:
                    s04_hash = json.load(f).get("cache_key", "")
            except Exception:
                pass

        voice_audio = project.stages_dir / "s02_voice" / "voice.wav"
        if not voice_audio.is_file():
            voice_audio = project.stages_dir / "s02_voice_isolation" / "voice.wav"
        if not voice_audio.is_file():
            voice_audio = project.stages_dir / "s01_audio" / "audio_track_0.wav"
        if not voice_audio.is_file():
            voice_audio = project.stages_dir / "s01_audio_extract" / "audio_16k.wav"

        audio_stat = ""
        if voice_audio.is_file():
            st = voice_audio.stat()
            audio_stat = f"{st.st_size}_{st.st_mtime}"

        raw = f"{s04_hash}::{audio_stat}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "timeline": config.timeline.model_dump(),
            "features": config.features.model_dump(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        # 1. Determine input voice track
        voice_path = project.stages_dir / "s02_voice" / "voice.wav"
        if not voice_path.is_file():
            voice_path = project.stages_dir / "s02_voice_isolation" / "voice.wav"
        if not voice_path.is_file():
            voice_path = project.stages_dir / "s01_audio" / "audio_track_0.wav"
        if not voice_path.is_file():
            voice_path = project.stages_dir / "s01_audio_extract" / "audio_16k.wav"
        if not voice_path.is_file():
            raise FileNotFoundError(f"No audio file found for features in {project.project_dir}")

        # 2. Load transcript segments from s04
        transcript_path = project.stages_dir / "s04_asr" / "transcript.json"
        segments: list[Segment] = []
        if transcript_path.is_file():
            with open(transcript_path, encoding="utf-8") as f:
                data = json.load(f)
                raw_segs = data if isinstance(data, list) else data.get("segments", [])
                segments = [Segment.model_validate(s) for s in raw_segs]

        # 3. Path to Russian lexicon
        lexicon_path = Path("plugins/common/lexicon_ru.yaml")
        if not lexicon_path.is_file():
            lexicon_path = Path(__file__).resolve().parents[4] / "plugins" / "common" / "lexicon_ru.yaml"

        # 4. Extract all features
        res = extract_all_audio_features(
            audio_path=voice_path,
            segments=segments,
            timeline_cfg=project.config.timeline,
            features_cfg=project.config.features,
            lexicon_path=lexicon_path,
            device=project.config.device,
        )

        # 5. Save features NPZ
        npz_path = output_dir / "audio_features.npz"
        np.savez_compressed(
            str(npz_path),
            timestamps=res.timestamps,
            loud_raw=res.loud_raw,
            loud_z=res.loud_z,
            f0_raw=res.f0_raw,
            pitch_z=res.pitch_z,
            laugh_p=res.laugh_p,
            scream_p=res.scream_p,
            speech_rate=res.speech_rate,
            hype_lex=res.hype_lex,
        )

        # 6. Save events JSON
        events_path = output_dir / "audio_events.json"
        events_dict = [e.model_dump() for e in res.events]
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump({"events": events_dict}, f, indent=2, ensure_ascii=False)

        return {
            "num_frames": len(res.timestamps),
            "duration_s": float(res.timestamps[-1]) if len(res.timestamps) > 0 else 0.0,
            "num_events": len(res.events),
            "max_loud_z": float(np.max(res.loud_z)) if len(res.loud_z) > 0 else 0.0,
            "max_laugh_p": float(np.max(res.laugh_p)) if len(res.laugh_p) > 0 else 0.0,
            "max_scream_p": float(np.max(res.scream_p)) if len(res.scream_p) > 0 else 0.0,
            "features_path": str(npz_path),
            "events_path": str(events_path),
        }
