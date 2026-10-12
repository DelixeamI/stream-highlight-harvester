"""Stage S1: Audio extraction to 16 kHz mono and 44.1 kHz stereo WAV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from narezka.config import AppConfig
from narezka.media.audio_extract import extract_audio_stream
from narezka.pipeline.stage import Stage
from narezka.project import Project


class S01AudioExtractStage(Stage):
    """Extracts audio tracks from VOD."""

    name = "s01_audio"
    code_version = "1.0.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        return project.metadata.vod.sha1_head

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {"tracks": config.audio.tracks}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        tracks_cfg = project.config.audio.tracks
        extracted_files: dict[str, str] = {}

        # Default track 0 extraction (mix)
        track_0_16k = output_dir / "audio_track_0.wav"
        extract_audio_stream(
            vod_path=project.vod_path,
            output_wav=track_0_16k,
            stream_index=0,
            sample_rate=16000,
            channels=1,
        )
        extracted_files["track_0_16k"] = str(track_0_16k)

        # For variant A (single mix), extract 44.1 kHz stereo for high-fidelity source separation
        track_0_44k = output_dir / "audio_mix_44k.wav"
        extract_audio_stream(
            vod_path=project.vod_path,
            output_wav=track_0_44k,
            stream_index=0,
            sample_rate=44100,
            channels=2,
        )
        extracted_files["track_0_44k"] = str(track_0_44k)

        # If custom tracks mapping provided (e.g. OBS multi-track)
        if isinstance(tracks_cfg, dict):
            for name, idx in tracks_cfg.items():
                track_file = output_dir / f"audio_{name}.wav"
                extract_audio_stream(
                    vod_path=project.vod_path,
                    output_wav=track_file,
                    stream_index=idx,
                    sample_rate=16000,
                    channels=1,
                )
                extracted_files[f"track_{name}"] = str(track_file)

        return extracted_files
