"""Stage S2: Voice and accompaniment separation."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from narezka.audio.separation import get_separation_backend
from narezka.config import AppConfig
from narezka.pipeline.stage import Stage
from narezka.project import Project

logger = logging.getLogger(__name__)


class S02VoiceIsolationStage(Stage):
    """Isolates streamer voice and accompaniment from audio mix."""

    name = "s02_voice"
    code_version = "1.0.0"
    is_gpu_stage = True

    def compute_input_hash(self, project: Project) -> str:
        s01_dir = project.stages_dir / "s01_audio"
        s01_meta = s01_dir / "meta.json"
        if s01_meta.is_file():
            return hashlib.sha256(s01_meta.read_bytes()).hexdigest()[:16]
        return project.metadata.vod.sha1_head

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "separation": config.audio.separation.model_dump(),
            "tracks": config.audio.tracks,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        tracks_cfg = project.config.audio.tracks
        sep_cfg = project.config.audio.separation
        s01_dir = project.stages_dir / "s01_audio"

        output_voice = output_dir / "voice.wav"
        output_accompaniment = output_dir / "accompaniment.wav"

        # Check if separation is explicitly disabled
        if sep_cfg.enabled is False or str(sep_cfg.enabled).lower() in ("false", "off", "no"):
            logger.info("Нейросетевое разделение источников отключено в конфигурации. Используется исходный трек.")
            track_0 = s01_dir / "audio_track_0.wav"
            if track_0.is_file():
                shutil.copy2(track_0, output_voice)
                shutil.copy2(track_0, output_accompaniment)
            return {
                "mode": "disabled",
                "voice_wav": str(output_voice),
                "accompaniment_wav": str(output_accompaniment),
            }

        # Check track variants:
        # Variant B/C: Dedicated mic track is provided in tracks mapping
        if isinstance(tracks_cfg, dict) and "mic" in tracks_cfg:
            mic_file = s01_dir / "audio_mic.wav"
            if mic_file.is_file():
                logger.info("Обнаружена отдельная дорожка микрофона OBS. Пропуск нейросетевого разделения (Вариант B/C).")
                shutil.copy2(mic_file, output_voice)
                # Accompaniment is game track if present or track 0
                game_file = s01_dir / "audio_game.wav"
                if game_file.is_file():
                    shutil.copy2(game_file, output_accompaniment)
                else:
                    track_0 = s01_dir / "audio_track_0.wav"
                    if track_0.is_file():
                        shutil.copy2(track_0, output_accompaniment)

                return {
                    "mode": "dedicated_mic_track",
                    "voice_wav": str(output_voice),
                    "accompaniment_wav": str(output_accompaniment),
                }

        # Variant A: Single mixed track -> neural source separation
        input_mix_44k = s01_dir / "audio_mix_44k.wav"
        if not input_mix_44k.is_file():
            input_mix_44k = s01_dir / "audio_track_0.wav"

        backend_name = sep_cfg.backend
        backend = get_separation_backend(backend_name, model_id=sep_cfg.model)

        backend.separate(
            input_audio=input_mix_44k,
            output_voice=output_voice,
            output_accompaniment=output_accompaniment,
            chunk_seconds=sep_cfg.chunk_s,
            device=project.config.device,
        )

        return {
            "mode": "neural_separation",
            "backend": backend_name,
            "voice_wav": str(output_voice),
            "accompaniment_wav": str(output_accompaniment),
        }
