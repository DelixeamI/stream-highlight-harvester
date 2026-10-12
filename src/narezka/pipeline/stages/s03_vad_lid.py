"""Stage S3: Voice Activity Detection (VAD) and Language Identification (LID)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from narezka.asr.base import SpeechSegment
from narezka.audio.lid import get_lid_backend
from narezka.audio.vad import get_vad_backend
from narezka.config import AppConfig
from narezka.pipeline.stage import Stage
from narezka.project import Project

logger = logging.getLogger(__name__)


class S03VadLidStage(Stage):
    """Detects speech intervals and identifies language (filtering English game voice lines)."""

    name = "s03_vad_lid"
    code_version = "1.0.0"
    is_gpu_stage = True

    def compute_input_hash(self, project: Project) -> str:
        s02_voice = project.stages_dir / "s02_voice" / "voice.wav"
        if s02_voice.is_file():
            stat = s02_voice.stat()
            return hashlib.sha256(f"{stat.st_size}_{stat.st_mtime}".encode("ascii")).hexdigest()[:16]
        return project.metadata.vod.sha1_head

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "vad": config.audio.vad.model_dump(),
            "lid": config.audio.lid.model_dump(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        vad_cfg = project.config.audio.vad
        lid_cfg = project.config.audio.lid

        voice_wav = project.stages_dir / "s02_voice" / "voice.wav"
        if not voice_wav.is_file():
            # Fallback to s01_audio
            voice_wav = project.stages_dir / "s01_audio" / "audio_track_0.wav"

        # 1. Voice Activity Detection
        vad = get_vad_backend("silero")
        raw_intervals = vad.detect_speech(
            audio_path=voice_wav,
            threshold=vad_cfg.threshold,
            min_speech_ms=vad_cfg.min_speech_ms,
            min_silence_ms=vad_cfg.min_silence_ms,
            pad_ms=vad_cfg.pad_ms,
            device=project.config.device,
        )

        # 2. Language Identification
        speech_segments: list[SpeechSegment] = []

        if lid_cfg.enabled and raw_intervals:
            lid_backend = get_lid_backend(lid_cfg.backend)
            lid_results = lid_backend.identify_language(
                audio_path=voice_wav,
                intervals=raw_intervals,
                min_segment_s=lid_cfg.min_segment_s,
                device=project.config.device,
            )

            for idx, item in enumerate(lid_results):
                lang = item.get("lang", "ru")
                conf = item.get("confidence", 1.0)
                # Check if it's a non-Russian game voice line
                is_game_line = False
                if lang != "ru" and conf >= lid_cfg.non_ru_confidence:
                    is_game_line = True

                speech_segments.append(
                    SpeechSegment(
                        id=idx,
                        start=round(item["start"], 3),
                        end=round(item["end"], 3),
                        lang=lang,
                        lang_confidence=round(conf, 3),
                        speaker="game" if is_game_line else "streamer",
                        source_track="voice_sep",
                        is_game_voice_line=is_game_line,
                    )
                )
        else:
            for idx, (st, en) in enumerate(raw_intervals):
                speech_segments.append(
                    SpeechSegment(
                        id=idx,
                        start=st,
                        end=en,
                        lang="ru",
                        lang_confidence=1.0,
                        speaker="streamer",
                        source_track="voice_sep",
                        is_game_voice_line=False,
                    )
                )

        # Save speech_segments.json
        out_json = output_dir / "speech_segments.json"
        data = [s.model_dump() for s in speech_segments]
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        game_lines_count = sum(1 for s in speech_segments if s.is_game_voice_line)
        logger.info(
            "Стадия S3: сформировано %d речевых сегментов (%d отфильтровано как реплики игры/не-RU)",
            len(speech_segments),
            game_lines_count,
        )

        return {
            "total_segments": len(speech_segments),
            "game_voice_lines": game_lines_count,
            "segments_file": str(out_json),
        }
