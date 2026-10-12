"""Stage S4: Russian ASR, word alignment, hallucination filter, glossary and exports."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from narezka.asr.alignment import CTCAlignmentBackend
from narezka.asr.base import Segment, SpeechSegment
from narezka.asr.factory import get_asr_backend
from narezka.asr.postprocess import GlossaryCorrector, HallucinationFilter
from narezka.config import AppConfig
from narezka.pipeline.stage import Stage
from narezka.project import Project

logger = logging.getLogger(__name__)


def seconds_to_srt_time(seconds: float) -> str:
    """Format seconds float into SRT timestamp format: HH:MM:SS,mmm."""
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def seconds_to_display_time(seconds: float) -> str:
    """Format seconds float into display timestamp format: HH:MM:SS.s."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    tenths = int(round((seconds - total_seconds) * 10))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{tenths}"


def export_srt(segments: list[Segment], out_path: Path) -> None:
    """Export transcribed segments to standard .srt subtitle file."""
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{seconds_to_srt_time(seg.start)} --> {seconds_to_srt_time(seg.end)}")
        speaker_prefix = f"[{seg.speaker.upper()}] " if seg.speaker != "streamer" else ""
        lines.append(f"{speaker_prefix}{seg.text}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_txt(segments: list[Segment], out_path: Path) -> None:
    """Export readable transcript with timestamps to .txt file."""
    lines: list[str] = []
    for seg in segments:
        display_t = seconds_to_display_time(seg.start)
        lines.append(f"[{seg.id:04d} | {display_t}] {seg.speaker.upper()}: {seg.text}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


class S04AsrStage(Stage):
    """Transcribes speech segments into words and phrases with hallucination filter and glossary."""

    name = "s04_asr"
    code_version = "1.0.0"
    is_gpu_stage = True

    def compute_input_hash(self, project: Project) -> str:
        s03_segments = project.stages_dir / "s03_vad_lid" / "speech_segments.json"
        if s03_segments.is_file():
            return hashlib.sha256(s03_segments.read_bytes()).hexdigest()[:16]
        return project.metadata.vod.sha1_head

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "asr": config.asr.model_dump(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        asr_cfg = project.config.asr
        voice_wav = project.stages_dir / "s02_voice" / "voice.wav"
        if not voice_wav.is_file():
            voice_wav = project.stages_dir / "s01_audio" / "audio_track_0.wav"

        s03_json = project.stages_dir / "s03_vad_lid" / "speech_segments.json"
        if not s03_json.is_file():
            raise FileNotFoundError(f"Файл речевых сегментов не найден: {s03_json}")

        with open(s03_json, encoding="utf-8") as f:
            raw_segments_data = json.load(f)

        speech_segments = [SpeechSegment.model_validate(item) for item in raw_segments_data]

        # 1. Run ASR
        backend_name = asr_cfg.backend
        backend = get_asr_backend(backend_name)

        raw_segments = backend.transcribe(
            audio_path=voice_wav,
            segments=speech_segments,
            config=asr_cfg,
            device=project.config.device,
        )

        # 2. Alignment (ensure word timestamps)
        aligner = CTCAlignmentBackend()
        aligned_segments = aligner.align(voice_wav, raw_segments, device=project.config.device)

        # 3. Hallucination filter
        hallu_filter = HallucinationFilter(
            config=asr_cfg.hallucination_filter,
            blacklist_path=Path(asr_cfg.hallucination_filter.blacklist_file),
        )
        filtered_segments = hallu_filter.filter_segments(aligned_segments)

        # 4. Glossary substitution
        glossary_path = Path("plugins") / project.config.game / "glossary.yaml"
        if glossary_path.is_file():
            glossary = GlossaryCorrector(glossary_file=glossary_path)
            processed_segments = glossary.apply(filtered_segments)
        else:
            processed_segments = filtered_segments

        # Re-index segments
        for i, s in enumerate(processed_segments):
            s.id = i

        # 5. Export results
        out_json = output_dir / "transcript.json"
        out_srt = output_dir / "transcript.srt"
        out_txt = output_dir / "transcript.txt"

        json_data = [s.model_dump() for s in processed_segments]
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        export_srt(processed_segments, out_srt)
        export_txt(processed_segments, out_txt)

        total_words = sum(len(s.words) for s in processed_segments)
        logger.info(
            "Стадия S4 завершена: %d сегментов, %d слов распознано.",
            len(processed_segments),
            total_words,
        )

        return {
            "backend": backend_name,
            "segments_count": len(processed_segments),
            "words_count": total_words,
            "transcript_json": str(out_json),
            "transcript_srt": str(out_srt),
            "transcript_txt": str(out_txt),
        }
