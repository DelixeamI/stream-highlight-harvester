"""GigaAM (Sber) ASR backend implementation."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from narezka.asr.alignment import interpolate_word_timestamps
from narezka.asr.base import ASRBackend, Segment, SpeechSegment
from narezka.config import AsrConfig
from narezka.utils.gpu import cleanup_vram

logger = logging.getLogger(__name__)


class GigaAMBackend(ASRBackend):
    """ASR backend using Sber's GigaAM models."""

    name = "gigaam"

    def transcribe(
        self,
        audio_path: Path,
        segments: list[SpeechSegment],
        config: AsrConfig,
        device: str = "cuda",
    ) -> list[Segment]:
        # VERIFY: gigaam.load_model and model.transcribe on Windows sm_120
        import gigaam

        target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        model_name = config.gigaam.model
        # Map generic v3_e2e / v3 to available models if needed
        if model_name in ("v3_e2e", "v3", "default"):
            model_name = "v2_ctc"

        logger.info("Загрузка модели GigaAM '%s' на устройство %s...", model_name, target_device)
        model = gigaam.load_model(
            model_name=model_name,
            device=target_device,
            fp16_encoder=(target_device == "cuda"),
        )

        output_segments: list[Segment] = []

        try:
            with sf.SoundFile(str(audio_path)) as sfile:
                sr = sfile.samplerate

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_chunk_path = Path(tmp_dir) / "chunk.wav"

                    for seg in segments:
                        if seg.is_game_voice_line or seg.lang != "ru":
                            continue

                        start_sample = int(seg.start * sr)
                        num_samples = int((seg.end - seg.start) * sr)
                        sfile.seek(start_sample)
                        chunk = sfile.read(num_samples, dtype="float32")

                        if chunk.ndim > 1:
                            chunk = chunk.mean(axis=1)

                        # Write small temporary chunk for GigaAM
                        sf.write(str(tmp_chunk_path), chunk, sr, subtype="PCM_16")

                        try:
                            model_any: Any = model
                            transcribed_text = str(model_any.transcribe(str(tmp_chunk_path))).strip()
                        except Exception as exc:
                            logger.debug("GigaAM failed on segment [%.1f - %.1f]: %s", seg.start, seg.end, exc)
                            transcribed_text = ""

                        if not transcribed_text:
                            continue

                        new_seg = Segment(
                            id=len(output_segments),
                            start=round(seg.start, 3),
                            end=round(seg.end, 3),
                            text=transcribed_text,
                            speaker=seg.speaker,
                            lang="ru",
                            source_track=seg.source_track,
                        )
                        # Word timestamps via interpolation for GigaAM
                        new_seg.words = interpolate_word_timestamps(new_seg)
                        output_segments.append(new_seg)
        finally:
            del model
            cleanup_vram()

        return output_segments
