"""Faster-Whisper ASR backend implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf
import torch

from narezka.asr.base import ASRBackend, Segment, SpeechSegment, Word
from narezka.config import AsrConfig
from narezka.utils.gpu import cleanup_vram

logger = logging.getLogger(__name__)


class FasterWhisperBackend(ASRBackend):
    """ASR backend using faster-whisper (CTranslate2)."""

    name = "faster_whisper"

    def transcribe(
        self,
        audio_path: Path,
        segments: list[SpeechSegment],
        config: AsrConfig,
        device: str = "cuda",
    ) -> list[Segment]:
        # VERIFY: faster_whisper WhisperModel parameters and word_timestamps on CUDA sm_120
        from faster_whisper import WhisperModel

        target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        compute_type = config.faster_whisper.compute_type
        if target_device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        model_name = config.faster_whisper.model
        logger.info(
            "Загрузка модели faster-whisper '%s' на %s (compute_type=%s)...",
            model_name,
            target_device,
            compute_type,
        )

        model = WhisperModel(
            model_name,
            device=target_device,
            compute_type=compute_type,
        )

        output_segments: list[Segment] = []

        try:
            with sf.SoundFile(str(audio_path)) as sfile:
                sr = sfile.samplerate

                # Filter valid segments (skip English game voice lines)
                valid_segs = [s for s in segments if not s.is_game_voice_line and s.lang == "ru"]

                # Group adjacent segments with pause <= 1.0s up to 20.0s max duration
                # to provide sentence context and achieve 5-10x faster transcription speed
                grouped: list[list[SpeechSegment]] = []
                current_group: list[SpeechSegment] = []
                group_start = 0.0

                for seg in valid_segs:
                    if not current_group:
                        current_group.append(seg)
                        group_start = seg.start
                    else:
                        prev_end = current_group[-1].end
                        gap = seg.start - prev_end
                        dur = seg.end - group_start
                        if gap <= 1.0 and dur <= 20.0:
                            current_group.append(seg)
                        else:
                            grouped.append(current_group)
                            current_group = [seg]
                            group_start = seg.start
                if current_group:
                    grouped.append(current_group)

                for grp in grouped:
                    g_start = grp[0].start
                    g_end = grp[-1].end
                    start_sample = int(g_start * sr)
                    num_samples = int((g_end - g_start) * sr)
                    if num_samples <= 0:
                        continue

                    sfile.seek(start_sample)
                    chunk = sfile.read(num_samples, dtype="float32")

                    if chunk.ndim > 1:
                        chunk = chunk.mean(axis=1)

                    whisper_segs, _ = model.transcribe(
                        chunk,
                        language="ru",
                        condition_on_previous_text=False,
                        word_timestamps=True,
                        vad_filter=False,  # VAD already applied in S3
                        beam_size=config.faster_whisper.beam_size,
                    )

                    for w_seg in whisper_segs:
                        seg_text = w_seg.text.strip()
                        if not seg_text:
                            continue

                        words: list[Word] = []
                        if w_seg.words:
                            for w in w_seg.words:
                                words.append(
                                    Word(
                                        text=w.word.strip(),
                                        start=round(g_start + w.start, 3),
                                        end=round(g_start + w.end, 3),
                                        prob=round(w.probability, 3) if w.probability else None,
                                    )
                                )

                        output_segments.append(
                            Segment(
                                id=len(output_segments),
                                start=round(g_start + w_seg.start, 3),
                                end=round(g_start + w_seg.end, 3),
                                text=seg_text,
                                speaker="streamer",
                                lang="ru",
                                words=words,
                                source_track="voice_sep",
                                avg_logprob=round(w_seg.avg_logprob, 3) if w_seg.avg_logprob else None,
                                no_speech_prob=round(w_seg.no_speech_prob, 3) if w_seg.no_speech_prob else None,
                                compression_ratio=round(w_seg.compression_ratio, 3) if w_seg.compression_ratio else None,
                            )
                        )
        finally:
            del model
            cleanup_vram()

        return output_segments
