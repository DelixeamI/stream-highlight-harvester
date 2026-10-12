"""Language Identification (LID) for speech segments using SpeechBrain and Whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from narezka.audio.base import LIDBackend
from narezka.utils.gpu import cleanup_vram

logger = logging.getLogger(__name__)


class SpeechBrainLidBackend(LIDBackend):
    """LID backend using SpeechBrain VoxLingua107 ECAPA embeddings."""

    def __init__(self, model_source: str = "speechbrain/lang-id-voxlingua107-ecapa") -> None:
        self.model_source = model_source
        self._classifier: Any = None

    def _load_model(self, device: str = "cuda") -> Any:
        # VERIFY: SpeechBrain EncoderClassifier.from_hparams and savedir caching on Windows
        from speechbrain.inference.classifiers import EncoderClassifier

        target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        save_dir = Path.home() / ".cache" / "narezka" / "models" / "speechbrain_voxlingua107"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Загрузка модели SpeechBrain LID '%s'...", self.model_source)
        classifier = EncoderClassifier.from_hparams(
            source=self.model_source,
            savedir=str(save_dir),
            run_opts={"device": target_device},
        )
        return classifier

    def identify_language(
        self,
        audio_path: Path,
        intervals: list[tuple[float, float]],
        min_segment_s: float = 1.5,
        device: str = "cuda",
    ) -> list[dict[str, Any]]:
        if not intervals:
            return []

        # If all segments are shorter than min_segment_s, return ru without loading model
        if not any((en - st) >= min_segment_s for st, en in intervals):
            return [
                {"start": st, "end": en, "lang": "ru", "confidence": 1.0}
                for st, en in intervals
            ]

        try:
            classifier = self._load_model(device=device)
        except Exception as exc:
            logger.warning("Не удалось загрузить SpeechBrain LID (%s). Переключение на Whisper LID.", exc)
            fallback = WhisperLidBackend()
            return fallback.identify_language(audio_path, intervals, min_segment_s, device)

        results: list[dict[str, Any]] = []

        try:
            with sf.SoundFile(str(audio_path)) as sfile:
                sr = sfile.samplerate
                for start_s, end_s in intervals:
                    dur = end_s - start_s
                    # Short segments below min_segment_s default to ru
                    if dur < min_segment_s:
                        results.append({
                            "start": start_s,
                            "end": end_s,
                            "lang": "ru",
                            "confidence": 1.0,
                        })
                        continue

                    start_frame = int(start_s * sr)
                    num_frames = int(dur * sr)
                    sfile.seek(start_frame)
                    chunk = sfile.read(num_frames, dtype="float32")

                    if chunk.ndim > 1:
                        chunk = chunk.mean(axis=1)

                    waveform = torch.from_numpy(chunk).unsqueeze(0)
                    with torch.no_grad():
                        out_prob, score, index, text_lab = classifier.classify_batch(waveform)
                        pred_lang = text_lab[0].lower() if text_lab else "ru"
                        conf = float(score[0].item()) if hasattr(score[0], "item") else float(score[0])

                    # Standardize language codes (e.g. ru: Russian, en: English)
                    clean_lang = "ru" if "ru" in pred_lang else ("en" if "en" in pred_lang else pred_lang)
                    results.append({
                        "start": start_s,
                        "end": end_s,
                        "lang": clean_lang,
                        "confidence": round(conf, 3),
                    })
        finally:
            del classifier
            cleanup_vram()

        return results


class WhisperLidBackend(LIDBackend):
    """Fallback LID backend using faster-whisper detect_language."""

    def __init__(self, model_size: str = "tiny") -> None:
        self.model_size = model_size

    def identify_language(
        self,
        audio_path: Path,
        intervals: list[tuple[float, float]],
        min_segment_s: float = 1.5,
        device: str = "cuda",
    ) -> list[dict[str, Any]]:
        # VERIFY: faster_whisper WhisperModel.detect_language
        from faster_whisper import WhisperModel

        if not intervals:
            return []

        # If all segments are shorter than min_segment_s, return ru without loading model
        if not any((en - st) >= min_segment_s for st, en in intervals):
            return [
                {"start": st, "end": en, "lang": "ru", "confidence": 1.0}
                for st, en in intervals
            ]

        target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        compute_type = "float16" if target_device == "cuda" else "int8"

        model = WhisperModel(self.model_size, device=target_device, compute_type=compute_type)
        results: list[dict[str, Any]] = []

        try:
            with sf.SoundFile(str(audio_path)) as sfile:
                sr = sfile.samplerate
                for start_s, end_s in intervals:
                    dur = end_s - start_s
                    if dur < min_segment_s:
                        results.append({
                            "start": start_s,
                            "end": end_s,
                            "lang": "ru",
                            "confidence": 1.0,
                        })
                        continue

                    start_frame = int(start_s * sr)
                    num_frames = int(dur * sr)
                    sfile.seek(start_frame)
                    chunk = sfile.read(num_frames, dtype="float32")
                    if chunk.ndim > 1:
                        chunk = chunk.mean(axis=1)

                    # faster_whisper accepts numpy float32 array
                    lang, prob, _ = model.detect_language(chunk)
                    results.append({
                        "start": start_s,
                        "end": end_s,
                        "lang": lang,
                        "confidence": round(prob, 3),
                    })
        finally:
            del model
            cleanup_vram()

        return results


def get_lid_backend(name: str = "speechbrain") -> LIDBackend:
    """Factory to instantiate LID backend."""
    if name == "speechbrain":
        return SpeechBrainLidBackend()
    elif name == "whisper":
        return WhisperLidBackend()
    raise ValueError(f"Неизвестный бэкенд LID: '{name}'. Доступны: 'speechbrain', 'whisper'")
