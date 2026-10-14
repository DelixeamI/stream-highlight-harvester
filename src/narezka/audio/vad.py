"""Voice Activity Detection (VAD) implementation using Silero VAD."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from narezka.audio.base import VADBackend
from narezka.utils.gpu import cleanup_vram

logger = logging.getLogger(__name__)


def load_audio_tensor(audio_path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file to 1D float32 torch.Tensor without torchaudio audio I/O."""
    data, sr = sf.read(str(audio_path.resolve()), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        import scipy.signal as signal

        num_samples = int(round(len(data) * target_sr / sr))
        res_data = signal.resample(data, num_samples)
        if isinstance(res_data, tuple):
            res_data = res_data[0]
        data = res_data

    return torch.from_numpy(data.astype(np.float32))


class SileroVADBackend(VADBackend):
    """VAD implementation using silero-vad."""

    def __init__(self, onnx: bool = True) -> None:
        self.onnx = onnx

    def detect_speech(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        pad_ms: int = 150,
        device: str = "cpu",
    ) -> list[tuple[float, float]]:
        # VERIFY: silero_vad.load_silero_vad and get_speech_timestamps
        import silero_vad

        logger.info("Запуск Silero VAD для файла %s...", audio_path.name)
        try:
            model = silero_vad.load_silero_vad(onnx=self.onnx)
        except Exception as exc:
            logger.warning("Не удалось загрузить ONNX версию Silero VAD (%s), используем PyTorch версию...", exc)
            model = silero_vad.load_silero_vad(onnx=False)

        wav = load_audio_tensor(audio_path, target_sr=16000)

        raw_timestamps = silero_vad.get_speech_timestamps(
            audio=wav,
            model=model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=pad_ms,
            return_seconds=True,
        )

        intervals: list[tuple[float, float]] = []
        for item in raw_timestamps:
            start_s = round(float(item["start"]), 3)
            end_s = round(float(item["end"]), 3)
            if end_s > start_s:
                intervals.append((start_s, end_s))

        del model, wav
        cleanup_vram()

        logger.info("VAD: обнаружено %d речевых интервалов в %s", len(intervals), audio_path.name)
        return intervals


def get_vad_backend(name: str = "silero") -> VADBackend:
    """Factory to instantiate VAD backend."""
    if name == "silero":
        return SileroVADBackend(onnx=True)
    raise ValueError(f"Неизвестный бэкенд VAD: '{name}'. Доступен: 'silero'")
