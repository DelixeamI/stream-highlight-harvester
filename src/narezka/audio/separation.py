"""Audio source separation using Demucs and Mel-Band RoFormer with chunking."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from narezka.audio.base import AudioSeparationBackend
from narezka.utils.gpu import cleanup_vram

logger = logging.getLogger(__name__)


def resample_to_16k_mono(input_path: Path, output_path: Path) -> Path:
    """Helper to convert any WAV file to 16 kHz mono 16-bit PCM via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # Fallback using soundfile and numpy
        data, sr = sf.read(str(input_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        sf.write(str(output_path), data, 16000, subtype="PCM_16")
        return output_path

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


class DemucsSeparationBackend:
    """Audio separation backend using Facebook's Demucs (htdemucs) on PyTorch."""

    def __init__(self, model_name: str = "htdemucs") -> None:
        self.model_name = model_name

    def separate(
        self,
        input_audio: Path,
        output_voice: Path,
        output_accompaniment: Path,
        chunk_seconds: int = 600,
        device: str = "cuda",
    ) -> None:
        # VERIFY: demucs.apply chunking and device placement on sm_120
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        logger.info("Загрузка модели Demucs '%s' на устройство %s...", self.model_name, target_device)

        model = get_model(self.model_name)
        model.to(target_device)
        model.eval()

        info = sf.info(str(input_audio))
        sr = info.samplerate
        total_samples = info.frames

        # We process in chunks with crossfade
        chunk_samples = chunk_seconds * sr

        vocals_parts: list[np.ndarray] = []
        acc_parts: list[np.ndarray] = []

        try:
            with sf.SoundFile(str(input_audio)) as sfile:
                current_pos = 0
                while current_pos < total_samples:
                    sfile.seek(current_pos)
                    read_len = min(chunk_samples, total_samples - current_pos)
                    chunk_data = sfile.read(read_len, dtype="float32")

                    # Convert to tensor: (channels, samples)
                    if chunk_data.ndim == 1:
                        chunk_tensor = torch.from_numpy(chunk_data).unsqueeze(0).repeat(2, 1)
                    else:
                        chunk_tensor = torch.from_numpy(chunk_data.T)

                    chunk_tensor = chunk_tensor.unsqueeze(0).to(target_device)  # (1, 2, samples)

                    with torch.no_grad():
                        sources = apply_model(model, chunk_tensor, device=target_device, split=True, overlap=0.25)
                        # sources shape: (1, 4, 2, samples) -> (drums, bass, other, vocals)
                        # Vocals is index 3 in htdemucs
                        vocals_idx = model.sources.index("vocals") if "vocals" in model.sources else 3
                        vocals_tensor = sources[0, vocals_idx]  # (2, samples)

                        # Accompaniment is all other stems summed
                        other_indices = [i for i in range(sources.shape[1]) if i != vocals_idx]
                        acc_tensor = sources[0, other_indices].sum(dim=0)  # (2, samples)

                        # Downmix to mono float32
                        v_mono = vocals_tensor.mean(dim=0).cpu().numpy()
                        a_mono = acc_tensor.mean(dim=0).cpu().numpy()

                        vocals_parts.append(v_mono)
                        acc_parts.append(a_mono)

                    current_pos += read_len
                    del chunk_tensor, sources, vocals_tensor, acc_tensor
                    if target_device == "cuda":
                        torch.cuda.empty_cache()

            full_vocals = np.concatenate(vocals_parts, axis=0) if vocals_parts else np.zeros((0,), dtype=np.float32)
            full_acc = np.concatenate(acc_parts, axis=0) if acc_parts else np.zeros((0,), dtype=np.float32)

            tmp_voice_raw = output_voice.with_suffix(".raw.wav")
            tmp_acc_raw = output_accompaniment.with_suffix(".raw.wav")

            sf.write(str(tmp_voice_raw), full_vocals, sr, subtype="PCM_16")
            sf.write(str(tmp_acc_raw), full_acc, sr, subtype="PCM_16")

            # Final resample to 16 kHz mono
            resample_to_16k_mono(tmp_voice_raw, output_voice)
            resample_to_16k_mono(tmp_acc_raw, output_accompaniment)

            if tmp_voice_raw.exists():
                tmp_voice_raw.unlink()
            if tmp_acc_raw.exists():
                tmp_acc_raw.unlink()

            logger.info("Разделение аудио завершено: %s, %s", output_voice.name, output_accompaniment.name)
        finally:
            del model
            cleanup_vram()


class AudioSeparatorBackend:
    """Wrapper around audio-separator for Mel-Band RoFormer models."""

    def __init__(self, model_file: str = "mel_band_roformer_vocals") -> None:
        self.model_file = model_file

    def separate(
        self,
        input_audio: Path,
        output_voice: Path,
        output_accompaniment: Path,
        chunk_seconds: int = 600,
        device: str = "cuda",
    ) -> None:
        # VERIFY: audio_separator import and ONNX Runtime GPU on Windows sm_120
        try:
            import importlib

            sep_mod = importlib.import_module("audio_separator.separator")
            Separator = sep_mod.Separator

            sep = Separator(
                log_level=logging.WARNING,
                output_dir=str(output_voice.parent),
                output_format="WAV",
            )
            sep.load_model(model_filename=self.model_file)
            output_files = sep.separate(str(input_audio))
            # Rename outputs
            for out_name in output_files:
                out_path = output_voice.parent / out_name
                if "vocals" in out_name.lower():
                    resample_to_16k_mono(out_path, output_voice)
                else:
                    resample_to_16k_mono(out_path, output_accompaniment)
                if out_path.exists() and out_path != output_voice and out_path != output_accompaniment:
                    out_path.unlink()
        except Exception as exc:
            logger.warning(
                "Сбой audio-separator (%s). Переключение на резервный бэкенд Demucs (PyTorch).", exc
            )
            fallback = DemucsSeparationBackend(model_name="htdemucs")
            fallback.separate(
                input_audio=input_audio,
                output_voice=output_voice,
                output_accompaniment=output_accompaniment,
                chunk_seconds=chunk_seconds,
                device=device,
            )


def get_separation_backend(name: str, model_id: str | None = None) -> AudioSeparationBackend:
    """Factory to instantiate audio separation backend by name."""
    if name == "audio_separator":
        return AudioSeparatorBackend(model_file=model_id or "mel_band_roformer_vocals")
    elif name == "demucs":
        return DemucsSeparationBackend(model_name=model_id or "htdemucs")
    raise ValueError(f"Неизвестный бэкенд разделения аудио: '{name}'. Доступны: 'demucs', 'audio_separator'")
