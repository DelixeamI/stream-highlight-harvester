"""Fundamental frequency (pitch/F0) tracking and robust z-score normalization.

Computes median F0 on voiced frames and robust z-score over a 10-minute rolling
median + MAD window, adapted to the streamer's vocal register as specified in TZ S6.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import soundfile as sf
import torch
import torchaudio.functional as F

logger = logging.getLogger(__name__)


def compute_pitch_track(
    audio: np.ndarray,
    sample_rate: int = 16000,
    resolution_s: float = 0.5,
    freq_low: int = 80,
    freq_high: int = 600,
    frame_time: float = 0.01,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame-level pitch track and downsample to timeline resolution grid.

    Args:
        audio: 1D mono float32 numpy array.
        sample_rate: Audio sampling rate.
        resolution_s: Output grid resolution in seconds (default 0.5s).
        freq_low: Minimum fundamental frequency in Hz.
        freq_high: Maximum fundamental frequency in Hz.
        frame_time: Frame analysis step in seconds.
        device: Torch computation device ("cpu" or "cuda").

    Returns:
        (f0_grid, voiced_mask) arrays on uniform resolution_s grid.
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=bool)

    total_duration = len(audio) / sample_rate
    num_frames = int(np.ceil(total_duration / resolution_s))
    if num_frames <= 0:
        num_frames = 1

    chunk_samples = 120 * sample_rate
    pitch_sample_list = []
    with torch.no_grad():
        for c_start in range(0, len(audio), chunk_samples):
            c_audio = audio[c_start : c_start + chunk_samples]
            tensor_chunk = torch.from_numpy(c_audio).unsqueeze(0).to(device)
            try:
                pitch_raw = F.detect_pitch_frequency(
                    tensor_chunk,
                    sample_rate=sample_rate,
                    frame_time=frame_time,
                    win_length=30,
                    freq_low=freq_low,
                    freq_high=freq_high,
                )
                pitch_sample_list.append(pitch_raw.squeeze(0).cpu().numpy())
            except Exception as e:
                logger.warning("TorchAudio detect_pitch_frequency failed on chunk: %s", e)
                pitch_sample_list.append(
                    np.zeros(int(len(c_audio) / (sample_rate * frame_time)), dtype=np.float32)
                )

    pitch_samples = np.concatenate(pitch_sample_list) if pitch_sample_list else np.array([], dtype=np.float32)

    samples_per_grid_frame = int(round(resolution_s / frame_time))
    f0_grid = np.zeros(num_frames, dtype=np.float32)
    voiced_mask = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start_idx = i * samples_per_grid_frame
        end_idx = min(len(pitch_samples), (i + 1) * samples_per_grid_frame)

        if start_idx >= len(pitch_samples):
            break

        chunk_pitches = pitch_samples[start_idx:end_idx]
        # Voiced frames have pitch > freq_low
        voiced_pitches = chunk_pitches[chunk_pitches >= freq_low]

        if len(voiced_pitches) > 0 and (len(voiced_pitches) / len(chunk_pitches)) >= 0.25:
            f0_grid[i] = float(np.median(voiced_pitches))
            voiced_mask[i] = True
        else:
            f0_grid[i] = 0.0
            voiced_mask[i] = False

    return f0_grid, voiced_mask


def extract_pitch_features(
    audio_path_or_data: Path | np.ndarray,
    sample_rate: int = 16000,
    resolution_s: float = 0.5,
    rolling_window_s: float = 600.0,
    cap: float = 4.0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw F0 and robust z-score pitch feature series.

    Args:
        audio_path_or_data: File path to WAV file or numpy float array.
        sample_rate: Audio sampling rate.
        resolution_s: Output grid resolution.
        rolling_window_s: Time window for rolling median and MAD (default 10 minutes = 600s).
        cap: Maximum cap for z-score clipping (default 4.0).
        device: Torch device ("cpu" or "cuda").

    Returns:
        (f0_raw, pitch_z) arrays on uniform grid.
    """
    if isinstance(audio_path_or_data, Path):
        data, sr = sf.read(str(audio_path_or_data), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    else:
        data = audio_path_or_data
        sr = sample_rate

    f0_raw, voiced_mask = compute_pitch_track(
        audio=data,
        sample_rate=sr,
        resolution_s=resolution_s,
        device=device,
    )

    if not np.any(voiced_mask):
        return f0_raw, np.zeros_like(f0_raw)

    window_frames = int(round(rolling_window_s / resolution_s))
    win = max(3, window_frames)
    if win % 2 == 0:
        win += 1

    # To calculate rolling baseline properly, interpolate F0 over unvoiced gaps
    voiced_indices = np.where(voiced_mask)[0]
    all_indices = np.arange(len(f0_raw))
    f0_interp = np.interp(all_indices, voiced_indices, f0_raw[voiced_indices])

    # Rolling median and MAD over interpolated F0
    win = min(win, len(f0_interp) if len(f0_interp) % 2 == 1 else max(1, len(f0_interp) - 1))
    if win >= 3:
        rolling_med = ndimage.median_filter(f0_interp, size=win, mode="nearest")
        rolling_mad = ndimage.median_filter(np.abs(f0_interp - rolling_med), size=win, mode="nearest")
        scale = 1.4826 * rolling_mad + 1e-4
        z = (f0_interp - rolling_med) / scale
    else:
        med = float(np.median(f0_raw[voiced_mask]))
        std = float(np.std(f0_raw[voiced_mask])) + 1e-4
        z = (f0_interp - med) / std

    # Zero out unvoiced frames
    z_clean = np.where(voiced_mask, z, 0.0)
    pitch_z = np.clip(z_clean, 0.0, cap).astype(np.float32)

    return f0_raw, pitch_z
