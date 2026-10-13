"""Short-term loudness and robust z-score feature extraction.

Implements EBU R128 / ITU-R BS.1770 short-term loudness with a 400ms window
and 10-minute rolling median + MAD robust normalization as specified in TZ S6.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import scipy.ndimage as ndimage
import soundfile as sf

from narezka.models import Event

logger = logging.getLogger(__name__)


def compute_robust_zscore(
    values: np.ndarray,
    window_frames: int,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute rolling robust z-score using median and MAD (Median Absolute Deviation).

    Formula: z = (x - median) / (1.4826 * MAD + eps)
    """
    if len(values) == 0:
        return np.array([], dtype=np.float32)

    # Window size must be odd and at least 3
    win = max(3, window_frames)
    if win % 2 == 0:
        win += 1

    # In case data length is smaller than window
    win = min(win, len(values) if len(values) % 2 == 1 else max(1, len(values) - 1))
    if win < 3:
        std = float(np.std(values)) + eps
        return ((values - np.median(values)) / std).astype(np.float32)

    rolling_med = ndimage.median_filter(values, size=win, mode="nearest")
    rolling_mad = ndimage.median_filter(np.abs(values - rolling_med), size=win, mode="nearest")

    scale = 1.4826 * rolling_mad + eps
    z = (values - rolling_med) / scale
    return z.astype(np.float32)


def compute_short_term_loudness(
    audio: np.ndarray,
    sample_rate: int,
    resolution_s: float = 0.5,
    window_s: float = 0.4,
) -> np.ndarray:
    """Compute short-term loudness (in LUFS/dB) on a uniform time grid.

    Args:
        audio: 1D mono float32 array in range [-1.0, 1.0].
        sample_rate: Audio sampling rate (e.g. 16000).
        resolution_s: Grid resolution step in seconds (default 0.5s).
        window_s: Analysis window length in seconds (default 0.4s = 400ms).

    Returns:
        1D array of short-term loudness values in LUFS (clamped to [-70.0, 0.0]).
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    total_duration = len(audio) / sample_rate
    num_frames = int(np.ceil(total_duration / resolution_s))
    if num_frames <= 0:
        num_frames = 1

    meter = pyln.Meter(sample_rate, filter_class="K-weighting", block_size=window_s)
    window_samples = int(round(window_s * sample_rate))

    loudness_values = np.full(num_frames, -70.0, dtype=np.float32)

    for i in range(num_frames):
        center_sample = int(round((i * resolution_s + resolution_s / 2.0) * sample_rate))
        start_sample = max(0, center_sample - window_samples // 2)
        end_sample = min(len(audio), start_sample + window_samples)

        chunk = audio[start_sample:end_sample]
        if len(chunk) < sample_rate * 0.1:  # skip chunks shorter than 100ms
            continue

        # Check for non-silence
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        if rms < 1e-4:
            loudness_values[i] = -70.0
            continue

        try:
            val = meter.integrated_loudness(chunk)
            if np.isnan(val) or np.isneginf(val) or val < -70.0:
                val = -70.0
            elif np.isposinf(val) or val > 0.0:
                val = 0.0
            loudness_values[i] = float(val)
        except Exception:
            # Fallback to RMS dBFS
            db = 20.0 * np.log10(max(float(rms), 1e-7))
            loudness_values[i] = float(np.clip(db, -70.0, 0.0))

    return loudness_values


def extract_loudness_features(
    audio_path_or_data: Path | np.ndarray,
    sample_rate: int = 16000,
    resolution_s: float = 0.5,
    window_s: float = 0.4,
    rolling_window_s: float = 600.0,
    cap: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, list[Event]]:
    """Extract raw loudness and robust z-score feature series plus discrete burst events.

    Args:
        audio_path_or_data: File path to WAV file or numpy float array.
        sample_rate: Audio sampling rate.
        resolution_s: Timeline resolution step in seconds.
        window_s: Loudness analysis window.
        rolling_window_s: Time window for rolling median and MAD (default 10 minutes = 600s).
        cap: Maximum cap for z-score clipping.

    Returns:
        (loud_raw, loud_z, events)
    """
    if isinstance(audio_path_or_data, Path):
        data, sr = sf.read(str(audio_path_or_data), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    else:
        data = audio_path_or_data
        sr = sample_rate

    raw_loudness = compute_short_term_loudness(
        audio=data,
        sample_rate=sr,
        resolution_s=resolution_s,
        window_s=window_s,
    )

    window_frames = int(round(rolling_window_s / resolution_s))
    z = compute_robust_zscore(raw_loudness, window_frames=window_frames)
    loud_z = np.clip(z, 0.0, cap)

    # Detect discrete loud events (loud_z >= 3.0 sustained for >= 0.5s)
    events: list[Event] = []
    is_active = False
    event_start = 0.0
    event_scores: list[float] = []

    for i, score in enumerate(loud_z):
        t = i * resolution_s
        if score >= 3.0:
            if not is_active:
                is_active = True
                event_start = t
                event_scores = [float(score)]
            else:
                event_scores.append(float(score))
        else:
            if is_active:
                event_end = t
                duration = event_end - event_start
                if duration >= 0.5:
                    max_s = max(event_scores)
                    norm_score = min(1.0, max_s / cap)
                    event_id = f"evt_loud_{int(event_start * 10):06d}"
                    events.append(
                        Event(
                            id=event_id,
                            source="loudness",
                            type="loud_burst",
                            start=round(event_start, 2),
                            end=round(event_end, 2),
                            score=round(norm_score, 3),
                            payload={"loud_z_max": round(max_s, 2), "duration_s": round(duration, 2)},
                        )
                    )
                is_active = False

    if is_active:
        t_end = len(loud_z) * resolution_s
        duration = t_end - event_start
        if duration >= 0.5:
            max_s = max(event_scores)
            norm_score = min(1.0, max_s / cap)
            event_id = f"evt_loud_{int(event_start * 10):06d}"
            events.append(
                Event(
                    id=event_id,
                    source="loudness",
                    type="loud_burst",
                    start=round(event_start, 2),
                    end=round(t_end, 2),
                    score=round(norm_score, 3),
                    payload={"loud_z_max": round(max_s, 2), "duration_s": round(duration, 2)},
                )
            )

    return raw_loudness, loud_z, events
