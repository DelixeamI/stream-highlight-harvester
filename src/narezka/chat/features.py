"""Extraction and temporal alignment of Twitch chat features.

Calculates chat message rate, robust z-score, laughter and hype ratios,
and automatically estimates viewer reaction latency via cross-correlation with audio excitation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal as signal
import yaml

from narezka.audio.loudness import compute_robust_zscore
from narezka.chat.base import ChatMessage
from narezka.config import ChatConfig, FeaturesConfig, TimelineConfig
from narezka.models import Event

logger = logging.getLogger(__name__)


@dataclass
class ChatFeaturesResult:
    """Extracted and aligned chat feature signals."""

    timestamps: np.ndarray  # (T,) timeline in seconds
    chat_rate: np.ndarray  # (T,) raw messages per second
    chat_rate_z: np.ndarray  # (T,) robust z-score capped
    chat_laugh: np.ndarray  # (T,) fraction of laughter messages
    chat_hype: np.ndarray  # (T,) fraction of hype messages
    estimated_delay_s: float  # estimated broadcast + reaction delay in seconds
    events: list[Event]  # discrete chat events


def load_chat_markers(markers_path: Path) -> tuple[set[str], set[str]]:
    """Load laugh and hype marker sets from YAML file."""
    laugh_markers: set[str] = set()
    hype_markers: set[str] = set()

    if not markers_path.is_file():
        logger.warning("Chat markers file not found at %s, using defaults", markers_path)
        return {"kekw", "lul", "omegalul", "ахах", "хаха", "ору", "лол"}, {"pog", "имба", "красава", "жесть"}

    try:
        with open(markers_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            for m in data.get("laugh_markers", []):
                laugh_markers.add(str(m).strip().lower())
            for m in data.get("hype_markers", []):
                hype_markers.add(str(m).strip().lower())
    except Exception as e:
        logger.error("Error reading chat markers from %s: %s", markers_path, e)

    return laugh_markers, hype_markers


def extract_raw_chat_signals(
    messages: list[ChatMessage],
    duration_s: float,
    markers_path: Path,
    resolution_s: float = 0.5,
    window_s: float = 2.0,
    cap_z: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute raw chat message rate, z-score, laugh and hype fractions on uniform grid."""
    num_frames = int(np.ceil(duration_s / resolution_s))
    if num_frames <= 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    if not messages:
        zeros = np.zeros(num_frames, dtype=np.float32)
        return zeros, zeros, zeros, zeros

    laugh_markers, hype_markers = load_chat_markers(markers_path)

    # Pre-parse messages: timestamp, is_laugh, is_hype
    msg_times = np.array([m.timestamp_s for m in messages], dtype=np.float32)
    is_laugh = np.zeros(len(messages), dtype=bool)
    is_hype = np.zeros(len(messages), dtype=bool)

    for i, m in enumerate(messages):
        words = set(m.text.lower().split())
        text_lower = m.text.lower()
        if any(lm in words or lm in text_lower for lm in laugh_markers):
            is_laugh[i] = True
        if any(hm in words or hm in text_lower for hm in hype_markers):
            is_hype[i] = True

    chat_rate = np.zeros(num_frames, dtype=np.float32)
    chat_laugh = np.zeros(num_frames, dtype=np.float32)
    chat_hype = np.zeros(num_frames, dtype=np.float32)

    half_win = window_s / 2.0

    for i in range(num_frames):
        t = i * resolution_s
        idx_start = int(np.searchsorted(msg_times, t - half_win, side="left"))
        idx_end = int(np.searchsorted(msg_times, t + half_win, side="right"))

        count = idx_end - idx_start
        if count > 0:
            chat_rate[i] = count / window_s
            laugh_count = int(np.sum(is_laugh[idx_start:idx_end]))
            hype_count = int(np.sum(is_hype[idx_start:idx_end]))
            chat_laugh[i] = laugh_count / count
            chat_hype[i] = hype_count / count

    # Compute robust z-score for chat rate
    window_frames = int(round(600.0 / resolution_s))  # 10 minutes rolling
    z = compute_robust_zscore(chat_rate, window_frames=window_frames)
    chat_rate_z = np.clip(z, 0.0, cap_z).astype(np.float32)

    return chat_rate, chat_rate_z, chat_laugh, chat_hype


def estimate_chat_delay(
    chat_rate_z: np.ndarray,
    audio_excitation: np.ndarray,
    resolution_s: float = 0.5,
    min_delay_s: float = 3.0,
    max_delay_s: float = 25.0,
    default_s: float = 8.0,
) -> float:
    """Estimate latency between stream excitation and chat reaction via cross-correlation.

    Returns the estimated delay in seconds, or default_s if correlation is insignificant.
    """
    if len(chat_rate_z) == 0 or len(audio_excitation) == 0:
        return default_s

    # Standardize signals
    n = min(len(chat_rate_z), len(audio_excitation))
    c = chat_rate_z[:n] - np.mean(chat_rate_z[:n])
    a = audio_excitation[:n] - np.mean(audio_excitation[:n])

    std_c = np.std(c)
    std_a = np.std(a)
    if std_c < 1e-4 or std_a < 1e-4:
        return default_s

    c_norm = c / std_c
    a_norm = a / std_a

    # Cross correlation: positive lag means chat lags behind audio
    corr = signal.correlate(c_norm, a_norm, mode="full") / n
    lags = signal.correlation_lags(len(c_norm), len(a_norm), mode="full")

    # Search in window [min_delay_s, max_delay_s]
    min_lag = int(round(min_delay_s / resolution_s))
    max_lag = int(round(max_delay_s / resolution_s))

    mask = (lags >= min_lag) & (lags <= max_lag)
    if not np.any(mask):
        return default_s

    valid_corrs = corr[mask]
    valid_lags = lags[mask]

    best_idx = int(np.argmax(valid_corrs))
    max_corr = float(valid_corrs[best_idx])
    best_lag = int(valid_lags[best_idx])

    estimated_delay = best_lag * resolution_s

    # Require minimum correlation peak
    if max_corr >= 0.12:
        logger.info(
            "Estimated chat delay: %.1f s (correlation coefficient: %.3f)",
            estimated_delay,
            max_corr,
        )
        return float(estimated_delay)
    else:
        logger.info(
            "Chat correlation weak (max: %.3f), falling back to default delay: %.1f s",
            max_corr,
            default_s,
        )
        return default_s


def shift_signal_backward(sig: np.ndarray, shift_seconds: float, resolution_s: float) -> np.ndarray:
    """Shift signal backward by shift_seconds so reactions align with stream events: f_aligned(t) = f(t + D)."""
    shift_frames = int(round(shift_seconds / resolution_s))
    if shift_frames <= 0:
        return sig.copy()

    aligned = np.zeros_like(sig)
    if shift_frames < len(sig):
        aligned[:-shift_frames] = sig[shift_frames:]
    return aligned


def extract_and_align_chat(
    messages: list[ChatMessage],
    duration_s: float,
    audio_excitation: np.ndarray | None,
    chat_cfg: ChatConfig,
    features_cfg: FeaturesConfig,
    timeline_cfg: TimelineConfig,
    markers_path: Path,
) -> ChatFeaturesResult:
    """Extract chat features, estimate delay against audio excitation, and align signals."""
    res_s = timeline_cfg.resolution_s
    num_frames = int(np.ceil(duration_s / res_s))
    timestamps = np.arange(num_frames, dtype=np.float32) * res_s

    if not messages:
        zeros = np.zeros(num_frames, dtype=np.float32)
        return ChatFeaturesResult(
            timestamps=timestamps,
            chat_rate=zeros,
            chat_rate_z=zeros,
            chat_laugh=zeros,
            chat_hype=zeros,
            estimated_delay_s=chat_cfg.delay.default_s,
            events=[],
        )

    raw_rate, raw_rate_z, raw_laugh, raw_hype = extract_raw_chat_signals(
        messages=messages,
        duration_s=duration_s,
        markers_path=markers_path,
        resolution_s=res_s,
        window_s=2.0,
        cap_z=features_cfg.caps.get("chat_rate_z", 6.0),
    )

    # Estimate chat delay
    if chat_cfg.delay.auto and audio_excitation is not None:
        delay_s = estimate_chat_delay(
            chat_rate_z=raw_rate_z,
            audio_excitation=audio_excitation,
            resolution_s=res_s,
            min_delay_s=chat_cfg.delay.min_delay_s,
            max_delay_s=chat_cfg.delay.max_delay_s,
            default_s=chat_cfg.delay.default_s,
        )
    else:
        delay_s = chat_cfg.delay.default_s

    # Align signals backwards
    aligned_rate = shift_signal_backward(raw_rate, delay_s, res_s)
    aligned_rate_z = shift_signal_backward(raw_rate_z, delay_s, res_s)
    aligned_laugh = shift_signal_backward(raw_laugh, delay_s, res_s)
    aligned_hype = shift_signal_backward(raw_hype, delay_s, res_s)

    # Detect discrete chat events
    events: list[Event] = []

    # 1. Chat rate spike (chat_rate_z >= 3.5)
    in_spike = False
    spk_start = 0.0
    spk_scores: list[float] = []
    for i, z_val in enumerate(aligned_rate_z):
        t = i * res_s
        if z_val >= 3.5:
            if not in_spike:
                in_spike = True
                spk_start = t
                spk_scores = [float(z_val)]
            else:
                spk_scores.append(float(z_val))
        else:
            if in_spike:
                dur = t - spk_start
                if dur >= 1.0:
                    max_z = max(spk_scores)
                    norm_score = min(1.0, max_z / features_cfg.caps.get("chat_rate_z", 6.0))
                    evt_id = f"evt_chat_spk_{int(spk_start * 10):06d}"
                    events.append(
                        Event(
                            id=evt_id,
                            source="chat",
                            type="chat_spike",
                            start=round(spk_start, 2),
                            end=round(t, 2),
                            score=round(norm_score, 3),
                            payload={"chat_rate_z_max": round(max_z, 2), "duration_s": round(dur, 2)},
                        )
                    )
                in_spike = False

    # 2. Chat laughter surge (chat_laugh >= 0.4 and rate_z >= 1.5)
    in_claugh = False
    cl_start = 0.0
    cl_scores: list[float] = []
    for i, l_val in enumerate(aligned_laugh):
        t = i * res_s
        if l_val >= 0.4 and aligned_rate_z[i] >= 1.5:
            if not in_claugh:
                in_claugh = True
                cl_start = t
                cl_scores = [float(l_val)]
            else:
                cl_scores.append(float(l_val))
        else:
            if in_claugh:
                dur = t - cl_start
                if dur >= 1.0:
                    max_l = max(cl_scores)
                    evt_id = f"evt_chat_laugh_{int(cl_start * 10):06d}"
                    events.append(
                        Event(
                            id=evt_id,
                            source="chat",
                            type="chat_laughter",
                            start=round(cl_start, 2),
                            end=round(t, 2),
                            score=round(max_l, 3),
                            payload={"chat_laugh_max": round(max_l, 3), "duration_s": round(dur, 2)},
                        )
                    )
                in_claugh = False

    return ChatFeaturesResult(
        timestamps=timestamps,
        chat_rate=aligned_rate,
        chat_rate_z=aligned_rate_z,
        chat_laugh=aligned_laugh,
        chat_hype=aligned_hype,
        estimated_delay_s=delay_s,
        events=events,
    )
