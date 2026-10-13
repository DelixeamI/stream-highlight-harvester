"""Unified timeline construction and integral interest score calculation.

Calculates integral score:
S(t) = sum(w_i * clip(f_i(t), 0, cap_i))
and applies Gaussian smoothing (sigma = 1.5s) as specified in TZ S10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndimage

from narezka.config import FeaturesConfig, TimelineConfig

logger = logging.getLogger(__name__)


@dataclass
class TimelineResult:
    """Unified timeline with raw and smoothed integral interest curves."""

    timestamps: np.ndarray  # (T,) in seconds
    features: dict[str, np.ndarray]  # name -> (T,) array
    s_raw: np.ndarray  # (T,) raw weighted sum
    s_smooth: np.ndarray  # (T,) Gaussian smoothed curve


def build_unified_timeline(
    features_dict: dict[str, np.ndarray],
    duration_s: float,
    timeline_cfg: TimelineConfig,
    features_cfg: FeaturesConfig,
) -> TimelineResult:
    """Combine feature tracks, compute weighted interest score S(t), and apply Gaussian smoothing.

    Args:
        features_dict: Dictionary of feature name -> 1D array.
        duration_s: Total stream duration in seconds.
        timeline_cfg: TimelineConfig with resolution_s and smoothing_sigma_s.
        features_cfg: FeaturesConfig with feature weights and caps.

    Returns:
        TimelineResult containing all normalized features and smoothed interest curve.
    """
    res_s = timeline_cfg.resolution_s
    num_frames = int(np.ceil(duration_s / res_s))
    if num_frames <= 0:
        empty = np.array([], dtype=np.float32)
        return TimelineResult(timestamps=empty, features={}, s_raw=empty, s_smooth=empty)

    timestamps = np.arange(num_frames, dtype=np.float32) * res_s

    # Align all feature arrays to uniform length num_frames
    aligned_features: dict[str, np.ndarray] = {}
    for name, arr in features_dict.items():
        if len(arr) < num_frames:
            padded = np.pad(arr, (0, num_frames - len(arr)))
            aligned_features[name] = padded.astype(np.float32)
        else:
            aligned_features[name] = arr[:num_frames].astype(np.float32)

    # Fill any missing expected features with zeros
    for expected_feat in features_cfg.weights.keys():
        if expected_feat not in aligned_features:
            aligned_features[expected_feat] = np.zeros(num_frames, dtype=np.float32)

    # Compute raw integral score S_raw(t)
    s_raw = np.zeros(num_frames, dtype=np.float32)

    for feat_name, weight in features_cfg.weights.items():
        if feat_name in aligned_features:
            arr = aligned_features[feat_name]
            cap = features_cfg.caps.get(feat_name, None)
            if cap is not None and cap > 0:
                clipped = np.clip(arr, 0.0, cap)
            else:
                clipped = np.clip(arr, 0.0, None)
            s_raw += weight * clipped

    # Gaussian smoothing
    # sigma in samples = smoothing_sigma_s / resolution_s (default: 1.5s / 0.5s = 3.0 samples)
    sigma_samples = timeline_cfg.smoothing_sigma_s / res_s
    s_smooth = ndimage.gaussian_filter1d(s_raw, sigma=sigma_samples, mode="nearest").astype(np.float32)

    return TimelineResult(
        timestamps=timestamps,
        features=aligned_features,
        s_raw=s_raw,
        s_smooth=s_smooth,
    )
