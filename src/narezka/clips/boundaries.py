"""Rule-based boundary refinement and pause snapping.

Refines clip start and end timestamps using VAD speech pauses (+/-2s),
speech segment boundaries, and category length constraints as specified in TZ S12.
"""

from __future__ import annotations

import logging

from narezka.asr.base import Segment
from narezka.config import OutputConfig
from narezka.models import Candidate

logger = logging.getLogger(__name__)


def snap_to_vad_pause(
    target_t: float,
    vad_intervals: list[tuple[float, float]],
    search_radius_s: float = 2.0,
    prefer_direction: str = "before",  # "before" for start, "after" for end
) -> float:
    """Snap timestamp to nearest silence/pause gap between speech intervals within search_radius."""
    if not vad_intervals:
        return target_t

    # A pause occurs between vad_intervals[i].end and vad_intervals[i+1].start
    # or before the first interval / after the last interval
    pauses: list[tuple[float, float]] = []

    # Before first speech
    if vad_intervals[0][0] > 0.0:
        pauses.append((0.0, vad_intervals[0][0]))

    # Between intervals
    for i in range(len(vad_intervals) - 1):
        p_start = vad_intervals[i][1]
        p_end = vad_intervals[i + 1][0]
        if p_end > p_start:
            pauses.append((p_start, p_end))

    # Search for pauses within radius
    candidates: list[float] = []
    for p_start, p_end in pauses:
        # Check if pause overlaps [target_t - search_radius_s, target_t + search_radius_s]
        if not (p_end < target_t - search_radius_s or p_start > target_t + search_radius_s):
            # Best point in pause
            mid_p = (p_start + p_end) / 2.0
            candidates.append(mid_p)

    if not candidates:
        return target_t

    # Select closest candidate
    if prefer_direction == "before":
        # Prefer slightly before
        candidates.sort(key=lambda c: (abs(c - target_t), c > target_t))
    else:
        candidates.sort(key=lambda c: (abs(c - target_t), c < target_t))

    return round(candidates[0], 2)


def refine_clip_boundaries(
    candidate: Candidate,
    category: str,
    segments: list[Segment],
    output_cfg: OutputConfig,
    duration_s: float,
    vad_intervals: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Refine start and end boundaries for candidate window respecting category limits and speech."""
    pad_before = output_cfg.pad_before_s
    pad_after = output_cfg.pad_after_s

    # Category limits
    cat_limits = output_cfg.clip_length_s.get(category, [10, 60])
    min_len = float(cat_limits[0])
    max_len = float(cat_limits[1])

    # Absolute safety limits
    abs_min = 6.0
    abs_max = 90.0
    min_len = max(abs_min, min_len)
    max_len = min(abs_max, max_len)

    # Initial start around peak - half_target or candidate start
    target_len = (min_len + max_len) / 2.0

    # Search for speech segments around peak
    segs_in_window = [
        s for s in segments if not (s.end < candidate.start - 5.0 or s.start > candidate.end + 5.0)
    ]

    start_t = candidate.peak - (target_len * 0.65)
    end_t = candidate.peak + (target_len * 0.35)

    # If speech segments exist, align start to speech phrase
    if segs_in_window:
        # Find segment closest to start_t
        lead_in_segs = [s for s in segs_in_window if s.start <= candidate.peak]
        if lead_in_segs:
            # Take start of lead-in segment minus pad_before
            best_lead = min(lead_in_segs, key=lambda s: abs(s.start - start_t))
            start_t = min(start_t, best_lead.start - pad_before)

        # Find segment after reaction
        react_segs = [s for s in segs_in_window if s.end >= candidate.peak]
        if react_segs:
            best_react = min(react_segs, key=lambda s: abs(s.end - end_t))
            end_t = max(end_t, best_react.end + pad_after)

    # Ensure candidate.peak is securely inside clip
    if candidate.peak < start_t + 12.0:
        start_t = max(0.0, candidate.peak - (target_len * 0.65))
    if candidate.peak > end_t - 5.0:
        end_t = candidate.peak + 10.0

    # Snap to VAD pause if provided
    if vad_intervals:
        start_t = snap_to_vad_pause(start_t, vad_intervals, search_radius_s=3.0, prefer_direction="before")
        end_t = snap_to_vad_pause(end_t, vad_intervals, search_radius_s=3.0, prefer_direction="after")

    # Enforce minimum and maximum duration constraints
    dur = end_t - start_t
    if dur < min_len:
        diff = min_len - dur
        start_t = max(0.0, start_t - diff * 0.7)
        end_t = end_t + diff * 0.3
    elif dur > max_len:
        ideal_start = candidate.peak - max_len * 0.65
        ideal_end = candidate.peak + max_len * 0.35
        start_t = max(start_t, ideal_start)
        end_t = min(end_t, ideal_end)
        if end_t - start_t > max_len:
            start_t = end_t - max_len

    # Clamp to VOD duration
    start_t = max(0.0, start_t)
    if duration_s > 0.0:
        end_t = min(duration_s, end_t)

    # Re-verify minimum length
    if (end_t - start_t) < abs_min and duration_s >= abs_min:
        if start_t + abs_min <= duration_s:
            end_t = start_t + abs_min
        else:
            start_t = max(0.0, end_t - abs_min)

    return round(start_t, 2), round(end_t, 2)
