"""Candidate window generation, trigger detection and interval merging.

Implements peak detection via scipy find_peaks, trigger rules (multikill, sustained laughter, chat spike),
window expansion [-30s, +15s], and 8-second gap merging as specified in TZ S10.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.signal as signal

from narezka.config import AppConfig
from narezka.fusion.timeline import TimelineResult
from narezka.models import Candidate, Event

logger = logging.getLogger(__name__)


def find_timeline_peaks(
    s_smooth: np.ndarray,
    resolution_s: float = 0.5,
    min_distance_s: float = 20.0,
    prominence: float = 0.15,
) -> list[int]:
    """Find peaks in smoothed interest score with minimum distance and prominence."""
    min_dist_samples = max(1, int(round(min_distance_s / resolution_s)))
    peaks, _ = signal.find_peaks(
        s_smooth,
        distance=min_dist_samples,
        prominence=prominence,
    )
    return [int(p) for p in peaks]


def detect_rule_triggers(
    timeline: TimelineResult,
    events: list[Event],
    resolution_s: float = 0.5,
) -> list[tuple[float, float, str, str]]:
    """Detect candidate trigger points independent of S(t) peaks.

    Returns:
        List of (start, end, reason, category_hint) tuples.
    """
    triggers: list[tuple[float, float, str, str]] = []
    features = timeline.features

    # 1. Discrete events (multikill, clutch, lucky_shot, gameplay_fail, high-strength semantic flag)
    for evt in events:
        if evt.type == "multikill":
            # For multikill: from first kill - 20s to last kill + 15s (ensure full setup is included!)
            triggers.append((evt.start - 20.0, evt.end + 15.0, "multikill_event", "multikill"))
        elif evt.type == "kill" and evt.score >= 0.7:
            # Personal kill: generous lead-in setup (18s) and reaction (12s)
            triggers.append((evt.start - 18.0, evt.end + 12.0, "kill_event", "epic_play"))
        elif evt.type == "epic_play":
            triggers.append((evt.start - 20.0, evt.end + 15.0, "epic_play_event", "epic_play"))
        elif evt.type == "clutch":
            triggers.append((evt.start - 25.0, evt.end + 15.0, "clutch_event", "epic_play"))
        elif evt.type == "lucky_shot":
            triggers.append((evt.start - 20.0, evt.end + 15.0, "lucky_shot_event", "epic_play"))
        elif evt.type == "gameplay_fail":
            triggers.append((evt.start - 20.0, evt.end + 15.0, "gameplay_fail_event", "fail"))
        elif evt.type == "semantic_flag" and evt.score >= 0.5:
            triggers.append((evt.start - 20.0, evt.end + 15.0, "semantic_flag", "funny_joke"))

    # 2. Continuous laughter trigger: laugh_p > 0.7 for > 1.5s (3 consecutive frames at 0.5s)
    laugh_p = features.get("laugh_p")
    if laugh_p is not None:
        count = 0
        l_start = 0.0
        for i, val in enumerate(laugh_p):
            t = i * resolution_s
            if val > 0.7:
                if count == 0:
                    l_start = t
                count += 1
            else:
                if count >= 3:
                    triggers.append((l_start - 25.0, t + 15.0, "sustained_laughter", "funny_reaction"))
                count = 0
        if count >= 3:
            t_end = len(laugh_p) * resolution_s
            triggers.append((l_start - 25.0, t_end + 15.0, "sustained_laughter", "funny_reaction"))

    # 3. Continuous chat rate spike trigger: chat_rate_z > 4.0
    chat_rate_z = features.get("chat_rate_z")
    if chat_rate_z is not None:
        count = 0
        c_start = 0.0
        for i, val in enumerate(chat_rate_z):
            t = i * resolution_s
            if val > 4.0:
                if count == 0:
                    c_start = t
                count += 1
            else:
                if count >= 2:
                    triggers.append((c_start - 25.0, t + 15.0, "chat_spike", "funny_reaction"))
                count = 0
        if count >= 2:
            t_end = len(chat_rate_z) * resolution_s
            triggers.append((c_start - 25.0, t_end + 15.0, "chat_spike", "funny_reaction"))

    return triggers


def merge_candidate_intervals(
    intervals: list[tuple[float, float, str]],
    gap_merge_s: float = 8.0,
    max_window_s: float = 75.0,
    duration_s: float = 0.0,
) -> list[tuple[float, float, list[str]]]:
    """Merge overlapping intervals and intervals separated by less than gap_merge_s up to max_window_s."""
    if not intervals:
        return []

    # Clamp to [0, duration_s]
    clamped: list[tuple[float, float, str]] = []
    for s, e, reason in intervals:
        s_c = max(0.0, s)
        e_c = min(duration_s, e) if duration_s > 0 else max(s_c + 1.0, e)
        if e_c > s_c:
            clamped.append((s_c, e_c, reason))

    # Sort by start time
    clamped.sort(key=lambda x: x[0])

    merged: list[tuple[float, float, list[str]]] = []
    cur_start, cur_end, cur_reason = clamped[0]
    cur_reasons = [cur_reason]

    for next_start, next_end, next_reason in clamped[1:]:
        can_merge = (next_start <= cur_end + gap_merge_s) and ((max(cur_end, next_end) - cur_start) <= max_window_s)
        if can_merge:
            # Merge
            cur_end = max(cur_end, next_end)
            if next_reason not in cur_reasons:
                cur_reasons.append(next_reason)
        else:
            merged.append((cur_start, cur_end, cur_reasons))
            cur_start, cur_end, cur_reason = next_start, next_end, next_reason
            cur_reasons = [cur_reason]

    merged.append((cur_start, cur_end, cur_reasons))
    return merged


def infer_suspected_categories(
    features_summary: dict[str, float],
    events: list[Event],
    reasons: list[str],
) -> list[str]:
    """Infer likely categories for candidate highlight."""
    categories: list[str] = []

    # 1. Multikill
    if any(e.type == "multikill" for e in events) or "multikill_event" in reasons:
        categories.append("multikill")
        categories.append("epic_play")

    # 1b. Clutch, Lucky shot, Kill & Epic play
    if (
        any(e.type in ["clutch", "lucky_shot", "kill", "epic_play"] for e in events)
        or any(r in ["clutch_event", "lucky_shot_event", "kill_event", "epic_play_event"] for r in reasons)
    ):
        if "epic_play" not in categories:
            categories.append("epic_play")

    # 1c. Gameplay Fail
    if any(e.type == "gameplay_fail" for e in events) or "gameplay_fail_event" in reasons:
        if "fail" not in categories:
            categories.append("fail")

    # 2. Laughter / Jokes
    laugh_p = features_summary.get("laugh_p", 0.0)
    chat_laugh = features_summary.get("chat_laugh", 0.0)
    if laugh_p >= 0.55 or chat_laugh >= 0.35 or "sustained_laughter" in reasons:
        if "funny_reaction" not in categories:
            categories.append("funny_reaction")
        if "funny_joke" not in categories:
            categories.append("funny_joke")

    # 3. Screams / Loud bursts / Fails
    scream_p = features_summary.get("scream_p", 0.0)
    loud_z = features_summary.get("loud_z", 0.0)
    if scream_p >= 0.5 or loud_z >= 3.5:
        if "funny_reaction" not in categories:
            categories.append("funny_reaction")
        if "fail" not in categories:
            categories.append("fail")

    # 4. Reaction lexicon / hype
    hype_lex = features_summary.get("hype_lex", 0.0)
    chat_hype = features_summary.get("chat_hype", 0.0)
    if hype_lex >= 0.6 or chat_hype >= 0.3:
        if "epic_play" not in categories:
            categories.append("epic_play")
        if "funny_reaction" not in categories:
            categories.append("funny_reaction")

    if not categories:
        categories.append("other")

    return categories


def generate_candidates(
    timeline: TimelineResult,
    events: list[Event],
    config: AppConfig,
    duration_s: float,
) -> list[Candidate]:
    """Generate and rank candidate windows from S(t) peaks and trigger rules.

    Args:
        timeline: Unified TimelineResult.
        events: List of all detected events (audio, chat, vision, lexicon).
        config: AppConfig.
        duration_s: Total stream duration in seconds.

    Returns:
        List of Candidate objects selected for judge.
    """
    res_s = config.timeline.resolution_s
    s_smooth = timeline.s_smooth

    # 1. Collect initial intervals from peaks: [peak - 30s, peak + 15s]
    peak_indices = find_timeline_peaks(s_smooth, resolution_s=res_s, min_distance_s=20.0, prominence=0.12)
    raw_intervals: list[tuple[float, float, str]] = []

    for p_idx in peak_indices:
        t_peak = p_idx * res_s
        raw_intervals.append((t_peak - 30.0, t_peak + 15.0, f"peak_S={s_smooth[p_idx]:.2f}"))

    # 2. Add trigger intervals
    trigger_intervals = detect_rule_triggers(timeline, events, resolution_s=res_s)
    for s, e, reason, _ in trigger_intervals:
        raw_intervals.append((s, e, reason))

    if not raw_intervals:
        return []

    # 3. Merge overlapping and nearby intervals with strict maximum window limit (75s)
    merged_intervals = merge_candidate_intervals(
        raw_intervals, gap_merge_s=8.0, max_window_s=75.0, duration_s=duration_s
    )

    # 4. Build Candidate objects
    candidates: list[Candidate] = []

    for i, (start_t, end_t, reasons) in enumerate(merged_intervals):
        cid = f"c_{i+1:04d}"

        # Find peak within [start_t, end_t]
        start_idx = max(0, int(np.floor(start_t / res_s)))
        end_idx = min(len(s_smooth), int(np.ceil(end_t / res_s)))

        if end_idx > start_idx:
            sub_s = s_smooth[start_idx:end_idx]
            local_max_idx = int(np.argmax(sub_s))
            peak_t = (start_idx + local_max_idx) * res_s
            signal_score = float(sub_s[local_max_idx])
        else:
            peak_t = (start_t + end_t) / 2.0
            signal_score = 0.5

        # Feature summary (max value of each feature in window)
        features_summary: dict[str, float] = {}
        for feat_name, arr in timeline.features.items():
            if len(arr) > 0 and end_idx > start_idx:
                sub_arr = arr[start_idx:end_idx]
                features_summary[feat_name] = round(float(np.max(sub_arr)), 3)
            else:
                features_summary[feat_name] = 0.0

        # Events overlapping window
        matching_events = [
            e for e in events if not (e.end < start_t or e.start > end_t)
        ]
        event_ids = [e.id for e in matching_events]

        suspected = infer_suspected_categories(features_summary, matching_events, reasons)

        candidates.append(
            Candidate(
                id=cid,
                start=round(start_t, 2),
                end=round(end_t, 2),
                peak=round(peak_t, 2),
                signal_score=round(signal_score, 3),
                suspected_categories=suspected,
                event_ids=event_ids,
                features_summary=features_summary,
            )
        )

    # 5. Selection for judge: Top-K by signal_score + all trigger candidates
    # Ensure at least 60 candidates are evaluated by judge
    max_k = min(max(60, 3 * config.output.top_n), config.judge.max_candidates)

    # Sort descending by signal score
    candidates.sort(key=lambda c: c.signal_score, reverse=True)

    # Always prioritize candidates with gameplay triggers (multikill, clutch, lucky shot, fail) or strong laughter
    must_include = []
    others = []
    for c in candidates:
        has_gameplay = (
            "multikill" in c.suspected_categories
            or any(cat in c.suspected_categories for cat in ["epic_play", "fail"])
            or c.features_summary.get("laugh_p", 0.0) >= 0.65
        )
        if has_gameplay:
            must_include.append(c)
        else:
            others.append(c)

    selected = must_include[:max_k]
    remaining_slots = max_k - len(selected)
    if remaining_slots > 0:
        selected.extend(others[:remaining_slots])

    # Re-sort chronologically for downstream ease
    selected.sort(key=lambda c: c.start)
    for idx, c in enumerate(selected):
        c.id = f"c_{idx+1:04d}"

    logger.info("Generated %d candidate windows (selected %d for judge)", len(candidates), len(selected))
    return selected
