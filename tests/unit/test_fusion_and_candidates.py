"""Unit tests for Stage S10: Timeline fusion and candidate generation."""

from __future__ import annotations

import numpy as np

from narezka.config import AppConfig, FeaturesConfig, TimelineConfig
from narezka.fusion.candidates import (
    detect_rule_triggers,
    find_timeline_peaks,
    generate_candidates,
    merge_candidate_intervals,
)
from narezka.fusion.timeline import build_unified_timeline
from narezka.models import Event


def test_build_unified_timeline_weighted_sum_and_smoothing():
    num_frames = 100
    res_s = 0.5
    duration_s = 50.0

    loud_z = np.zeros(num_frames, dtype=np.float32)
    laugh_p = np.zeros(num_frames, dtype=np.float32)
    # Spike at frame 50 (t = 25s)
    loud_z[50] = 4.0
    laugh_p[50] = 0.8

    features = {"loud_z": loud_z, "laugh_p": laugh_p}

    t_cfg = TimelineConfig(resolution_s=res_s, smoothing_sigma_s=1.5)
    f_cfg = FeaturesConfig(
        weights={"loud_z": 0.20, "laugh_p": 0.25},
        caps={"loud_z": 5.0, "laugh_p": 1.0},
    )

    res = build_unified_timeline(features, duration_s=duration_s, timeline_cfg=t_cfg, features_cfg=f_cfg)
    assert len(res.timestamps) == num_frames
    # Raw score at frame 50: 0.20 * 4.0 + 0.25 * 0.8 = 0.8 + 0.2 = 1.0
    assert abs(res.s_raw[50] - 1.0) < 1e-4

    # Gaussian smoothed curve should spread energy to neighboring frames
    assert res.s_smooth[50] > 0.1
    assert res.s_smooth[50] > res.s_smooth[45]
    assert res.s_smooth[49] > 0.08
    assert res.s_smooth[51] > 0.08


def test_find_timeline_peaks():
    res_s = 0.5
    s_smooth = np.zeros(200, dtype=np.float32)
    # Peak at t=20s (frame 40)
    s_smooth[40] = 1.2
    # Second peak at t=25s (frame 50) - distance is only 5s (< 20s min distance)
    s_smooth[50] = 0.9
    # Third peak at t=60s (frame 120) - distance 35s (> 20s)
    s_smooth[120] = 1.5

    peaks = find_timeline_peaks(s_smooth, resolution_s=res_s, min_distance_s=20.0, prominence=0.15)
    # The peak at frame 50 should be suppressed because it is too close to frame 40
    assert 40 in peaks
    assert 120 in peaks
    assert 50 not in peaks


def test_merge_candidate_intervals_gap():
    # Interval A: [10, 20], Interval B: [25, 35] -> gap is 5s (< 8s) -> should be merged to [10, 35]
    # Interval C: [50, 60] -> gap is 15s (> 8s) -> separate
    raw = [
        (10.0, 20.0, "reason_a"),
        (25.0, 35.0, "reason_b"),
        (50.0, 60.0, "reason_c"),
    ]

    merged = merge_candidate_intervals(raw, gap_merge_s=8.0, duration_s=100.0)
    assert len(merged) == 2
    assert merged[0][0] == 10.0
    assert merged[0][1] == 35.0
    assert "reason_a" in merged[0][2]
    assert "reason_b" in merged[0][2]

    assert merged[1][0] == 50.0
    assert merged[1][1] == 60.0


def test_detect_rule_triggers_multikill_and_laughter():
    res_s = 0.5
    num_frames = 60
    duration_s = 30.0

    loud_z = np.zeros(num_frames, dtype=np.float32)
    laugh_p = np.zeros(num_frames, dtype=np.float32)
    # Laughter lasting 2 seconds (4 frames) from frame 10 to 14
    laugh_p[10:14] = 0.85

    t_cfg = TimelineConfig(resolution_s=res_s, smoothing_sigma_s=1.5)
    f_cfg = FeaturesConfig()
    timeline = build_unified_timeline(
        {"loud_z": loud_z, "laugh_p": laugh_p},
        duration_s=duration_s,
        timeline_cfg=t_cfg,
        features_cfg=f_cfg,
    )

    mk_event = Event(
        id="evt_mk_1",
        source="killfeed",
        type="multikill",
        start=15.0,
        end=20.0,
        score=0.9,
    )

    triggers = detect_rule_triggers(timeline, [mk_event], resolution_s=res_s)
    # Should detect multikill and sustained laughter triggers
    reasons = [t[2] for t in triggers]
    assert "multikill_event" in reasons
    assert "sustained_laughter" in reasons


def test_generate_candidates_full_pipeline():
    cfg = AppConfig()
    num_frames = 200
    duration_s = 100.0

    loud_z = np.zeros(num_frames, dtype=np.float32)
    laugh_p = np.zeros(num_frames, dtype=np.float32)
    # High interest spike at t = 50s (frame 100)
    loud_z[98:103] = 4.5
    laugh_p[98:103] = 0.8

    timeline = build_unified_timeline(
        {"loud_z": loud_z, "laugh_p": laugh_p},
        duration_s=duration_s,
        timeline_cfg=cfg.timeline,
        features_cfg=cfg.features,
    )

    events = [
        Event(
            id="evt_1",
            source="audio_tagger",
            type="laughter",
            start=49.0,
            end=52.0,
            score=0.85,
        )
    ]

    candidates = generate_candidates(timeline, events, cfg, duration_s=duration_s)
    assert len(candidates) >= 1
    c = candidates[0]
    # Candidate should cover the peak around 50s
    assert c.start <= 50.0 <= c.end
    assert "funny_reaction" in c.suspected_categories
    assert "evt_1" in c.event_ids
    assert c.features_summary.get("loud_z", 0) > 3.0
