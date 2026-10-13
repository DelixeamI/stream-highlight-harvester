"""Unit tests for Stage S12: Boundary refinement, NMS, diversity and moment creation."""

from __future__ import annotations

from narezka.clips.boundaries import refine_clip_boundaries, snap_to_vad_pause
from narezka.clips.ranking import (
    apply_category_diversity,
    apply_nms,
    calculate_overlap_ratio,
)
from narezka.config import OutputConfig
from narezka.models import Candidate, Moment


def test_snap_to_vad_pause():
    # Speech intervals: [0..10] and [12..20].
    # There is a 2-second pause between 10.0 and 12.0 (midpoint 11.0).
    vad_intervals = [(0.0, 10.0), (12.0, 20.0)]

    # Target timestamp at 10.5 should snap to pause around 11.0
    snapped = snap_to_vad_pause(10.5, vad_intervals, search_radius_s=2.0)
    assert abs(snapped - 11.0) < 0.2

    # Target timestamp far from pause (e.g. at 5.0) should remain unchanged
    unchanged = snap_to_vad_pause(5.0, vad_intervals, search_radius_s=1.0)
    assert unchanged == 5.0


def test_refine_clip_boundaries_category_limits():
    cfg = OutputConfig()
    # Candidate for funny_reaction: limits [8, 30]
    cand = Candidate(
        id="c_001",
        start=40.0,
        end=55.0,
        peak=48.0,
        signal_score=0.9,
        suspected_categories=["funny_reaction"],
    )

    s_t, e_t = refine_clip_boundaries(
        candidate=cand,
        category="funny_reaction",
        segments=[],
        output_cfg=cfg,
        duration_s=100.0,
    )

    duration = e_t - s_t
    cat_limits = cfg.clip_length_s.get("funny_reaction", [8, 50])
    assert cat_limits[0] <= duration <= cat_limits[1]
    # Peak must be contained in clip
    assert s_t <= cand.peak <= e_t


def test_calculate_overlap_ratio():
    # Moment 1: [10, 30] (len 20), Moment 2: [20, 40] (len 20)
    # Overlap is [20, 30] = 10s. Overlap ratio = 10 / 20 = 0.5 (50%)
    ratio = calculate_overlap_ratio(10.0, 30.0, 20.0, 40.0)
    assert abs(ratio - 0.5) < 1e-4

    # Non-overlapping
    assert calculate_overlap_ratio(10.0, 20.0, 25.0, 35.0) == 0.0


def test_apply_nms():
    # Two moments overlapping by 60%
    m1 = Moment(
        id="m_1",
        candidate_id="c_1",
        category="funny_reaction",
        title="Reaction 1",
        reason="",
        start=10.0,
        end=30.0,
        peak=20.0,
        judge_score=9,
        confidence=0.9,
        final_score=0.85,
    )
    m2 = Moment(
        id="m_2",
        candidate_id="c_2",
        category="funny_reaction",
        title="Reaction 2",
        reason="",
        start=12.0,
        end=32.0,
        peak=21.0,
        judge_score=7,
        confidence=0.8,
        final_score=0.65,
    )

    kept = apply_nms([m1, m2], overlap_threshold=0.40)
    assert len(kept) == 1
    assert kept[0].id == "m_1"  # Higher score kept


def test_apply_category_diversity():
    # 5 moments of category 'funny_reaction' and 3 moments of 'fail'
    # With top_n = 4 and max_share = 0.5, at most 2 funny_reaction should be picked
    moments = []
    for i in range(5):
        moments.append(
            Moment(
                id=f"m_r{i}",
                candidate_id=f"c_r{i}",
                category="funny_reaction",
                title="Reaction",
                reason="",
                start=float(i * 10),
                end=float(i * 10 + 8),
                peak=float(i * 10 + 4),
                judge_score=8,
                confidence=0.8,
                final_score=0.90 - i * 0.01,
            )
        )
    for i in range(3):
        moments.append(
            Moment(
                id=f"m_f{i}",
                candidate_id=f"c_f{i}",
                category="fail",
                title="Fail",
                reason="",
                start=float(100 + i * 10),
                end=float(100 + i * 10 + 8),
                peak=float(100 + i * 10 + 4),
                judge_score=7,
                confidence=0.8,
                final_score=0.70 - i * 0.01,
            )
        )

    selected = apply_category_diversity(moments, top_n=4, max_share_per_category=0.50)
    assert len(selected) == 4
    reaction_count = sum(1 for m in selected if m.category == "funny_reaction")
    assert reaction_count == 2
