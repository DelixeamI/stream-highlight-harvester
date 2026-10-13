"""Moment ranking, Non-Maximum Suppression (NMS), and category diversity selection."""

from __future__ import annotations

import logging
from typing import Any

from narezka.asr.base import Segment
from narezka.clips.boundaries import refine_clip_boundaries, snap_to_vad_pause
from narezka.config import OutputConfig
from narezka.models import Candidate, Event, Moment

logger = logging.getLogger(__name__)


def calculate_overlap_ratio(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap ratio relative to the shorter interval."""
    overlap = max(0.0, min(end1, end2) - max(start1, start2))
    len1 = max(1e-4, end1 - start1)
    len2 = max(1e-4, end2 - start2)
    return overlap / min(len1, len2)


def apply_nms(moments: list[Moment], overlap_threshold: float = 0.40) -> list[Moment]:
    """Suppress moments that overlap more than threshold with a higher-scoring moment."""
    # Sort descending by final score
    sorted_moments = sorted(moments, key=lambda m: m.final_score, reverse=True)
    kept: list[Moment] = []

    for m in sorted_moments:
        suppress = False
        for k in kept:
            ratio = calculate_overlap_ratio(m.start, m.end, k.start, k.end)
            if ratio >= overlap_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(m)

    return kept


def apply_category_diversity(
    moments: list[Moment],
    top_n: int = 20,
    max_share_per_category: float = 0.50,
) -> list[Moment]:
    """Select top_n moments while respecting maximum share per category limit."""
    max_per_cat = max(1, int(round(top_n * max_share_per_category)))
    cat_counts: dict[str, int] = {}
    selected: list[Moment] = []

    # Moments are assumed to be sorted by final_score descending
    for m in moments:
        c = m.category
        count = cat_counts.get(c, 0)
        if count < max_per_cat:
            selected.append(m)
            cat_counts[c] = count + 1
        if len(selected) >= top_n:
            break

    # If slots remain and we haven't reached top_n, backfill from remaining
    if len(selected) < top_n and len(moments) > len(selected):
        selected_ids = {m.id for m in selected}
        for m in moments:
            if m.id not in selected_ids:
                selected.append(m)
                if len(selected) >= top_n:
                    break

    return selected


def format_time_code(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def generate_rule_based_title_and_reason(
    category: str,
    peak_t: float,
    quotes: list[str],
    features_summary: dict[str, float],
) -> tuple[str, str]:
    """Generate title in Russian and explanatory reason without LLM judge."""
    tc = format_time_code(peak_t)
    first_quote = quotes[0] if quotes else ""

    if category == "multikill":
        title = f"Мультикилл стримера ({tc})"
        reason = "Серия быстрых убийств стримером с яркой реакцией."
    elif category == "funny_reaction":
        if first_quote and len(first_quote) < 35:
            title = f"«{first_quote}» ({tc})"
        else:
            title = f"Бурная реакция стримера ({tc})"
        reason = f"Пик смеха/громкости: смех p={features_summary.get('laugh_p', 0.0):.2f}, loud_z={features_summary.get('loud_z', 0.0):.1f}."
    elif category == "funny_joke":
        if first_quote and len(first_quote) < 35:
            title = f"Шутка: «{first_quote}»"
        else:
            title = f"Шутка стримера ({tc})"
        reason = "Юмористическая фраза и реакция зрителей в чате."
    elif category == "epic_play":
        title = f"Красивый игровой момент ({tc})"
        reason = "Высокая эмоциональная насыщенность и активные действия в матче."
    elif category == "fail":
        title = f"Эпичный фейл ({tc})"
        reason = "Неожиданная ошибка или смерть с бурным криком/эмоцией."
    else:
        title = f"Интересный фрагмент ({tc})"
        reason = "Заметный всплеск интереса на графике S(t)."

    return title, reason


def build_rule_based_moments(
    candidates: list[Candidate],
    segments: list[Segment],
    events: list[Event],
    output_cfg: OutputConfig,
    duration_s: float,
    vad_intervals: list[tuple[float, float]] | None = None,
) -> list[Moment]:
    """Build, rank, deduplicate and finalize highlight moments without LLM judge."""
    moments_raw: list[Moment] = []

    for i, cand in enumerate(candidates):
        # Best category
        cat = cand.suspected_categories[0] if cand.suspected_categories else "other"
        if "multikill" in cand.suspected_categories:
            cat = "multikill"
        elif "epic_play" in cand.suspected_categories and cat not in ["multikill", "funny_reaction"]:
            cat = "epic_play"
        if cat not in ["multikill", "epic_play", "funny_joke", "funny_reaction", "fail", "other"]:
            cat = "other"

        # Quotes from transcript
        window_segs = [
            s for s in segments if not (s.end < cand.start or s.start > cand.end)
        ]
        quotes = [s.text.strip() for s in window_segs if s.text.strip() and s.speaker != "game"][:3]

        # Boundaries
        start_t, end_t = refine_clip_boundaries(
            candidate=cand,
            category=cat,
            segments=segments,
            output_cfg=output_cfg,
            duration_s=duration_s,
            vad_intervals=vad_intervals,
        )

        title, reason = generate_rule_based_title_and_reason(
            category=cat,
            peak_t=cand.peak,
            quotes=quotes,
            features_summary=cand.features_summary,
        )

        # Signal score to judge score 0..10
        # Typical S(t) peak is between 0.3 and 1.5
        norm_score = min(1.0, cand.signal_score / 1.2)
        judge_score = max(5, min(10, int(round(norm_score * 10))))
        final_score = round(norm_score, 3)

        evidence = {
            "quotes": quotes,
            "features_summary": cand.features_summary,
            "event_ids": cand.event_ids,
            "peak_s_score": cand.signal_score,
        }

        mid = f"m_{i+1:04d}"
        moments_raw.append(
            Moment(
                id=mid,
                candidate_id=cand.id,
                category=cat,  # type: ignore
                title=title,
                reason=reason,
                start=start_t,
                end=end_t,
                peak=cand.peak,
                judge_score=judge_score,
                confidence=0.85,
                final_score=final_score,
                evidence=evidence,
                flags=["rule_based_draft"],
            )
        )

    # Apply NMS (40% overlap suppression)
    non_overlapping = apply_nms(moments_raw, overlap_threshold=0.40)

    # Filter by minimum score
    filtered = [m for m in non_overlapping if m.final_score >= output_cfg.min_score]
    if not filtered and non_overlapping:
        # Keep at least top 3 if all were below min_score
        filtered = non_overlapping[:3]

    # Apply category diversity
    top_moments = apply_category_diversity(
        moments=filtered,
        top_n=output_cfg.top_n,
        max_share_per_category=output_cfg.max_share_per_category,
    )

    # Re-sort chronologically for report
    top_moments.sort(key=lambda m: m.start)
    for idx, m in enumerate(top_moments):
        m.id = f"m_{idx+1:04d}"

    logger.info("Built %d final moments from %d candidates", len(top_moments), len(candidates))
    return top_moments


def build_moments_with_judge(
    candidates: list[Candidate],
    verdicts: dict[str, tuple[Any, list[str]]],
    segments: list[Segment],
    output_cfg: OutputConfig,
    duration_s: float,
    vad_intervals: list[tuple[float, float]] | None = None,
) -> list[Moment]:
    """Build, refine boundaries, deduplicate with NMS and rank moments using LLM Judge verdicts."""
    seg_by_id = {s.id: s for s in segments}
    alpha = output_cfg.alpha_judge
    pad_before = output_cfg.pad_before_s
    pad_after = output_cfg.pad_after_s

    moments_raw: list[Moment] = []

    for i, cand in enumerate(candidates):
        if cand.id not in verdicts:
            continue
        verdict, flags = verdicts[cand.id]
        if verdict.verdict != "keep":
            continue

        cat = verdict.category
        title = verdict.title_ru
        reason = verdict.reason_ru
        judge_score = verdict.score
        confidence = verdict.confidence

        # Determine boundaries from start_seg and end_seg
        s_seg = seg_by_id.get(verdict.start_seg)
        e_seg = seg_by_id.get(verdict.end_seg)

        # Safety check: if s_seg or e_seg are unreasonably far from cand.peak (> 50s),
        # they do not represent the climax of this candidate! Anchor to cand.peak.
        if s_seg and abs(s_seg.start - cand.peak) > 50.0:
            s_seg = None
        if e_seg and abs(e_seg.end - cand.peak) > 50.0:
            e_seg = None

        # Determine lead-in and reaction targets
        # In esports/gaming highlights, action occurs 15-25 seconds BEFORE the emotional climax!
        min_lead_in = 20.0 if cat in ["multikill", "epic_play", "fail"] else 15.0

        if s_seg:
            if s_seg.start <= cand.peak - min_lead_in:
                target_start = s_seg.start - pad_before
            else:
                # s_seg is too close to or after the peak (reaction speech).
                # Guarantee adequate pre-peak lead-in for context:
                target_start = min(s_seg.start - pad_before, cand.peak - min_lead_in)
        else:
            target_start = cand.peak - min_lead_in

        # Also respect candidate window start
        target_start = min(target_start, cand.start)

        if e_seg:
            target_end = max(e_seg.end + pad_after, cand.peak + 10.0)
        else:
            target_end = cand.peak + 15.0

        # Enforce that cand.peak is ALWAYS securely inside the clip
        if cand.peak < target_start + 12.0:
            target_start = max(0.0, cand.peak - min_lead_in)
        if cand.peak > target_end - 5.0:
            target_end = cand.peak + 10.0

        # Snap to VAD pause
        if vad_intervals:
            target_start = snap_to_vad_pause(target_start, vad_intervals, search_radius_s=3.0, prefer_direction="before")
            target_end = snap_to_vad_pause(target_end, vad_intervals, search_radius_s=3.0, prefer_direction="after")

        # Category limits
        cat_limits = output_cfg.clip_length_s.get(cat, [15, 60])
        min_len = max(8.0, float(cat_limits[0]))
        max_len = min(90.0, float(cat_limits[1]))

        dur = target_end - target_start
        if dur < min_len:
            diff = min_len - dur
            # Expand 70% backwards into setup, 30% forwards
            target_start = max(0.0, target_start - diff * 0.7)
            target_end = target_end + diff * 0.3
        elif dur > max_len:
            # Anchor trimming around cand.peak: 65% lead-in, 35% reaction
            ideal_start = cand.peak - max_len * 0.65
            ideal_end = cand.peak + max_len * 0.35
            target_start = max(target_start, ideal_start)
            target_end = min(target_end, ideal_end)
            if target_end - target_start > max_len:
                target_start = target_end - max_len

        target_start = max(0.0, target_start)
        if duration_s > 0.0:
            target_end = min(duration_s, target_end)

        # Final score
        norm_signal = min(1.0, cand.signal_score / 1.2)
        norm_judge = judge_score / 10.0
        final_score = round(alpha * norm_judge + (1.0 - alpha) * norm_signal, 3)

        evidence = {
            "quotes": verdict.evidence_quotes,
            "features_summary": cand.features_summary,
            "event_ids": cand.event_ids,
            "peak_s_score": cand.signal_score,
            "judge_score": judge_score,
        }

        mid = f"m_{i+1:04d}"
        moments_raw.append(
            Moment(
                id=mid,
                candidate_id=cand.id,
                category=cat,
                title=title,
                reason=reason,
                start=round(target_start, 2),
                end=round(target_end, 2),
                peak=cand.peak,
                judge_score=judge_score,
                confidence=confidence,
                final_score=final_score,
                evidence=evidence,
                flags=list(flags),
            )
        )

    # Apply NMS (40% overlap suppression)
    non_overlapping = apply_nms(moments_raw, overlap_threshold=0.40)

    # Filter by minimum score
    filtered = [m for m in non_overlapping if m.final_score >= output_cfg.min_score]
    if not filtered and non_overlapping:
        filtered = non_overlapping[:3]

    # Apply category diversity
    top_moments = apply_category_diversity(
        moments=filtered,
        top_n=output_cfg.top_n,
        max_share_per_category=output_cfg.max_share_per_category,
    )

    top_moments.sort(key=lambda m: m.start)
    for idx, m in enumerate(top_moments):
        m.id = f"m_{idx+1:04d}"

    logger.info("Built %d final moments using Judge verdicts", len(top_moments))
    return top_moments
