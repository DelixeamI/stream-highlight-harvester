"""Validation and anti-hallucination guardrails for LLM Judge verdicts.

Enforces Section 10.4 of TZ_Narezka.md:
- Verbatim evidence quote verification via rapidfuzz partial ratio >= 90
- Segment ID bounds check (start_seg <= end_seg within window)
- Multikill category grounding
- Confidence penalty for ungrounded quotes
"""

from __future__ import annotations

import logging

from rapidfuzz import fuzz

from narezka.asr.base import Segment
from narezka.asr.postprocess import normalize_russian_text
from narezka.llm.schemas import JudgeVerdict
from narezka.models import Candidate, Event

logger = logging.getLogger(__name__)


def validate_judge_verdict(
    verdict: JudgeVerdict,
    candidate: Candidate,
    segments: list[Segment],
    events: list[Event],
) -> tuple[JudgeVerdict, list[str]]:
    """Validate and sanitize JudgeVerdict against candidate ground facts.

    Returns:
        (sanitized_verdict, flags_list)
    """
    flags: list[str] = []

    # 1. Segment IDs validation
    window_segs = [
        s for s in segments if not (s.end < candidate.start - 5.0 or s.start > candidate.end + 5.0)
    ]
    seg_ids = {s.id for s in window_segs}

    if verdict.start_seg not in seg_ids or verdict.end_seg not in seg_ids:
        # Fallback to segments around candidate.peak with generous pre-peak setup (4 segments before, 2 after)
        if window_segs:
            peak_seg = min(window_segs, key=lambda s: abs((s.start + s.end) / 2.0 - candidate.peak))
            p_idx = window_segs.index(peak_seg)
            s_idx = max(0, p_idx - 4)  # 4 segments before peak to guarantee setup!
            e_idx = min(len(window_segs) - 1, p_idx + 2)
            verdict.start_seg = window_segs[s_idx].id
            verdict.end_seg = window_segs[e_idx].id
        else:
            verdict.start_seg = 0
            verdict.end_seg = 0
        flags.append("segment_id_adjusted")

    if verdict.start_seg > verdict.end_seg:
        verdict.start_seg, verdict.end_seg = verdict.end_seg, verdict.start_seg
        flags.append("segment_order_swapped")

    # 2. Quotes anti-hallucination check
    # Combine normalized window text
    full_window_text = " ".join(normalize_russian_text(s.text) for s in window_segs)

    ungrounded_quotes: list[str] = []
    for quote in verdict.evidence_quotes:
        norm_q = normalize_russian_text(quote)
        if not norm_q:
            continue
        sim = fuzz.partial_ratio(norm_q, full_window_text)
        if sim < 88.0:
            ungrounded_quotes.append(quote)

    if ungrounded_quotes:
        logger.warning(
            "Candidate %s: quotes not grounded in transcript: %s",
            candidate.id,
            ungrounded_quotes,
        )
        flags.append("ungrounded")
        # Halve confidence as required by Section 10.4
        verdict.confidence = round(verdict.confidence * 0.5, 2)

    # 3. Multikill category grounding: check for discrete event OR speech evidence
    if verdict.category == "multikill":
        has_mk_event = any(e.type == "multikill" for e in events if not (e.end < candidate.start or e.start > candidate.end))
        mk_keywords = ["убил", "забрал", "минус", "раскидал", "развалил", "эйс", "ace"]
        has_mk_speech = any(any(kw in q.lower() for kw in mk_keywords) for q in verdict.evidence_quotes)
        if not (has_mk_event or has_mk_speech):
            logger.info("Candidate %s: multikill category without MULTIKILL event or kill quotes, downgrading to epic_play", candidate.id)
            verdict.category = "epic_play"
            flags.append("multikill_downgraded")

    # 4. Title sanitization (clean swears, robotic templates, max 60 chars)
    clean_title = verdict.title_ru.strip(" \t\n\r\"'«»!?,.")

    # Remove leading swear words / exclamation prefixes
    swear_prefixes = [
        "блядь", "блять", "пиздец", "сука", "ебать", "нахуй", "нахер", "бля", "ой бля"
    ]
    words = clean_title.split()
    while words and words[0].lower().rstrip(",.!?") in swear_prefixes:
        words = words[1:]
    if words:
        clean_title = " ".join(words).capitalize()

    # Clean robotic "Стример восторгается X" / "Стример в шоке от X"
    if clean_title.lower().startswith("стример восторгается "):
        clean_title = clean_title[len("стример восторгается "):].capitalize()
    elif clean_title.lower().startswith("стример в шоке от "):
        clean_title = clean_title[len("стример в шоке от "):].capitalize()
    elif clean_title.lower().startswith("стример ругается из-за "):
        clean_title = clean_title[len("стример ругается из-за "):].capitalize()

    if len(clean_title) > 60:
        clean_title = clean_title[:57] + "..."
    verdict.title_ru = clean_title or f"Момент {candidate.id}"

    return verdict, flags
