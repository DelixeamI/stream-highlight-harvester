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
            "Candidate %s: quotes not grounded in transcript: %s -> HARD REJECT",
            candidate.id,
            ungrounded_quotes,
        )
        flags.append("ungrounded")
        verdict.verdict = "reject"
        verdict.score = 0.0
        verdict.confidence = 0.0
        verdict.reason_ru = f"Отклонено guardrail: цитаты не найдены в аудио ({', '.join(ungrounded_quotes)})"

    # 3. Streamer speech analysis (anti-AFK, anti-routine, anti-tech-check)
    streamer_segs = [
        s for s in window_segs
        if s.speaker == "streamer" and not getattr(s, "is_game_voice_line", False)
    ]
    streamer_text = " ".join(normalize_russian_text(s.text) for s in streamer_segs).strip()
    streamer_words = streamer_text.split()
    word_count = len(streamer_words)

    tech_keywords = {
        "слышно", "слышишь", "микрофон", "дискорд", "пофиксил", "поправил",
        "настройки", "настроил", "отошел", "отойти", "пауза", "афк", "afk", "проверка", "тест"
    }

    # Anti-tech check: if streamer speaks only about technical issues/mic/settings
    if verdict.verdict == "keep" and word_count > 0:
        tech_words_count = sum(1 for w in streamer_words if any(k in w for k in tech_keywords))
        if tech_words_count >= 2 and (tech_words_count / max(1, word_count)) > 0.40:
            logger.info("Candidate %s: technical/mic check chatter detected (%s) -> REJECT", candidate.id, streamer_text)
            verdict.verdict = "reject"
            verdict.score = 1.0
            verdict.confidence = 0.9
            flags.append("tech_check_rejected")
            verdict.reason_ru = "Отклонено guardrail: технический разговор / проверка микрофона / AFK"

    # Anti-AFK & Category grounding:
    if verdict.verdict == "keep":
        has_clutch = any(e.type == "clutch" for e in events if not (e.end < candidate.start or e.start > candidate.end))
        has_laughter = any(e.type == "laughter" and e.score >= 0.75 for e in events if not (e.end < candidate.start or e.start > candidate.end))

        # Can't have a joke or funny reaction if streamer spoke < 3 words and no laughter
        if verdict.category in ["funny_joke", "funny_reaction"]:
            if word_count < 3 and not has_laughter:
                logger.info("Candidate %s: funny category with no speech (< 3 words) or laughter -> REJECT", candidate.id)
                verdict.verdict = "reject"
                verdict.score = 2.0
                flags.append("afk_funny_rejected")
                verdict.reason_ru = "Отклонено guardrail: нет речи или реакции стримера"

        # Multikill grounding: must have personal combat context
        elif verdict.category == "multikill":
            mk_keywords = ["убил", "забрал", "минус", "раскидал", "развалил", "эйс", "ace", "добил", "помог", "убили", "минусанул", "фраг"]
            has_mk_speech = any(any(kw in w for kw in mk_keywords) for w in streamer_words)
            has_mk_quotes = any(any(kw in q.lower() for kw in mk_keywords) for q in verdict.evidence_quotes)

            # If streamer said nothing or is AFK, match killfeed events are NOT personal multikills!
            if word_count < 3 and not (has_clutch or has_mk_speech):
                logger.info("Candidate %s: multikill event without streamer speech or action -> REJECT (AFK/creep farming)", candidate.id)
                verdict.verdict = "reject"
                verdict.score = 2.0
                flags.append("afk_multikill_rejected")
                verdict.reason_ru = "Отклонено guardrail: стример молчит во время событий киллфида матча"
            elif not (has_mk_speech or has_mk_quotes):
                logger.info("Candidate %s: multikill category without kill speech, downgrading to epic_play", candidate.id)
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
