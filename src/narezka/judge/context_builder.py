"""Formats candidate context into rich prompt for LLM Judge.

Conforms strictly to Section 10.2 of TZ_Narezka.md.
"""

from __future__ import annotations

from typing import Any

from narezka.asr.base import Segment
from narezka.models import Candidate, Event


def format_seconds_time(seconds: float) -> str:
    """Format seconds float into HH:MM:SS.s."""
    s = int(seconds)
    fraction = int(round((seconds - s) * 10))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{fraction}"


def build_candidate_prompt_context(
    candidate: Candidate,
    segments: list[Segment],
    events: list[Event],
    game_name: str = "Deadlock",
    streamer_hero: str | None = None,
    chat_summary: dict[str, Any] | None = None,
) -> str:
    """Build standardized candidate prompt context matching Section 10.2."""
    t_start_str = format_seconds_time(candidate.start)
    t_end_str = format_seconds_time(candidate.end)
    cats_str = ", ".join(candidate.suspected_categories)

    lines: list[str] = [
        f"КАНДИДАТ {candidate.id} | окно {t_start_str} – {t_end_str} | предполагаемые категории: {cats_str}",
        f"ИГРА: {game_name}" + (f" | герой стримера: {streamer_hero}" if streamer_hero else ""),
        "",
        f"ТРАНСКРИПТ (-45 с ... +30 с от пика {format_seconds_time(candidate.peak)}):",
    ]

    # Select segments in window: [peak - 45s, peak + 30s]
    w_start = max(0.0, candidate.peak - 45.0)
    w_end = candidate.peak + 30.0

    window_segs = [s for s in segments if not (s.end < w_start or s.start > w_end)]
    if window_segs:
        min_seg = window_segs[0].id
        max_seg = window_segs[-1].id
        lines.append(
            f"ТРАНСКРИПТ (выбирай start_seg и end_seg строго из диапазона {min_seg}–{max_seg}. "
            f"start_seg ставь на подводку/завязку ДО кульминации, end_seg — на конец реакции):"
        )
        for seg in window_segs:
            t_seg = format_seconds_time(seg.start)
            if getattr(seg, "is_game_voice_line", False) or seg.speaker == "game":
                lines.append(f"[seg {seg.id:04d} | {t_seg}] [game_voice_line, {seg.lang}] (реплика героя: {seg.text})")
            else:
                spk = "СТРИМЕР" if seg.speaker == "streamer" else seg.speaker.upper()
                lines.append(f"[seg {seg.id:04d} | {t_seg}] {spk}: {seg.text}")
    else:
        lines.append("(Речь в окне кандидата не обнаружена, укажи start_seg=0, end_seg=0)")

    lines.append("")
    lines.append("СОБЫТИЯ:")

    # Events in window
    window_events = [e for e in events if not (e.end < candidate.start or e.start > candidate.end)]
    if window_events:
        for evt in window_events:
            evt_t = format_seconds_time(evt.start)
            if evt.type == "multikill":
                label = evt.payload.get("label", "")
                count = evt.payload.get("count", "")
                phrase = evt.payload.get("phrase", "")
                detail = label or (f"{count} убийства" if count else "убийство нескольких врагов")
                phr_str = f" «{phrase}»" if phrase else ""
                lines.append(f"{evt_t} MULTIKILL [{detail}]{phr_str} (подтверждено по killfeed/HUD!)")
            elif evt.type == "clutch":
                phrase = evt.payload.get("phrase", "")
                lines.append(f"{evt_t} CLUTCH / EPIC PLAY «{phrase}» (клатч / затащил раунд)")
            elif evt.type == "lucky_shot":
                phrase = evt.payload.get("phrase", "")
                lines.append(f"{evt_t} LUCKY SHOT «{phrase}» (лаки-выстрел / жесткий шот)")
            elif evt.type == "gameplay_fail":
                phrase = evt.payload.get("phrase", "")
                lines.append(f"{evt_t} GAMEPLAY FAIL «{phrase}» (промах / нелепый провал)")
            elif evt.type == "kill":
                method = evt.payload.get("method", "killfeed")
                lines.append(f"{evt_t} KILL стример → враг (подтверждено визуально по {method})")
            elif evt.type == "laughter":
                dur = evt.payload.get("duration_s", round(evt.end - evt.start, 1))
                lines.append(f"{evt_t} LAUGHTER p={evt.score:.2f} ({dur} с)")
            elif evt.type == "scream":
                lines.append(f"{evt_t} SCREAM p={evt.score:.2f}")
            elif evt.type == "loud_burst":
                lines.append(f"{evt_t} LOUD loud_z max {evt.payload.get('loud_z_max', 3.5)}")
            elif evt.type == "chat_spike":
                lines.append(f"{evt_t} CHAT SPIKE rate_z max {evt.payload.get('chat_rate_z_max', 4.0)}")
            elif evt.type == "hype_phrase":
                lines.append(f"{evt_t} HYPE «{evt.payload.get('phrase', '')}» (score={evt.score:.2f})")
            else:
                lines.append(f"{evt_t} {evt.type.upper()} (score={evt.score:.2f})")
    else:
        lines.append("(Специальных дискретных событий в окне нет)")

    # Chat summary
    lines.append("")
    feats = candidate.features_summary
    chat_z = feats.get("chat_rate_z", 0.0)
    chat_l = feats.get("chat_laugh", 0.0)
    if chat_z > 1.0 or chat_l > 0.1:
        lines.append(f"ЧАТ (сдвиг задержки учтён): rate_z max {chat_z:.1f}; доля смеха: {chat_l * 100:.0f}%")
    else:
        lines.append("ЧАТ: спокойный или отсутствует")

    # Features summary
    lines.append("")
    has_sem = any(e.type == "semantic_flag" for e in window_events)
    sem_str = "есть" if has_sem else "нет"
    lines.append(f"ПРИЗНАКИ: S(t) peak {candidate.signal_score:.2f} | semantic_flag: {sem_str}")

    return "\n".join(lines)
