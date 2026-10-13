"""Export moments to CSV and YouTube description timestamp formats."""

from __future__ import annotations

import csv
from pathlib import Path

from narezka.clips.ranking import format_time_code
from narezka.models import MomentsDocument


def format_time_code_detailed(seconds: float) -> str:
    """Format seconds into HH:MM:SS.s string."""
    s = int(seconds)
    fraction = int(round((seconds - s) * 10))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{fraction}"


def export_moments_csv(doc: MomentsDocument, output_path: Path) -> Path:
    """Export moments to CSV file with readable timecodes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "start_timecode",
        "end_timecode",
        "start_s",
        "end_s",
        "duration_s",
        "category",
        "title",
        "score",
        "confidence",
        "reason",
        "quotes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in doc.moments:
            quotes_str = " | ".join(m.evidence.get("quotes", []))
            writer.writerow(
                {
                    "id": m.id,
                    "start_timecode": format_time_code_detailed(m.start),
                    "end_timecode": format_time_code_detailed(m.end),
                    "start_s": round(m.start, 2),
                    "end_s": round(m.end, 2),
                    "duration_s": round(m.end - m.start, 2),
                    "category": m.category,
                    "title": m.title,
                    "score": round(m.final_score, 3),
                    "confidence": round(m.confidence, 3),
                    "reason": m.reason,
                    "quotes": quotes_str,
                }
            )

    return output_path


def export_youtube_timestamps(doc: MomentsDocument, output_path: Path) -> Path:
    """Export moments to YouTube chapter timestamp format: HH:MM:SS Title."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    # YouTube chapters require first chapter to be at 00:00:00
    if not any(m.start < 5.0 for m in doc.moments):
        lines.append("00:00:00 Начало стрима")

    for m in doc.moments:
        tc = format_time_code(m.start)
        lines.append(f"{tc} {m.title}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path
