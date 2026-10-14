"""HTML report generator with inline SVG interest timeline and video media fragment player."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jinja2
import numpy as np

from narezka.clips.ranking import format_time_code
from narezka.export.csv_export import format_time_code_detailed
from narezka.models import MomentsDocument

logger = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "multikill": "#ef4444",
    "epic_play": "#f59e0b",
    "funny_reaction": "#10b981",
    "funny_joke": "#06b6d4",
    "fail": "#f97316",
    "other": "#8b5cf6",
}


def generate_svg_timeline_chart(
    timestamps: np.ndarray,
    s_smooth: np.ndarray,
    moments: list[Any],
    duration_s: float,
    width: int = 1200,
    height: int = 130,
) -> str:
    """Generate lightweight responsive inline SVG chart of S(t) with moment markers."""
    if len(timestamps) == 0 or duration_s <= 0.0:
        return "<svg width='100%' height='60'><text x='50%' y='50%' fill='#94a3b8' text-anchor='middle'>Нет данных графика</text></svg>"

    # Downsample points for fast SVG rendering if long VOD (max 1000 points)
    n_pts = len(timestamps)
    step = max(1, n_pts // 1000)
    t_sampled = timestamps[::step]
    s_sampled = s_smooth[::step]

    max_s = float(np.max(s_smooth)) if len(s_smooth) > 0 else 1.0
    max_s = max(0.5, max_s * 1.1)

    pad_left = 50
    pad_right = 20
    pad_top = 15
    pad_bottom = 25
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Generate path data
    points = []
    for t_val, s_val in zip(t_sampled, s_sampled, strict=False):
        x = pad_left + (t_val / duration_s) * plot_w
        y = pad_top + plot_h - (s_val / max_s) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    path_d = f"M {pad_left},{pad_top + plot_h} L " + " L ".join(points) + f" L {pad_left + plot_w},{pad_top + plot_h} Z"
    line_d = "M " + " L ".join(points)

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">',
        '<defs>',
        '  <linearGradient id="curveGrad" x1="0" y1="0" x2="0" y2="1">',
        '    <stop offset="0%" stop-color="#6366f1" stop-opacity="0.45"/>',
        '    <stop offset="100%" stop-color="#6366f1" stop-opacity="0.0"/>',
        '  </linearGradient>',
        '</defs>',
        # Background grid line
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#111422" rx="4"/>',
    ]

    # Moment highlight bands
    for m in moments:
        m_start = getattr(m, "start", 0.0)
        m_end = getattr(m, "end", 0.0)
        m_cat = getattr(m, "category", "other")
        m_id = getattr(m, "id", "")

        x1 = pad_left + (m_start / duration_s) * plot_w
        x2 = pad_left + (m_end / duration_s) * plot_w
        bw = max(3.0, x2 - x1)
        color = CATEGORY_COLORS.get(m_cat, "#8b5cf6")

        svg_parts.append(
            f'<rect x="{x1:.1f}" y="{pad_top}" width="{bw:.1f}" height="{plot_h}" '
            f'fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="1">'
            f'<title>{m_id}: {m_cat} ({format_time_code(m_start)})</title>'
            f'</rect>'
        )

    # Plot filled polygon and line
    svg_parts.append(f'<path d="{path_d}" fill="url(#curveGrad)"/>')
    svg_parts.append(f'<path d="{line_d}" fill="none" stroke="#818cf8" stroke-width="2"/>')

    # Time ticks
    num_ticks = 6
    for i in range(num_ticks + 1):
        frac = i / num_ticks
        tick_t = frac * duration_s
        x = pad_left + frac * plot_w
        tc_str = format_time_code(tick_t)
        svg_parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top + plot_h}" x2="{x:.1f}" y2="{pad_top + plot_h + 4}" stroke="#475569" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 16}" fill="#64748b" font-size="10" font-family="monospace" text-anchor="middle">{tc_str}</text>'
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def render_html_report(project_dir: Path, output_path: Path | None = None) -> Path:
    """Read project artifacts and generate standalone report.html using Jinja2."""
    moments_json = project_dir / "s12_clips" / "moments.json"
    if not moments_json.is_file():
        moments_json = project_dir / "moments.json"

    if not moments_json.is_file():
        raise FileNotFoundError(f"Cannot generate report: moments.json not found in {project_dir}")

    with open(moments_json, encoding="utf-8") as f:
        doc = MomentsDocument.model_validate_json(f.read())

    # Load timeline data if available
    s10_npz = project_dir / "s10_candidates" / "timeline.npz"
    timestamps = np.array([], dtype=np.float32)
    s_smooth = np.array([], dtype=np.float32)
    if s10_npz.is_file():
        try:
            tdata = np.load(str(s10_npz))
            timestamps = tdata["timestamps"]
            s_smooth = tdata["s_smooth"]
        except Exception as e:
            logger.warning("Could not read timeline.npz for report chart: %s", e)

    # Number of candidates
    num_candidates = 0
    c_file = project_dir / "s10_candidates" / "candidates.jsonl"
    if c_file.is_file():
        with open(c_file, encoding="utf-8") as f:
            num_candidates = sum(1 for line in f if line.strip())

    # Calculate stage elapsed times from meta.json files
    total_elapsed_s = 0.0
    for stage_meta_path in project_dir.glob("*/meta.json"):
        try:
            with open(stage_meta_path, encoding="utf-8") as f:
                sm = json.load(f)
                total_elapsed_s += float(sm.get("elapsed_s", 0.0))
        except Exception:
            pass

    # Relative path from project_dir to VOD
    vod_path = Path(doc.vod.path)
    vod_filename = vod_path.name
    try:
        rel_vod_path = vod_path.relative_to(project_dir)
        vod_url_base = str(rel_vod_path).replace("\\", "/")
    except ValueError:
        # If on different drive or outside, use file URI or fallback to normalized string
        try:
            vod_url_base = vod_path.resolve().as_uri()
        except Exception:
            vod_url_base = str(vod_path).replace("\\", "/")

    # Prepare moments view models
    moments_vm: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}

    for m in doc.moments:
        cat = m.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

        # Video source
        if m.clip_path and (project_dir / m.clip_path).is_file():
            video_src = m.clip_path.replace("\\", "/")
        else:
            video_src = f"{vod_url_base}#t={m.start:.1f},{m.end:.1f}"

        # Evidence pills
        pills: dict[str, Any] = {}
        feats = m.evidence.get("features_summary", {})
        if "loud_z" in feats and feats["loud_z"] > 1.5:
            pills["loud_z"] = f"{feats['loud_z']:.1f}"
        if "laugh_p" in feats and feats["laugh_p"] > 0.3:
            pills["laugh_p"] = f"{feats['laugh_p']:.2f}"
        if "chat_rate_z" in feats and feats["chat_rate_z"] > 1.5:
            pills["chat_rate_z"] = f"{feats['chat_rate_z']:.1f}"
        if "speech_rate" in feats and feats["speech_rate"] > 2.5:
            pills["speech_rate"] = f"{feats['speech_rate']:.1f} w/s"
        if "kill" in feats and feats["kill"] > 0.1:
            pills["kill"] = f"🎯 {feats['kill']:.1f}"

        moments_vm.append(
            {
                "id": m.id,
                "category": m.category,
                "title": m.title,
                "reason": m.reason,
                "start": m.start,
                "end": m.end,
                "start_tc": format_time_code_detailed(m.start),
                "end_tc": format_time_code_detailed(m.end),
                "duration_s": round(m.end - m.start, 1),
                "score_display": round(m.final_score * 10, 1),
                "final_score": m.final_score,
                "quotes": m.evidence.get("quotes", []),
                "evidence_pills": pills,
                "video_src": video_src,
            }
        )

    # Inline SVG Chart
    svg_chart = generate_svg_timeline_chart(
        timestamps=timestamps,
        s_smooth=s_smooth,
        moments=doc.moments,
        duration_s=doc.vod.duration_s,
    )

    # Render Jinja2 template
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("report.html.j2")

    rendered_html = template.render(
        vod_filename=vod_filename,
        duration_formatted=format_time_code(doc.vod.duration_s),
        duration_s=round(doc.vod.duration_s, 1),
        num_candidates=num_candidates,
        run=doc.run,
        total_elapsed_s=round(total_elapsed_s, 1),
        moments=moments_vm,
        category_counts=category_counts,
        svg_chart=svg_chart,
    )

    target_path = output_path or (project_dir / "report.html")
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    logger.info("HTML report generated at %s", target_path)
    return target_path
