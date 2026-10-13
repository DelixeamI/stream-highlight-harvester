"""Evaluation of highlight candidate generation and moment selection against ground truth.

Implements metrics from TZ Section 18.2:
- Candidate Recall (target >= 85%)
- Moment Recall@N (target >= 70%)
- Boundary accuracy (+/-3s)
- Miss analysis: "не было кандидата" vs "отклонён судьёй / правилами"
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from narezka.clips.ranking import format_time_code
from narezka.models import Candidate, Moment

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthItem:
    """Ground truth highlight annotation."""

    start: float
    end: float
    category: str
    note: str

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


def parse_time_str(val: str) -> float:
    """Parse HH:MM:SS or float seconds string."""
    val = val.strip()
    if ":" in val:
        parts = val.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    return float(val)


def load_ground_truth(gt_path: Path) -> list[GroundTruthItem]:
    """Load ground truth highlights from CSV file."""
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_path}")

    items: list[GroundTruthItem] = []
    with open(gt_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_str = row.get("start") or row.get("start_timecode") or "0"
            e_str = row.get("end") or row.get("end_timecode") or "0"
            cat = row.get("category") or "other"
            note = row.get("note") or row.get("title") or ""

            s = parse_time_str(s_str)
            e = parse_time_str(e_str)
            items.append(GroundTruthItem(start=s, end=e, category=cat, note=note))

    return items


def is_match(gt_start: float, gt_end: float, pred_start: float, pred_end: float) -> bool:
    """Check if prediction matches GT: overlap >= 1.0s or center distance <= 10.0s."""
    overlap = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    if overlap >= 1.0:
        return True

    gt_center = (gt_start + gt_end) / 2.0
    pred_center = (pred_start + pred_end) / 2.0
    return abs(gt_center - pred_center) <= 10.0


@dataclass
class EvalReport:
    """Evaluation summary metrics and missed highlights report."""

    total_gt: int
    num_candidates: int
    num_moments: int
    candidate_recall: float
    moment_recall: float
    boundary_accuracy_3s: float
    missed_items: list[dict[str, Any]]


def evaluate_highlights(
    gt_items: list[GroundTruthItem],
    candidates: list[Candidate],
    moments: list[Moment],
) -> EvalReport:
    """Evaluate candidate generation and moment selection against ground truth."""
    if not gt_items:
        return EvalReport(0, len(candidates), len(moments), 0.0, 0.0, 0.0, [])

    candidate_hits = 0
    moment_hits = 0
    boundary_hits = 0
    missed: list[dict[str, Any]] = []

    for gt in gt_items:
        # 1. Check candidate match
        matching_cand = [c for c in candidates if is_match(gt.start, gt.end, c.start, c.end)]
        has_cand = len(matching_cand) > 0
        if has_cand:
            candidate_hits += 1

        # 2. Check moment match
        matching_moments = [m for m in moments if is_match(gt.start, gt.end, m.start, m.end)]
        has_moment = len(matching_moments) > 0
        if has_moment:
            moment_hits += 1
            best_m = matching_moments[0]
            # Check boundary accuracy (+/-3s)
            if abs(best_m.start - gt.start) <= 3.0 and abs(best_m.end - gt.end) <= 3.0:
                boundary_hits += 1
        else:
            # Analyze miss reason
            reason = "не было кандидата" if not has_cand else "отклонён судьёй / отсеян правилами"
            missed.append(
                {
                    "gt": gt,
                    "reason": reason,
                    "candidate": matching_cand[0] if has_cand else None,
                }
            )

    cand_recall = candidate_hits / len(gt_items)
    mom_recall = moment_hits / len(gt_items)
    bound_acc = (boundary_hits / moment_hits) if moment_hits > 0 else 0.0

    return EvalReport(
        total_gt=len(gt_items),
        num_candidates=len(candidates),
        num_moments=len(moments),
        candidate_recall=cand_recall,
        moment_recall=mom_recall,
        boundary_accuracy_3s=bound_acc,
        missed_items=missed,
    )


def print_highlights_report(report: EvalReport, console: Console) -> None:
    """Display evaluation metrics table and missed highlight explanations."""
    console.print("\n[bold cyan]═══ Результаты оценки качества поиска моментов ═══[/bold cyan]")

    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Метрика", style="dim")
    t.add_column("Значение", justify="right")
    t.add_column("Цель ТЗ", justify="right")
    t.add_column("Статус", justify="center")

    cand_status = "[green][OK][/green]" if report.candidate_recall >= 0.85 else "[red][FAIL][/red]"
    mom_status = "[green][OK][/green]" if report.moment_recall >= 0.70 else "[yellow][WARN][/yellow]"
    bound_status = "[green][OK][/green]" if report.boundary_accuracy_3s >= 0.70 else "[yellow][WARN][/yellow]"

    t.add_row("Эталонных моментов (GT)", str(report.total_gt), "-", "-")
    t.add_row("Кандидатов (S10)", str(report.num_candidates), "-", "-")
    t.add_row("Итоговых моментов (S12)", str(report.num_moments), "-", "-")
    t.add_row("Recall кандидатов", f"{report.candidate_recall * 100:.1f}%", ">= 85%", cand_status)
    t.add_row("Recall@N моментов", f"{report.moment_recall * 100:.1f}%", ">= 70%", mom_status)
    t.add_row("Точность границ (+/-3с)", f"{report.boundary_accuracy_3s * 100:.1f}%", ">= 70%", bound_status)

    console.print(t)

    if report.missed_items:
        console.print("\n[bold yellow]Пропущенные эталонные моменты:[/bold yellow]")
        miss_table = Table(show_header=True, header_style="bold red")
        miss_table.add_column("Таймкод GT")
        miss_table.add_column("Категория")
        miss_table.add_column("Заметка")
        miss_table.add_column("Причина пропуска")

        for m in report.missed_items:
            gt: GroundTruthItem = m["gt"]
            tc = f"{format_time_code(gt.start)} – {format_time_code(gt.end)}"
            miss_table.add_row(tc, gt.category, gt.note, m["reason"])

        console.print(miss_table)
    else:
        console.print("[bold green]Все эталонные моменты успешно найдены![/bold green]")
