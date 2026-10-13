"""Evaluation module for calculating WER and CER of ASR backends on Russian speech."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jiwer
from rich.console import Console
from rich.table import Table

from narezka.asr.postprocess import normalize_russian_text

logger = logging.getLogger(__name__)
console = Console()


def compute_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    """Calculate WER and CER between reference and hypothesis after Russian text normalization."""
    norm_ref = normalize_russian_text(reference)
    norm_hyp = normalize_russian_text(hypothesis)

    if not norm_ref:
        return {"wer": 0.0 if not norm_hyp else 1.0, "cer": 0.0 if not norm_hyp else 1.0}

    wer = float(jiwer.wer(norm_ref, norm_hyp))
    cer = float(jiwer.cer(norm_ref, norm_hyp))

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "wer_percent": round(wer * 100, 2),
        "cer_percent": round(cer * 100, 2),
    }


def evaluate_gold_directory(
    gold_dir: Path,
    predictions: dict[str, str],
) -> dict[str, Any]:
    """Compare predictions against gold reference .txt files in gold_dir."""
    gold_path = Path(gold_dir)
    if not gold_path.is_dir():
        raise FileNotFoundError(f"Папка с эталонными транскриптами не найдена: {gold_path}")

    gold_files = list(gold_path.glob("*.txt"))
    if not gold_files:
        raise ValueError(f"В папке {gold_path} не найдено .txt файлов для оценки.")

    file_results: list[dict[str, Any]] = []
    all_refs: list[str] = []
    all_hyps: list[str] = []

    for g_file in gold_files:
        ref_text = g_file.read_text(encoding="utf-8").strip()
        stem = g_file.stem
        hyp_text = predictions.get(stem, "").strip()

        metrics = compute_metrics(ref_text, hyp_text)
        file_results.append({
            "file": g_file.name,
            "metrics": metrics,
            "ref_words": len(normalize_russian_text(ref_text).split()),
            "hyp_words": len(normalize_russian_text(hyp_text).split()),
        })

        all_refs.append(normalize_russian_text(ref_text))
        all_hyps.append(normalize_russian_text(hyp_text))

    overall_wer = float(jiwer.wer(all_refs, all_hyps)) if all_refs else 0.0
    overall_cer = float(jiwer.cer(all_refs, all_hyps)) if all_refs else 0.0

    return {
        "overall_wer": round(overall_wer, 4),
        "overall_cer": round(overall_cer, 4),
        "overall_wer_percent": round(overall_wer * 100, 2),
        "overall_cer_percent": round(overall_cer * 100, 2),
        "files": file_results,
    }


def print_evaluation_report(results: dict[str, Any], backend_name: str = "ASR") -> None:
    """Print Rich table summarizing WER and CER results."""
    table = Table(title=f"Оценка качества распознавания ({backend_name})", show_lines=True)
    table.add_column("Файл", style="cyan")
    table.add_column("Слов (эталон)", justify="right")
    table.add_column("Слов (гипотеза)", justify="right")
    table.add_column("WER (%)", justify="center", style="bold")
    table.add_column("CER (%)", justify="center")

    for f in results.get("files", []):
        m = f["metrics"]
        wer_color = "green" if m["wer_percent"] <= 20.0 else "yellow"
        table.add_row(
            f["file"],
            str(f["ref_words"]),
            str(f["hyp_words"]),
            f"[{wer_color}]{m['wer_percent']:.1f}%[/{wer_color}]",
            f"{m['cer_percent']:.1f}%",
        )

    overall_wer = results["overall_wer_percent"]
    overall_cer = results["overall_cer_percent"]
    ow_color = "bold green" if overall_wer <= 20.0 else "bold red"

    table.add_row(
        "[bold]ИТОГО[/bold]",
        "-",
        "-",
        f"[{ow_color}]{overall_wer:.1f}%[/{ow_color}]",
        f"[bold]{overall_cer:.1f}%[/bold]",
    )

    console.print(table)
