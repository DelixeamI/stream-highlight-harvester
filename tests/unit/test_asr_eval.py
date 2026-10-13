"""Unit tests for ASR WER and CER evaluation."""

from __future__ import annotations

from pathlib import Path

from narezka.eval.asr_eval import compute_metrics, evaluate_gold_directory


def test_compute_metrics_identical() -> None:
    ref = "Привет мир, это тест!"
    hyp = "привет мир это тест"
    metrics = compute_metrics(ref, hyp)
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0


def test_compute_metrics_mismatch() -> None:
    ref = "Хейз забирает троих"
    hyp = "Хейз забирает двоих"
    metrics = compute_metrics(ref, hyp)
    assert metrics["wer"] > 0.0
    assert metrics["cer"] > 0.0


def test_evaluate_gold_directory(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()

    file1 = gold_dir / "sample1.txt"
    file1.write_text("я иду на мид", encoding="utf-8")

    file2 = gold_dir / "sample2.txt"
    file2.write_text("минус три у урны", encoding="utf-8")

    predictions = {
        "sample1": "я иду на мид",
        "sample2": "минус три у урны",
    }

    results = evaluate_gold_directory(gold_dir, predictions)
    assert results["overall_wer"] == 0.0
    assert results["overall_cer"] == 0.0
    assert len(results["files"]) == 2
