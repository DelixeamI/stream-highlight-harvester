"""Unit tests for highlights evaluation against ground truth."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from narezka.eval.highlights_eval import (
    evaluate_highlights,
    load_ground_truth,
    parse_time_str,
    print_highlights_report,
)
from narezka.models import Candidate, Moment


def test_parse_time_str():
    assert parse_time_str("01:23:45") == 3600 + 23 * 60 + 45
    assert parse_time_str("10:30") == 630.0
    assert parse_time_str("125.5") == 125.5


def test_load_ground_truth(tmp_path: Path):
    gt_file = tmp_path / "gt.csv"
    gt_file.write_text(
        """start,end,category,note
01:23:10,01:23:55,multikill,тройное у урны
02:05:01,02:05:20,funny_joke,шутка про крипов
""",
        encoding="utf-8",
    )

    items = load_ground_truth(gt_file)
    assert len(items) == 2
    assert items[0].start == 3600 + 23 * 60 + 10
    assert items[0].category == "multikill"
    assert items[0].note == "тройное у урны"


def test_evaluate_highlights_recall_and_misses():
    # 2 GT items
    from narezka.eval.highlights_eval import GroundTruthItem

    gt_items = [
        GroundTruthItem(start=100.0, end=130.0, category="multikill", note="triple kill"),
        GroundTruthItem(start=500.0, end=530.0, category="funny_joke", note="good joke"),
        GroundTruthItem(start=900.0, end=920.0, category="fail", note="died to creep"),
    ]

    # Candidates: hits item 1 and item 2, misses item 3
    candidates = [
        Candidate(id="c_1", start=95.0, end=135.0, peak=115.0, signal_score=0.9),
        Candidate(id="c_2", start=490.0, end=540.0, peak=515.0, signal_score=0.8),
    ]

    # Final moments: item 1 is kept, but candidate 2 was filtered out!
    moments = [
        Moment(
            id="m_1",
            candidate_id="c_1",
            category="multikill",
            title="Triple kill",
            reason="",
            start=101.0,  # within 3s of 100.0
            end=129.0,    # within 3s of 130.0
            peak=115.0,
            judge_score=9,
            confidence=0.9,
            final_score=0.88,
        )
    ]

    report = evaluate_highlights(gt_items, candidates, moments)
    assert report.total_gt == 3
    # Candidates hit 2 out of 3 -> recall = 2/3 = 66.7%
    assert abs(report.candidate_recall - (2.0 / 3.0)) < 1e-3
    # Moments hit 1 out of 3 -> recall = 1/3 = 33.3%
    assert abs(report.moment_recall - (1.0 / 3.0)) < 1e-3
    # Boundary accuracy for item 1 is 100% (+/-3s)
    assert report.boundary_accuracy_3s == 1.0

    assert len(report.missed_items) == 2
    # Item 2 had candidate but no moment -> reason is "отклонён судьёй / отсеян правилами"
    assert report.missed_items[0]["reason"] == "отклонён судьёй / отсеян правилами"
    # Item 3 had no candidate -> reason is "не было кандидата"
    assert report.missed_items[1]["reason"] == "не было кандидата"

    # Test report printing doesn't raise exception
    console = Console(record=True)
    print_highlights_report(report, console=console)
    output = console.export_text()
    assert "Recall кандидатов" in output
    assert "не было кандидата" in output
