"""Unit tests for S09 Semantic Scan stage."""

from __future__ import annotations

import json
from unittest.mock import patch

from narezka.asr.base import Segment
from narezka.llm.schemas import SemanticItem, SemanticScanResult
from narezka.pipeline.stages.s09_semantic_scan import S09SemanticScanStage
from narezka.project import Project


def test_semantic_scan_disabled(test_project: Project):
    test_project.config.semantic_scan.enabled = False

    stage = S09SemanticScanStage()
    out_dir = test_project.stages_dir / stage.name
    out_dir.mkdir(parents=True, exist_ok=True)

    res = stage.run(test_project, out_dir)
    assert res["enabled"] is False
    assert res["num_events"] == 0

    events_file = out_dir / "semantic_events.json"
    assert events_file.is_file()
    data = json.loads(events_file.read_text(encoding="utf-8"))
    assert data["events"] == []


def test_semantic_scan_with_mocked_llm(test_project: Project):
    test_project.config.semantic_scan.enabled = True
    test_project.config.semantic_scan.window_min = 2  # 2 min windows for fast test
    test_project.config.semantic_scan.overlap_min = 1

    # Fake S04 ASR transcript
    s04_dir = test_project.stages_dir / "s04_asr"
    s04_dir.mkdir(parents=True, exist_ok=True)
    segments = [
        Segment(id=1, start=10.0, end=15.0, text="Опять эти крипы не туда пошли", speaker="streamer"),
        Segment(id=2, start=16.0, end=20.0, text="Ну да, ведь крипы сами виноваты, ага", speaker="streamer"),
        Segment(id=3, start=21.0, end=25.0, text="Потрясающая тактика команды", speaker="streamer"),
        Segment(id=4, start=70.0, end=75.0, text="Обычный разговор дальше", speaker="streamer"),
    ]
    with open(s04_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump({"segments": [s.model_dump() for s in segments]}, f)

    stage = S09SemanticScanStage()
    out_dir = test_project.stages_dir / stage.name
    out_dir.mkdir(parents=True, exist_ok=True)

    mock_scan_res = SemanticScanResult(
        items=[
            SemanticItem(
                start_seg=2,
                end_seg=3,
                type="irony",
                strength=4,
                why="Сарказм по поводу вины крипов и тактики",
            )
        ]
    )

    with patch("narezka.pipeline.stages.s09_semantic_scan.LLMClient.generate_json", return_value=(mock_scan_res, {"cached": False})):
        res = stage.run(test_project, out_dir)

    assert res["enabled"] is True
    assert res["num_events"] >= 1

    events_file = out_dir / "semantic_events.json"
    assert events_file.is_file()
    data = json.loads(events_file.read_text(encoding="utf-8"))
    events = data["events"]
    assert len(events) >= 1
    assert events[0]["type"] == "semantic_flag"
    assert events[0]["payload"]["subtype"] == "irony"
    assert events[0]["score"] == 0.8  # 4 / 5.0
