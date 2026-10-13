"""End-to-end integration test for M3 DAG flow (S00 -> S06/08 -> S09 -> S10 -> S11 -> S12 -> S13)."""

from __future__ import annotations

import json
from unittest.mock import patch

from narezka.asr.base import Segment, Word
from narezka.llm.schemas import JudgeVerdict, SemanticItem, SemanticScanResult
from narezka.pipeline.dag import PipelineDAG
from narezka.pipeline.stages.s00_probe import S00ProbeStage
from narezka.pipeline.stages.s01_audio_extract import S01AudioExtractStage
from narezka.pipeline.stages.s06_audio_features import S06AudioFeaturesStage
from narezka.pipeline.stages.s08_chat_features import S08ChatFeaturesStage
from narezka.pipeline.stages.s09_semantic_scan import S09SemanticScanStage
from narezka.pipeline.stages.s10_candidates import S10CandidatesStage
from narezka.pipeline.stages.s11_judge import S11JudgeStage
from narezka.pipeline.stages.s12_clips import S12ClipsStage
from narezka.pipeline.stages.s13_render import S13RenderStage
from narezka.project import Project


def test_m3_full_flow(test_project: Project):
    # Configure CPU / fast test settings
    test_project.config.device = "cpu"
    test_project.config.render.codec = "libx264"
    test_project.config.render.max_parallel = 2
    test_project.config.semantic_scan.enabled = True
    test_project.config.semantic_scan.window_min = 1
    test_project.config.semantic_scan.overlap_min = 0

    dag = PipelineDAG()
    dag.register(S00ProbeStage())
    dag.register(S01AudioExtractStage())

    # Run S00 and S01
    dag.run(test_project, until_stage="s01_audio")

    # Mock S04 ASR transcript
    s04_dir = test_project.stages_dir / "s04_asr"
    s04_dir.mkdir(parents=True, exist_ok=True)
    seg1 = Segment(
        id=1,
        start=0.2,
        end=0.9,
        text="Вот это развалил их красиво!",
        words=[Word(text="Вот", start=0.2, end=0.4), Word(text="красиво", start=0.5, end=0.9)],
    )
    seg2 = Segment(
        id=2,
        start=1.0,
        end=1.8,
        text="Невероятная победа на миду",
        words=[Word(text="Невероятная", start=1.0, end=1.4), Word(text="победа", start=1.5, end=1.8)],
    )
    (s04_dir / "transcript.json").write_text(
        json.dumps({"segments": [seg1.model_dump(), seg2.model_dump()]}), encoding="utf-8"
    )
    (s04_dir / "meta.json").write_text(
        json.dumps({"cache_key": "s04_test_key", "status": "completed", "duration_s": 2.0}),
        encoding="utf-8",
    )

    # Register M2 and M3 stages
    dag.register(S06AudioFeaturesStage())
    dag.register(S08ChatFeaturesStage())
    dag.register(S09SemanticScanStage())
    dag.register(S10CandidatesStage())
    dag.register(S11JudgeStage())
    dag.register(S12ClipsStage())
    dag.register(S13RenderStage())

    # Mock responses for Semantic Scan and LLM Judge
    mock_scan_res = SemanticScanResult(
        items=[
            SemanticItem(
                start_seg=1,
                end_seg=2,
                type="irony",
                strength=5,
                why="Эмоциональное торжество стримера",
            )
        ]
    )

    def mock_judge_evaluate(*args, **kwargs):
        cand = kwargs.get("candidate") or (args[0] if args else None)
        cand_id = cand.id if cand else "c_0001"
        verdict = JudgeVerdict(
            candidate_id=cand_id,
            verdict="keep",
            category="epic_play",
            title_ru="Развалил на миду",
            reason_ru="Яркий клатч стримера с чистой реакцией",
            start_seg=1,
            end_seg=2,
            score=9,
            confidence=0.95,
            evidence_quotes=["Вот это развалил их красиво"],
        )
        return verdict, []

    with patch("narezka.pipeline.stages.s09_semantic_scan.LLMClient.generate_json", return_value=(mock_scan_res, {"cached": False})), \
         patch("narezka.judge.judge.LLMJudge.evaluate_candidate", side_effect=mock_judge_evaluate):
        results = dag.run(test_project, until_stage="s13_render")

    stage_names = [r.stage_name for r in results]
    assert "s09_semantic_scan" in stage_names
    assert "s10_candidates" in stage_names
    assert "s11_judge" in stage_names
    assert "s12_clips" in stage_names
    assert "s13_render" in stage_names

    # Verify stage artifacts exist
    s09_dir = test_project.stages_dir / "s09_semantic_scan"
    assert (s09_dir / "semantic_events.json").is_file()

    s10_dir = test_project.stages_dir / "s10_candidates"
    assert (s10_dir / "candidates.jsonl").is_file()

    s11_dir = test_project.stages_dir / "s11_judge"
    assert (s11_dir / "judge_verdicts.jsonl").is_file()
    assert (s11_dir / "judge_summary.json").is_file()

    # Verify moments.json contains moments with Judge verdicts
    moments_file = test_project.project_dir / "moments.json"
    assert moments_file.is_file()
    doc = json.loads(moments_file.read_text(encoding="utf-8"))
    assert len(doc["moments"]) >= 1

    first_moment = doc["moments"][0]
    assert first_moment["title"] == "Развалил на миду"
    assert first_moment["judge_score"] == 9
    assert first_moment["clip_path"] is not None

    # Verify rendered video clip file physically exists!
    clip_file = test_project.project_dir / first_moment["clip_path"]
    assert clip_file.is_file()
    assert clip_file.stat().st_size > 0

    # Verify HTML report was updated and exists
    report_file = test_project.project_dir / "report.html"
    assert report_file.is_file()
    report_html = report_file.read_text(encoding="utf-8")
    assert "Развалил на миду" in report_html
    assert "clips/" in report_html
