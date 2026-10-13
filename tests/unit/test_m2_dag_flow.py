"""Integration test for full M2 DAG pipeline execution."""

from __future__ import annotations

import json

from narezka.asr.base import Segment, Word
from narezka.pipeline.dag import PipelineDAG
from narezka.pipeline.stages.s00_probe import S00ProbeStage
from narezka.pipeline.stages.s01_audio_extract import S01AudioExtractStage
from narezka.pipeline.stages.s06_audio_features import S06AudioFeaturesStage
from narezka.pipeline.stages.s08_chat_features import S08ChatFeaturesStage
from narezka.pipeline.stages.s10_candidates import S10CandidatesStage
from narezka.pipeline.stages.s12_clips import S12ClipsStage
from narezka.project import Project


def test_m2_dag_full_execution(test_project: Project):
    dag = PipelineDAG()
    dag.register(S00ProbeStage())
    dag.register(S01AudioExtractStage())

    # Pre-seed a dummy S04 transcript to simulate completed speech recognition
    s04_dir = test_project.stages_dir / "s04_asr"
    s04_dir.mkdir(parents=True, exist_ok=True)
    seg = Segment(
        id=1,
        start=0.2,
        end=1.8,
        text="Это проверка стрима, вот это да!",
        words=[Word(text="Это", start=0.2, end=0.5), Word(text="проверка", start=0.6, end=1.0)],
    )
    (s04_dir / "transcript.json").write_text(
        json.dumps({"segments": [seg.model_dump()]}), encoding="utf-8"
    )
    (s04_dir / "meta.json").write_text(
        json.dumps({"cache_key": "s04_test_key", "status": "completed", "duration_s": 1.0}),
        encoding="utf-8",
    )

    # Register M2 stages
    dag.register(S06AudioFeaturesStage())
    dag.register(S08ChatFeaturesStage())
    dag.register(S10CandidatesStage())
    dag.register(S12ClipsStage())

    # Execute all stages through S12
    results = dag.run(test_project, until_stage="s12_clips")
    assert len(results) == 6

    # Verify stage artifacts exist
    s06_dir = test_project.stages_dir / "s06_audio_features"
    assert (s06_dir / "audio_features.npz").is_file()
    assert (s06_dir / "audio_events.json").is_file()

    s08_dir = test_project.stages_dir / "s08_chat_features"
    assert (s08_dir / "chat_features.npz").is_file()
    assert (s08_dir / "chat_events.json").is_file()

    s10_dir = test_project.stages_dir / "s10_candidates"
    assert (s10_dir / "timeline.npz").is_file()
    assert (s10_dir / "candidates.jsonl").is_file()

    # Verify project-level exports
    assert (test_project.project_dir / "moments.json").is_file()
    assert (test_project.project_dir / "moments.csv").is_file()
    assert (test_project.project_dir / "youtube_timestamps.txt").is_file()
    assert (test_project.project_dir / "report.html").is_file()

    # Verify HTML report contains required elements
    html_content = (test_project.project_dir / "report.html").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "Интегральный интерес S(t)" in html_content
    assert "<svg" in html_content
