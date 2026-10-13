"""Unit tests for Stage disk caching, invalidation, and DAG execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from narezka.config import AppConfig
from narezka.pipeline.dag import PipelineDAG
from narezka.pipeline.stage import Stage
from narezka.project import Project


class DummyStage(Stage):
    """Test stage implementation that tracks execution counts."""

    name = "dummy_stage"
    code_version = "1.0.0"

    def __init__(self, stage_name: str = "dummy_stage", code_ver: str = "1.0.0") -> None:
        self.name = stage_name
        self.code_version = code_ver
        self.run_count = 0
        self.custom_input = "initial_input_value"

    def compute_input_hash(self, project: Project) -> str:
        return hashlib.sha256(self.custom_input.encode("utf-8")).hexdigest()

    def compute_params_hash(self, config: AppConfig) -> str:
        # Depends only on vad threshold for testing
        val = str(config.audio.vad.threshold)
        return hashlib.sha256(val.encode("utf-8")).hexdigest()

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        self.run_count += 1
        output_file = output_dir / "output.txt"
        output_file.write_text(f"run_count={self.run_count}", encoding="utf-8")
        return {"count": self.run_count, "file": str(output_file)}


def test_stage_cache_skips_on_second_run(test_project: Project) -> None:
    stage = DummyStage()

    # First run: should execute
    res1 = stage.execute(test_project)
    assert not res1.is_cached
    assert stage.run_count == 1
    assert stage.get_meta_file(test_project).exists()

    # Second run: should be cached and skip execution
    res2 = stage.execute(test_project)
    assert res2.is_cached
    assert stage.run_count == 1  # Not executed again
    assert res2.metadata.cache_key == res1.metadata.cache_key


def test_stage_cache_invalidates_on_param_change(test_project: Project) -> None:
    stage = DummyStage()

    # Initial run
    res1 = stage.execute(test_project)
    assert not res1.is_cached
    assert stage.run_count == 1

    # Modify config parameter that this stage depends on
    test_project.config.audio.vad.threshold = 0.85

    # Run again: should detect param change and re-execute
    res2 = stage.execute(test_project)
    assert not res2.is_cached
    assert stage.run_count == 2
    assert res2.metadata.cache_key != res1.metadata.cache_key


def test_stage_cache_invalidates_on_input_change(test_project: Project) -> None:
    stage = DummyStage()

    # Initial run
    stage.execute(test_project)
    assert stage.run_count == 1

    # Modify input
    stage.custom_input = "modified_input_value"

    # Run again: should detect input change and re-execute
    res = stage.execute(test_project)
    assert not res.is_cached
    assert stage.run_count == 2


def test_stage_cache_invalidates_on_code_version_change(test_project: Project) -> None:
    stage_v1 = DummyStage(code_ver="1.0.0")
    stage_v1.execute(test_project)
    assert stage_v1.run_count == 1

    # Create new instance representing updated stage code
    stage_v2 = DummyStage(code_ver="2.0.0")
    res = stage_v2.execute(test_project)
    assert not res.is_cached
    assert stage_v2.run_count == 1


def test_stage_force_flag_bypasses_cache(test_project: Project) -> None:
    stage = DummyStage()
    stage.execute(test_project)
    assert stage.run_count == 1

    # Execute with force=True
    res = stage.execute(test_project, force=True)
    assert not res.is_cached
    assert stage.run_count == 2


def test_pipeline_dag_execution(test_project: Project) -> None:
    dag = PipelineDAG()
    s1 = DummyStage("stage_one")
    s2 = DummyStage("stage_two")
    dag.register(s1)
    dag.register(s2)

    results = dag.run(test_project)
    assert len(results) == 2
    assert results[0].stage_name == "stage_one"
    assert results[1].stage_name == "stage_two"
    assert s1.run_count == 1
    assert s2.run_count == 1

    # Run with until_stage
    s3 = DummyStage("stage_three")
    dag.register(s3)
    results_until = dag.run(test_project, until_stage="stage_two")
    assert len(results_until) == 2
    assert s3.run_count == 0
