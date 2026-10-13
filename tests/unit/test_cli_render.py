"""Unit tests for narezka CLI render command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from narezka.cli import app
from narezka.models import Moment, MomentsDocument, RunMeta, VodMeta

runner = CliRunner()


def test_cli_render_command(synthetic_video_mp4: Path, tmp_path: Path):
    proj_dir = tmp_path / "test_proj"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Prepare moments.json pointing to synthetic_video_mp4
    doc = MomentsDocument(
        schema_version="1.0",
        vod=VodMeta(
            path=str(synthetic_video_mp4),
            sha1_head="testsha1",
            duration_s=2.0,
        ),
        run=RunMeta(
            config_hash="testhash",
            asr_backend="faster_whisper",
            judge_model="rule_based",
            finished_at="2026-09-28T18:00:00",
        ),
        moments=[
            Moment(
                id="m_0001",
                candidate_id="c_0001",
                category="epic_play",
                title="Тестовый момент для рендера",
                reason="Тест",
                start=0.2,
                end=1.2,
                peak=0.7,
                judge_score=9,
                confidence=0.9,
                final_score=0.88,
            )
        ],
    )
    moments_path = proj_dir / "moments.json"
    moments_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    # Run narezka render with --fast flag
    result = runner.invoke(app, ["render", str(proj_dir), "--fast"])
    assert result.exit_code == 0
    assert "report.html" in result.output

    # Verify clip was created
    clips_dir = proj_dir / "clips"
    assert clips_dir.is_dir()
    clips = list(clips_dir.glob("*.mp4"))
    assert len(clips) == 1

    # Verify moments.json was updated with clip_path
    updated_doc = MomentsDocument.model_validate_json(moments_path.read_text(encoding="utf-8"))
    assert updated_doc.moments[0].clip_path is not None
    assert "clips/" in updated_doc.moments[0].clip_path
