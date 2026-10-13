"""Unit tests for exports: moments.json, moments.csv, youtube_timestamps.txt and HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from narezka.export.csv_export import export_moments_csv, export_youtube_timestamps
from narezka.export.json_export import export_moments_json
from narezka.export.report_html import generate_svg_timeline_chart, render_html_report
from narezka.models import Moment, MomentsDocument, RunMeta, VodMeta


@pytest.fixture
def sample_moments_doc() -> MomentsDocument:
    vod = VodMeta(path="D:/vods/test_stream.mp4", duration_s=3600.0, sha1_head="abc123sha1")
    run = RunMeta(
        app_version="0.1.0",
        config_hash="cfg_hash_123",
        asr_backend="faster_whisper",
        judge_model="rule_based_draft",
        cost_usd=0.0,
        finished_at="2026-09-28T18:00:00Z",
    )
    moments = [
        Moment(
            id="m_0001",
            candidate_id="c_0001",
            category="multikill",
            title="Мультикилл стримера (00:15:30)",
            reason="Тройное убийство за 6 секунд.",
            start=920.0,
            end=950.0,
            peak=930.0,
            judge_score=9,
            confidence=0.9,
            final_score=0.88,
            evidence={"quotes": ["захожу сзади"], "features_summary": {"loud_z": 4.2}},
        ),
        Moment(
            id="m_0002",
            candidate_id="c_0002",
            category="funny_reaction",
            title="«Ты это видел?!» (00:25:10)",
            reason="Громкий смех и реакция чата.",
            start=1500.0,
            end=1525.0,
            peak=1510.0,
            judge_score=8,
            confidence=0.85,
            final_score=0.82,
            evidence={"quotes": ["ты это видел"], "features_summary": {"laugh_p": 0.95}},
        ),
    ]
    return MomentsDocument(schema_version="1.0", vod=vod, run=run, moments=moments)


def test_export_moments_json(tmp_path: Path, sample_moments_doc: MomentsDocument):
    out_file = tmp_path / "moments.json"
    export_moments_json(sample_moments_doc, out_file)
    assert out_file.is_file()

    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert data["vod"]["duration_s"] == 3600.0
    assert len(data["moments"]) == 2
    assert data["moments"][0]["category"] == "multikill"


def test_export_moments_csv(tmp_path: Path, sample_moments_doc: MomentsDocument):
    csv_file = tmp_path / "moments.csv"
    export_moments_csv(sample_moments_doc, csv_file)
    assert csv_file.is_file()

    lines = csv_file.read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 3  # Header + 2 moments
    assert "start_timecode" in lines[0]
    assert "multikill" in lines[1]


def test_export_youtube_timestamps(tmp_path: Path, sample_moments_doc: MomentsDocument):
    yt_file = tmp_path / "youtube_timestamps.txt"
    export_youtube_timestamps(sample_moments_doc, yt_file)
    assert yt_file.is_file()

    lines = yt_file.read_text(encoding="utf-8").splitlines()
    # First line should be 00:00:00
    assert lines[0].startswith("00:00:00")
    # Contains moment titles
    assert any("Мультикилл" in line for line in lines)


def test_generate_svg_timeline_chart(sample_moments_doc: MomentsDocument):
    t = np.linspace(0, 3600, 7200, dtype=np.float32)
    s = np.zeros_like(t)
    s[1800:1900] = 1.2
    s[3000:3050] = 0.9

    svg = generate_svg_timeline_chart(
        timestamps=t,
        s_smooth=s,
        moments=sample_moments_doc.moments,
        duration_s=3600.0,
    )
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "multikill" in svg
    assert "#ef4444" in svg  # Red color for multikill marker


def test_render_html_report(tmp_path: Path, sample_moments_doc: MomentsDocument):
    project_dir = tmp_path / "test_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save moments.json
    export_moments_json(sample_moments_doc, project_dir / "moments.json")

    # Save timeline.npz
    t = np.linspace(0, 3600, 7200, dtype=np.float32)
    s = np.zeros_like(t)
    np.savez_compressed(str(project_dir / "s10_candidates" / "timeline.npz") if False else str(project_dir / "timeline.npz"), timestamps=t, s_smooth=s)

    report_html = render_html_report(project_dir)
    assert report_html.is_file()

    content = report_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "test_stream.mp4" in content
    assert "Мультикилл стримера" in content
    assert "video controls" in content
    assert "copyTimecode" in content
