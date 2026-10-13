"""Unit tests for media rendering (NVENC/FFmpeg and stream copy)."""

from __future__ import annotations

from pathlib import Path

from narezka.config import RenderConfig
from narezka.media.render import (
    format_clip_filename,
    render_clips_batch,
    render_single_clip,
    slugify_title,
)
from narezka.models import Moment


def test_slugify_title():
    assert slugify_title("Привет Мир!") == "privet-mir"
    assert slugify_title("«Хорош, минус один!»") == "horosh-minus-odin"
    assert slugify_title("Эпический момент Deadlock 2026") == "epicheskiy-moment-deadlock-2026"
    assert slugify_title("") == "clip"
    long_title = "Очень длинное название клипа которое точно превышает сорок символов"
    assert len(slugify_title(long_title)) <= 40


def test_format_clip_filename():
    m = Moment(
        id="m_0001",
        candidate_id="c_0001",
        category="funny_reaction",
        title="Опять крипы",
        reason="Смешной комментарий",
        start=3665.0,  # 01h01m05s
        end=3680.0,
        peak=3670.0,
        judge_score=8,
        confidence=0.9,
        final_score=0.85,
    )
    fname = format_clip_filename(1, m)
    assert fname == "001_01h01m05s_funny_reaction_opyat-kripy.mp4"


def test_render_single_clip_copy_mode(synthetic_video_mp4: Path, tmp_path: Path):
    render_cfg = RenderConfig(codec="libx264", max_parallel=2)
    m = Moment(
        id="m_0001",
        candidate_id="c_0001",
        category="epic_play",
        title="Тестовый клип",
        reason="Тест",
        start=0.2,
        end=1.2,
        peak=0.5,
        judge_score=7,
        confidence=0.8,
        final_score=0.75,
    )
    out_clip = tmp_path / "clip_copy.mp4"
    rendered = render_single_clip(
        vod_path=synthetic_video_mp4,
        moment=m,
        output_path=out_clip,
        render_cfg=render_cfg,
        fast=True,
    )
    assert rendered.is_file()
    assert rendered.stat().st_size > 0


def test_render_single_clip_reencode_mode(synthetic_video_mp4: Path, tmp_path: Path):
    render_cfg = RenderConfig(codec="libx264", max_parallel=2)
    m = Moment(
        id="m_0001",
        candidate_id="c_0001",
        category="epic_play",
        title="Тестовый клип 2",
        reason="Тест",
        start=0.5,
        end=1.5,
        peak=1.0,
        judge_score=9,
        confidence=0.95,
        final_score=0.9,
    )
    out_clip = tmp_path / "clip_reencode.mp4"
    rendered = render_single_clip(
        vod_path=synthetic_video_mp4,
        moment=m,
        output_path=out_clip,
        render_cfg=render_cfg,
        fast=False,
    )
    assert rendered.is_file()
    assert rendered.stat().st_size > 0


def test_render_clips_batch(synthetic_video_mp4: Path, tmp_path: Path):
    render_cfg = RenderConfig(codec="libx264", max_parallel=2)
    moments = [
        Moment(
            id="m_0001",
            candidate_id="c_0001",
            category="epic_play",
            title="Клип Один",
            reason="Тест",
            start=0.1,
            end=0.8,
            peak=0.4,
            judge_score=8,
            confidence=0.9,
            final_score=0.85,
        ),
        Moment(
            id="m_0002",
            candidate_id="c_0002",
            category="funny_joke",
            title="Клип Два",
            reason="Тест",
            start=0.9,
            end=1.8,
            peak=1.3,
            judge_score=7,
            confidence=0.8,
            final_score=0.75,
        ),
    ]

    clips_dir = tmp_path / "clips"
    updated = render_clips_batch(
        vod_path=synthetic_video_mp4,
        moments=moments,
        clips_dir=clips_dir,
        render_cfg=render_cfg,
        fast=False,
    )

    assert len(updated) == 2
    for m in updated:
        assert m.clip_path is not None
        actual_file = tmp_path / m.clip_path
        assert actual_file.is_file()
        assert actual_file.stat().st_size > 0
