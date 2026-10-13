"""Stage S13: Hardware NVENC video clip rendering and report update."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from narezka.config import AppConfig
from narezka.export.json_export import export_moments_json
from narezka.export.report_html import render_html_report
from narezka.media.render import render_clips_batch
from narezka.models import MomentsDocument
from narezka.pipeline.stage import Stage
from narezka.project import Project


class S13RenderStage(Stage):
    """Render MP4 video clips using hardware NVENC or fast stream copy."""

    name = "s13_render"
    code_version = "1.0.0"
    is_gpu_stage = True

    def compute_input_hash(self, project: Project) -> str:
        s12_meta = project.stages_dir / "s12_clips" / "meta.json"
        s12_hash = ""
        if s12_meta.is_file():
            try:
                with open(s12_meta, encoding="utf-8") as f:
                    s12_hash = json.load(f).get("cache_key", "")
            except Exception:
                pass
        return hashlib.sha256(f"{s12_hash}::{project.metadata.vod.sha1_head}".encode()).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        return hashlib.sha256(
            json.dumps(config.render.model_dump(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        moments_file = project.project_dir / "moments.json"
        if not moments_file.is_file():
            moments_file = project.stages_dir / "s12_clips" / "moments.json"

        if not moments_file.is_file():
            raise FileNotFoundError(f"moments.json not found in {project.project_dir}")

        with open(moments_file, encoding="utf-8") as f:
            doc = MomentsDocument.model_validate_json(f.read())

        clips_dir = project.clips_dir

        # Render clips in parallel using NVENC
        updated_moments = render_clips_batch(
            vod_path=project.vod_path,
            moments=doc.moments,
            clips_dir=clips_dir,
            render_cfg=project.config.render,
            fast=False,
            max_workers=project.config.render.max_parallel,
        )
        doc.moments = updated_moments

        # Save updated moments.json with clip_path
        export_moments_json(doc, project.project_dir / "moments.json")
        export_moments_json(doc, output_dir / "moments.json")

        # Re-render HTML report so video players link to the new clips!
        report_path = render_html_report(project.project_dir)

        rendered_count = sum(1 for m in doc.moments if m.clip_path is not None)

        return {
            "num_clips_rendered": rendered_count,
            "clips_dir": str(clips_dir),
            "report_path": str(report_path),
        }
