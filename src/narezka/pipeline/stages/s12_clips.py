"""Stage S12: Final boundary refinement, moment generation, and NMS deduplication."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from narezka.asr.base import Segment
from narezka.clips.ranking import build_moments_with_judge, build_rule_based_moments
from narezka.config import AppConfig
from narezka.export.csv_export import export_moments_csv, export_youtube_timestamps
from narezka.export.json_export import export_moments_json
from narezka.export.report_html import render_html_report
from narezka.llm.schemas import JudgeVerdict
from narezka.models import Candidate, Event, MomentsDocument, RunMeta, VodMeta
from narezka.pipeline.stage import Stage
from narezka.project import Project


class S12ClipsStage(Stage):
    """Refine boundaries, rank moments, deduplicate with NMS, and generate exports & HTML report."""

    name = "s12_clips"
    code_version = "1.1.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        s10_meta = project.stages_dir / "s10_candidates" / "meta.json"
        s04_meta = project.stages_dir / "s04_asr" / "meta.json"
        s11_meta = project.stages_dir / "s11_judge" / "meta.json"
        h10 = ""
        h4 = ""
        h11 = ""
        if s10_meta.is_file():
            try:
                with open(s10_meta, encoding="utf-8") as f:
                    h10 = json.load(f).get("cache_key", "")
            except Exception:
                pass
        if s04_meta.is_file():
            try:
                with open(s04_meta, encoding="utf-8") as f:
                    h4 = json.load(f).get("cache_key", "")
            except Exception:
                pass
        if s11_meta.is_file():
            try:
                with open(s11_meta, encoding="utf-8") as f:
                    h11 = json.load(f).get("cache_key", "")
            except Exception:
                pass

        raw = f"{h10}::{h4}::{h11}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        return hashlib.sha256(
            json.dumps(config.output.model_dump(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        # 1. Load candidates from S10
        candidates_file = project.stages_dir / "s10_candidates" / "candidates.jsonl"
        candidates: list[Candidate] = []
        if candidates_file.is_file():
            with open(candidates_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates.append(Candidate.model_validate_json(line))

        # 2. Load events from S10
        events_file = project.stages_dir / "s10_candidates" / "events.json"
        events: list[Event] = []
        if events_file.is_file():
            with open(events_file, encoding="utf-8") as f:
                data = json.load(f)
                events = [Event.model_validate(e) for e in data.get("events", [])]

        # 3. Load transcript segments from S04
        transcript_file = project.stages_dir / "s04_asr" / "transcript.json"
        segments: list[Segment] = []
        if transcript_file.is_file():
            with open(transcript_file, encoding="utf-8") as f:
                tdata = json.load(f)
                raw_segs = tdata if isinstance(tdata, list) else tdata.get("segments", [])
                segments = [Segment.model_validate(s) for s in raw_segs]

        # 4. Load VAD intervals from S03
        vad_file = project.stages_dir / "s03_vad_lid" / "speech_segments.json"
        if not vad_file.is_file():
            vad_file = project.stages_dir / "s03_vad_lid" / "vad_segments.json"
        vad_intervals: list[tuple[float, float]] = []
        if vad_file.is_file():
            with open(vad_file, encoding="utf-8") as f:
                vdata = json.load(f)
                v_items = vdata if isinstance(vdata, list) else vdata.get("speech_segments", [])
                for item in v_items:
                    vad_intervals.append((float(item["start"]), float(item["end"])))

        # 5. Load VOD probe info
        duration_s = float(project.metadata.vod.duration_s or 0.0)
        vod_path_str = str(project.vod_path)
        sha1_head_str = project.metadata.vod.sha1_head

        # 6. Check for LLM Judge verdicts from S11
        verdicts_file = project.stages_dir / "s11_judge" / "judge_verdicts.jsonl"
        verdicts_map: dict[str, tuple[JudgeVerdict, list[str]]] = {}
        judge_cost_usd = 0.0
        judge_model_name = "rule_based_draft"

        if verdicts_file.is_file():
            with open(verdicts_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        flags = row.pop("flags", [])
                        v = JudgeVerdict.model_validate(row)
                        verdicts_map[v.candidate_id] = (v, flags)

            # Read summary for cost & model
            summary_file = project.stages_dir / "s11_judge" / "judge_summary.json"
            if summary_file.is_file():
                try:
                    sdata = json.loads(summary_file.read_text(encoding="utf-8"))
                    judge_cost_usd = float(sdata.get("cost_usd", 0.0))
                except Exception:
                    pass
            prof = project.config.judge.profiles.get(project.config.judge.profile)
            if prof:
                judge_model_name = prof.model

        moments = []
        if verdicts_map:
            # Build moments using LLM verdicts
            moments = build_moments_with_judge(
                candidates=candidates,
                verdicts=verdicts_map,
                segments=segments,
                output_cfg=project.config.output,
                duration_s=duration_s,
                vad_intervals=vad_intervals,
            )

        if not moments:
            # Fallback to rule-based moments if LLM produced no accepted moments
            moments = build_rule_based_moments(
                candidates=candidates,
                segments=segments,
                events=events,
                output_cfg=project.config.output,
                duration_s=duration_s,
                vad_intervals=vad_intervals,
            )

        # 7. Construct MomentsDocument according to TZ 11.3
        vod_meta = VodMeta(
            path=vod_path_str,
            duration_s=duration_s,
            sha1_head=sha1_head_str,
        )
        run_meta = RunMeta(
            app_version="0.1.0",
            config_hash=project.config.compute_hash(),
            asr_backend=project.config.asr.backend,
            judge_model=judge_model_name,
            cost_usd=judge_cost_usd,
            finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        doc = MomentsDocument(
            schema_version="1.0",
            vod=vod_meta,
            run=run_meta,
            moments=moments,
        )

        # 8. Save moments.json inside stage dir and project root
        moments_stage_path = output_dir / "moments.json"
        export_moments_json(doc, moments_stage_path)

        moments_root_path = project.project_dir / "moments.json"
        export_moments_json(doc, moments_root_path)

        # 9. Save moments.csv and youtube_timestamps.txt in project root
        csv_path = project.project_dir / "moments.csv"
        export_moments_csv(doc, csv_path)

        yt_path = project.project_dir / "youtube_timestamps.txt"
        export_youtube_timestamps(doc, yt_path)

        # 10. Generate standalone report.html in project root
        report_path = render_html_report(project.project_dir)

        return {
            "num_candidates": len(candidates),
            "num_moments": len(moments),
            "used_judge": bool(verdicts_map),
            "judge_model": judge_model_name,
            "judge_cost_usd": judge_cost_usd,
            "moments_json": str(moments_root_path),
            "moments_csv": str(csv_path),
            "youtube_timestamps": str(yt_path),
            "report_html": str(report_path),
            "categories": [m.category for m in moments],
        }
