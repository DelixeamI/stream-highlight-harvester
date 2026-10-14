"""Stage S11: LLM Judge evaluation of highlight candidate windows."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from narezka.asr.base import Segment
from narezka.config import AppConfig
from narezka.judge.judge import LLMJudge
from narezka.models import Candidate, Event
from narezka.pipeline.stage import Stage
from narezka.project import Project

logger = logging.getLogger(__name__)


class S11JudgeStage(Stage):
    """Run LLM Judge to evaluate candidate highlight windows."""

    name = "s11_judge"
    code_version = "1.2.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        s10_meta = project.stages_dir / "s10_candidates" / "meta.json"
        s04_meta = project.stages_dir / "s04_asr" / "meta.json"
        h10 = ""
        h4 = ""
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

        return hashlib.sha256(f"{h10}::{h4}".encode()).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "judge": config.judge.model_dump(),
            "game": config.game,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        # 1. Load candidates from S10
        cand_file = project.stages_dir / "s10_candidates" / "candidates.jsonl"
        candidates: list[Candidate] = []
        if cand_file.is_file():
            with open(cand_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates.append(Candidate.model_validate_json(line))

        # 2. Load transcript segments from S04
        transcript_file = project.stages_dir / "s04_asr" / "transcript.json"
        segments: list[Segment] = []
        if transcript_file.is_file():
            with open(transcript_file, encoding="utf-8") as f:
                tdata = json.load(f)
                raw_segs = tdata if isinstance(tdata, list) else tdata.get("segments", [])
                segments = [Segment.model_validate(s) for s in raw_segs]

        # 3. Load events from S10
        events_file = project.stages_dir / "s10_candidates" / "events.json"
        events: list[Event] = []
        if events_file.is_file():
            with open(events_file, encoding="utf-8") as f:
                edata = json.load(f)
                events = [Event.model_validate(e) for e in edata.get("events", [])]

        # 4. Initialize Judge
        judge = LLMJudge(
            judge_config=project.config.judge,
            cache_dir=project.llm_cache_dir,
            game_name=project.config.game,
        )

        # 5. Evaluate batch of candidates
        eval_results = judge.evaluate_batch(
            candidates=candidates,
            segments=segments,
            events=events,
            max_candidates=project.config.judge.max_candidates,
            vod_path=project.vod_path,
        )

        # 6. Save verdicts JSONL
        verdicts_file = output_dir / "judge_verdicts.jsonl"
        with open(verdicts_file, "w", encoding="utf-8") as f:
            for verdict, flags in eval_results:
                v_dict = verdict.model_dump()
                v_dict["flags"] = flags
                f.write(json.dumps(v_dict, ensure_ascii=False) + "\n")

        num_keep = sum(1 for v, _ in eval_results if v.verdict == "keep")
        num_reject = sum(1 for v, _ in eval_results if v.verdict == "reject")

        # Save summary
        summary = {
            "num_candidates_evaluated": len(eval_results),
            "num_keep": num_keep,
            "num_reject": num_reject,
            "cost_usd": round(judge.client.total_cost_usd, 4),
            "verdicts_path": str(verdicts_file),
        }
        with open(output_dir / "judge_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary
