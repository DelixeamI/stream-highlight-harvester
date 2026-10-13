"""Stage S09: Semantic scan over transcript to detect dry humor, irony and punchlines."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from narezka.asr.base import Segment
from narezka.config import AppConfig
from narezka.judge.context_builder import format_seconds_time
from narezka.llm.client import LLMClient
from narezka.llm.schemas import SemanticScanResult
from narezka.models import Event
from narezka.pipeline.stage import Stage
from narezka.project import Project

logger = logging.getLogger(__name__)


class S09SemanticScanStage(Stage):
    """Scan transcript in 8-minute overlapping windows using LLM to find semantic highlight flags."""

    name = "s09_semantic_scan"
    code_version = "1.0.0"
    is_gpu_stage = False

    def compute_input_hash(self, project: Project) -> str:
        s04_meta = project.stages_dir / "s04_asr" / "meta.json"
        s04_hash = ""
        if s04_meta.is_file():
            try:
                with open(s04_meta, encoding="utf-8") as f:
                    s04_hash = json.load(f).get("cache_key", "")
            except Exception:
                pass
        return hashlib.sha256(s04_hash.encode("utf-8")).hexdigest()[:16]

    def compute_params_hash(self, config: AppConfig) -> str:
        data = {
            "semantic_scan": config.semantic_scan.model_dump(),
            "judge_profile": config.judge.profile,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run(self, project: Project, output_dir: Path) -> dict[str, Any]:
        if not project.config.semantic_scan.enabled:
            logger.info("Semantic scan is disabled in configuration, skipping")
            events_file = output_dir / "semantic_events.json"
            events_file.write_text(json.dumps({"events": []}), encoding="utf-8")
            return {"enabled": False, "num_events": 0, "events_path": str(events_file)}

        # Load transcript
        transcript_file = project.stages_dir / "s04_asr" / "transcript.json"
        segments: list[Segment] = []
        if transcript_file.is_file():
            with open(transcript_file, encoding="utf-8") as f:
                data = json.load(f)
                raw_segs = data if isinstance(data, list) else data.get("segments", [])
                segments = [Segment.model_validate(s) for s in raw_segs]

        if not segments:
            events_file = output_dir / "semantic_events.json"
            events_file.write_text(json.dumps({"events": []}), encoding="utf-8")
            return {"enabled": True, "num_events": 0, "events_path": str(events_file)}

        # Setup LLM client
        profile_key = project.config.judge.profile
        profile = project.config.judge.profiles.get(profile_key)
        if not profile:
            raise ValueError(f"Judge profile '{profile_key}' not found in configuration")

        llm_client = LLMClient(
            profile_config=profile,
            cache_dir=project.llm_cache_dir,
            max_cost_usd=project.config.judge.max_cost_usd,
        )

        # Load prompt
        prompt_file = Path("prompts/semantic_scan_ru.md")
        if not prompt_file.is_file():
            prompt_file = Path(__file__).resolve().parents[4] / "prompts" / "semantic_scan_ru.md"
        prompt_text = prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else "Найди шутки в транскрипте."

        # Chunk into 8-min windows with 1-min overlap
        window_s = project.config.semantic_scan.window_min * 60.0
        overlap_s = project.config.semantic_scan.overlap_min * 60.0
        step_s = max(60.0, window_s - overlap_s)

        total_duration = max(s.end for s in segments)
        seg_by_id = {s.id: s for s in segments}

        all_events: list[Event] = []
        w_start = 0.0

        while w_start < total_duration:
            w_end = w_start + window_s
            window_segs = [s for s in segments if not (s.end < w_start or s.start > w_end)]

            if len(window_segs) >= 1:
                # Format window text
                text_lines = []
                for s in window_segs:
                    t_str = format_seconds_time(s.start)
                    spk = "СТРИМЕР" if s.speaker == "streamer" else s.speaker.upper()
                    text_lines.append(f"[seg {s.id:04d} | {t_str}] {spk}: {s.text}")

                window_content = "\n".join(text_lines)
                messages = [
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": f"Окно транскрипта {format_seconds_time(w_start)} – {format_seconds_time(w_end)}:\n\n{window_content}"},
                ]

                try:
                    res, _ = llm_client.generate_json(
                        messages=messages,
                        schema_cls=SemanticScanResult,
                        temperature=0.1,
                    )
                    for item in res.items:
                        s_seg = seg_by_id.get(item.start_seg)
                        e_seg = seg_by_id.get(item.end_seg)
                        if s_seg and e_seg and e_seg.end >= s_seg.start:
                            score = round(min(1.0, item.strength / 5.0), 2)
                            evt_id = f"evt_sem_{int(s_seg.start * 10):06d}"
                            all_events.append(
                                Event(
                                    id=evt_id,
                                    source="semantic",
                                    type="semantic_flag",
                                    start=round(s_seg.start, 2),
                                    end=round(e_seg.end, 2),
                                    score=score,
                                    payload={
                                        "why": item.why,
                                        "subtype": item.type,
                                        "strength": item.strength,
                                        "start_seg": item.start_seg,
                                        "end_seg": item.end_seg,
                                    },
                                )
                            )
                except Exception as exc:
                    logger.warning("Semantic scan window %s - %s failed: %s", w_start, w_end, exc)

            w_start += step_s

        # Deduplicate overlapping semantic events
        all_events.sort(key=lambda e: e.start)
        unique_events: list[Event] = []
        for e in all_events:
            if not unique_events:
                unique_events.append(e)
            else:
                last = unique_events[-1]
                if abs(last.start - e.start) < 5.0 and last.type == e.type:
                    # Keep one with higher score
                    if e.score > last.score:
                        unique_events[-1] = e
                else:
                    unique_events.append(e)

        events_file = output_dir / "semantic_events.json"
        with open(events_file, "w", encoding="utf-8") as f:
            json.dump({"events": [e.model_dump() for e in unique_events]}, f, indent=2, ensure_ascii=False)

        return {
            "enabled": True,
            "num_events": len(unique_events),
            "events_path": str(events_file),
            "cost_usd": round(llm_client.total_cost_usd, 4),
        }
