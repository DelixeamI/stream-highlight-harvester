"""LLM Judge coordinator for evaluating highlight candidate windows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from narezka.asr.base import Segment
from narezka.config import JudgeConfig
from narezka.judge.context_builder import build_candidate_prompt_context, format_seconds_time
from narezka.judge.frames import extract_candidate_keyframes
from narezka.judge.validation import validate_judge_verdict
from narezka.llm.client import LLMClient
from narezka.llm.schemas import JudgeVerdict
from narezka.models import Candidate, Event

logger = logging.getLogger(__name__)


class LLMJudge:
    """Evaluates candidates using structured LLM completions and anti-hallucination validation."""

    def __init__(
        self,
        judge_config: JudgeConfig,
        cache_dir: Path | None = None,
        game_name: str = "Deadlock",
        prompt_file: Path | None = None,
        examples_file: Path | None = None,
    ) -> None:
        self.config = judge_config
        self.game_name = game_name

        profile_key = judge_config.profile
        profile = judge_config.profiles.get(profile_key)
        if not profile:
            raise ValueError(f"Judge profile '{profile_key}' not found in configuration")

        self.client = LLMClient(
            profile_config=profile,
            cache_dir=cache_dir,
            max_cost_usd=judge_config.max_cost_usd,
        )

        # Check if multimodal VLM mode should be enabled
        model_name = profile.model.lower()
        self.is_vlm = (
            profile_key in ["cloud", "openrouter"]
            or "gemini" in model_name
            or "vl" in model_name
            or "vision" in model_name
        )

        # Load system prompt (prefer judge_vlm_ru.md for VLM)
        vlm_path = Path(__file__).resolve().parents[3] / "prompts" / "judge_vlm_ru.md"
        p_path = prompt_file or Path(judge_config.prompt_file)
        if self.is_vlm and vlm_path.is_file() and not prompt_file:
            p_path = vlm_path
        elif not p_path.is_file():
            p_path = Path(__file__).resolve().parents[3] / "prompts" / "judge_ru.md"
        self.system_prompt = p_path.read_text(encoding="utf-8") if p_path.is_file() else "Ты — судья хайлайтов."

        # Load few-shot examples
        e_path = examples_file or Path(judge_config.examples_file)
        if not e_path.is_file():
            e_path = Path(__file__).resolve().parents[3] / "prompts" / "judge_examples.yaml"
        self.examples: list[dict[str, Any]] = []
        if e_path.is_file():
            try:
                data = yaml.safe_load(e_path.read_text(encoding="utf-8")) or {}
                self.examples = data.get("examples", [])
            except Exception as e:
                logger.warning("Could not load judge examples from %s: %s", e_path, e)

    def evaluate_candidate(
        self,
        candidate: Candidate,
        segments: list[Segment],
        events: list[Event],
        streamer_hero: str | None = None,
        vod_path: Path | None = None,
    ) -> tuple[JudgeVerdict, list[str]]:
        """Evaluate a single candidate window with the LLM / VLM Judge."""
        context_str = build_candidate_prompt_context(
            candidate=candidate,
            segments=segments,
            events=events,
            game_name=self.game_name,
            streamer_hero=streamer_hero,
        )

        frames: list[tuple[float, str]] = []
        if self.is_vlm and vod_path and vod_path.is_file():
            try:
                frames = extract_candidate_keyframes(
                    vod_path=vod_path,
                    start_s=candidate.start,
                    end_s=candidate.end,
                    peak_s=candidate.peak,
                    num_frames=8,
                )
            except Exception as e:
                logger.warning("Frame extraction failed for candidate %s: %s", candidate.id, e)

        messages: list[dict[str, Any]]
        if frames:
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": context_str}
            ]
            for f_t, b64_img in frames:
                user_content.append({"type": "text", "text": f"\n[Кадр геймплея на {format_seconds_time(f_t)}]:"})
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            # Add few-shot examples if available
            for ex in self.examples[:2]:
                ex_verdict = {k: v for k, v in ex.items() if k != "id"}
                messages.append({"role": "user", "content": f"Пример кандидата:\nКАНДИДАТ {ex.get('candidate_id', 'c_0001')}"})
                messages.append({"role": "assistant", "content": json.dumps(ex_verdict, ensure_ascii=False)})
            messages.append({"role": "user", "content": context_str})

        raw_verdict, meta = self.client.generate_json(
            messages=messages,
            schema_cls=JudgeVerdict,
            temperature=0.1,
            max_retries=2,
        )

        # Ensure candidate_id matches
        raw_verdict.candidate_id = candidate.id

        # Validate and sanitize verdict
        validated_verdict, flags = validate_judge_verdict(
            verdict=raw_verdict,
            candidate=candidate,
            segments=segments,
            events=events,
        )

        logger.info(
            "Candidate %s verdict: %s (%s, score=%d, conf=%.2f, flags=%s)",
            candidate.id,
            validated_verdict.verdict,
            validated_verdict.category,
            validated_verdict.score,
            validated_verdict.confidence,
            flags,
        )
        return validated_verdict, flags

    def evaluate_batch(
        self,
        candidates: list[Candidate],
        segments: list[Segment],
        events: list[Event],
        streamer_hero: str | None = None,
        max_candidates: int | None = None,
        vod_path: Path | None = None,
    ) -> list[tuple[JudgeVerdict, list[str]]]:
        """Evaluate list of candidates sequentially with budget protection."""
        limit = max_candidates or self.config.max_candidates
        selected_candidates = candidates[:limit]

        results: list[tuple[JudgeVerdict, list[str]]] = []
        for cand in selected_candidates:
            try:
                v, flags = self.evaluate_candidate(
                    candidate=cand,
                    segments=segments,
                    events=events,
                    streamer_hero=streamer_hero,
                    vod_path=vod_path,
                )
                results.append((v, flags))
            except Exception as e:
                logger.error("Failed to evaluate candidate %s: %s", cand.id, e)
                # Fallback reject verdict
                fallback = JudgeVerdict(
                    candidate_id=cand.id,
                    verdict="reject",
                    category="other",
                    score=0,
                    confidence=0.0,
                    title_ru=f"Ошибка оценки {cand.id}",
                    reason_ru=f"Ошибка LLM: {e}",
                    evidence_quotes=[],
                    start_seg=0,
                    end_seg=0,
                )
                results.append((fallback, ["llm_error"]))

        return results
