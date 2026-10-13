"""Pydantic schemas for LLM Structured Outputs (Judge and Semantic Scan)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SemanticItem(BaseModel):
    """A semantic speech event detected during transcript scanning."""

    start_seg: int = Field(description="Initial segment ID of the lead-in phrase.")
    end_seg: int = Field(description="Final segment ID of the punchline or story.")
    type: Literal["funny_joke", "story", "irony", "meme", "other"] = Field(
        default="funny_joke",
        description="Type of speech interest.",
    )
    strength: int = Field(ge=1, le=5, description="Strength from 1 to 5 (4-5 is a must-have highlight).")
    why: str = Field(description="One sentence in Russian explaining why this is interesting.")


class SemanticScanResult(BaseModel):
    """List of semantic events found in an 8-minute transcript window."""

    items: list[SemanticItem] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Structured decision output produced by the LLM Judge for a candidate window."""

    candidate_id: str
    verdict: Literal["keep", "reject"]
    category: Literal["multikill", "epic_play", "funny_joke", "funny_reaction", "fail", "other"]
    score: int = Field(ge=0, le=10, description="Quality score from 0 to 10 (7+ is worthy of publishing).")
    confidence: float = Field(ge=0.0, le=1.0, description="Judge certainty score from 0.0 to 1.0.")
    title_ru: str = Field(max_length=60, description="Russian title, max 60 chars, no caps, no emojis.")
    reason_ru: str = Field(description="1-2 sentences explaining why this moment is kept or rejected.")
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Exact verbatim speech quotes from the transcript window.",
    )
    start_seg: int = Field(description="Segment ID where clip lead-in begins.")
    end_seg: int = Field(description="Segment ID where clip reaction ends.")
    needs_visual_check: bool = Field(
        default=False,
        description="True if video image is required to confirm highlight quality.",
    )
