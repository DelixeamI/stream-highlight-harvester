"""Protocols and data structures for chat processing."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Normalized chat message with timestamp offset in seconds from stream start."""

    timestamp_s: float = Field(ge=0.0)
    author: str
    text: str


class ChatSource(Protocol):
    """Protocol for parsing chat messages from diverse dump formats."""

    def load_messages(self, path: Path) -> list[ChatMessage]:
        """Load and parse chat messages sorted by timestamp in seconds."""
        ...
