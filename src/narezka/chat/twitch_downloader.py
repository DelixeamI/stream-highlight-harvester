"""Parser for TwitchDownloader JSON chat export files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from narezka.chat.base import ChatMessage, ChatSource

logger = logging.getLogger(__name__)


class TwitchDownloaderJSON(ChatSource):
    """Parses TwitchDownloader JSON chat files."""

    def load_messages(self, path: Path) -> list[ChatMessage]:
        if not path.is_file():
            raise FileNotFoundError(f"Chat file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to parse chat JSON from %s: %s", path, e)
            return []

        raw_comments: list[dict[str, Any]] = []
        if isinstance(data, dict):
            raw_comments = data.get("comments", [])
        elif isinstance(data, list):
            raw_comments = data
        else:
            logger.warning("Unrecognized chat JSON format in %s", path)
            return []

        messages: list[ChatMessage] = []

        for item in raw_comments:
            if not isinstance(item, dict):
                continue

            # Offset timestamp
            offset = item.get("content_offset_seconds")
            if offset is None:
                offset = item.get("offset")
            if offset is None:
                offset = item.get("time_in_seconds")
            if offset is None:
                continue

            try:
                t_sec = float(offset)
            except (ValueError, TypeError):
                continue

            # Author
            commenter = item.get("commenter")
            author = "anonymous"
            if isinstance(commenter, dict):
                author = commenter.get("display_name") or commenter.get("name") or "anonymous"
            elif isinstance(item.get("user"), str):
                author = item["user"]

            # Message body
            msg = item.get("message")
            text = ""
            if isinstance(msg, dict):
                text = msg.get("body") or ""
            elif isinstance(item.get("body"), str):
                text = item["body"]
            elif isinstance(item.get("text"), str):
                text = item["text"]

            if text:
                messages.append(ChatMessage(timestamp_s=round(t_sec, 3), author=author, text=text))

        messages.sort(key=lambda m: m.timestamp_s)
        logger.info("Loaded %d chat messages from %s", len(messages), path)
        return messages
