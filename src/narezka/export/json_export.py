"""Export moments to JSON format."""

from __future__ import annotations

from pathlib import Path

from narezka.models import MomentsDocument


def export_moments_json(doc: MomentsDocument, output_path: Path) -> Path:
    """Save MomentsDocument to JSON file with indentation and UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(doc.model_dump_json(indent=2))
    return output_path
