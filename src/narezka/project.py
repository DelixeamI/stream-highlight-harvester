"""Project directory management, metadata persistence, and fast VOD hashing."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from narezka import __version__
from narezka.config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_HEAD_BYTES = 64 * 1024 * 1024  # 64 MB


def compute_sha1_head(file_path: Path, head_bytes: int = DEFAULT_HEAD_BYTES) -> str:
    """Compute fast SHA1 hash of the first 64 MB plus the total file size.

    Avoids reading 30 GB video files across network or slow storage.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл VOD не найден: {path}")

    file_size = path.stat().st_size
    hasher = hashlib.sha1()

    # Read first N bytes
    with open(path, "rb") as f:
        chunk = f.read(head_bytes)
        hasher.update(chunk)

    # Incorporate total file size into hash
    hasher.update(str(file_size).encode("ascii"))
    return hasher.hexdigest()


class VodMeta(BaseModel):
    path: str
    file_size_bytes: int
    sha1_head: str
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None
    audio_streams_count: int | None = None


class RunMeta(BaseModel):
    app_version: str = __version__
    config_hash: str
    asr_backend: str
    judge_model: str
    cost_usd: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str | None = None


class ProjectMetadata(BaseModel):
    schema_version: str = "1.0"
    vod: VodMeta
    run: RunMeta
    extra: dict[str, Any] = Field(default_factory=dict)


class Project:
    """Manages project folder on disk, stages, and project.json metadata."""

    def __init__(self, root: Path, vod_path: Path, config: AppConfig) -> None:
        self.root = root.resolve()
        self.vod_path = vod_path.resolve()
        self.config = config

        self.stages_dir = self.root / "stages"
        self.clips_dir = self.root / "clips"
        self.logs_dir = self.root / "logs"
        self.llm_cache_dir = self.root / "cache" / "llm"
        self.meta_path = self.root / "project.json"

        self._metadata: ProjectMetadata | None = None

    @property
    def project_dir(self) -> Path:
        """Root directory of the project."""
        return self.root

    def initialize_dirs(self) -> None:
        """Create standard project directory structure."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        vod_path: Path | str,
        config: AppConfig,
        projects_base_dir: Path | str | None = None,
    ) -> Project:
        """Create a new project directory named <vod_stem>_<short_hash>."""
        resolved_vod = Path(vod_path).resolve()
        if not resolved_vod.is_file():
            raise FileNotFoundError(f"Файл VOD не найден: {resolved_vod}")

        sha1_head = compute_sha1_head(resolved_vod)
        short_hash = sha1_head[:8]

        base = Path(projects_base_dir) if projects_base_dir else Path.cwd() / "projects"
        project_dir_name = f"{resolved_vod.stem}_{short_hash}"
        project_root = base / project_dir_name

        project = cls(root=project_root, vod_path=resolved_vod, config=config)
        project.initialize_dirs()

        # Check if project.json exists, otherwise create
        if not project.meta_path.exists():
            vod_meta = VodMeta(
                path=str(resolved_vod),
                file_size_bytes=resolved_vod.stat().st_size,
                sha1_head=sha1_head,
            )
            run_meta = RunMeta(
                config_hash=config.compute_hash(),
                asr_backend=config.asr.backend,
                judge_model=config.judge.profiles[config.judge.profile].model,
            )
            metadata = ProjectMetadata(vod=vod_meta, run=run_meta)
            project.save_metadata(metadata)

        return project

    def load_metadata(self) -> ProjectMetadata:
        """Load project metadata from project.json."""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Файл метаданных проекта не найден: {self.meta_path}")
        with open(self.meta_path, encoding="utf-8") as f:
            data = json.load(f)
        self._metadata = ProjectMetadata.model_validate(data)
        return self._metadata

    def save_metadata(self, metadata: ProjectMetadata | None = None) -> None:
        """Persist project metadata to project.json."""
        if metadata is not None:
            self._metadata = metadata
        if self._metadata is None:
            return

        with open(self.meta_path, "w", encoding="utf-8") as f:
            f.write(self._metadata.model_dump_json(indent=2))

    @property
    def metadata(self) -> ProjectMetadata:
        if self._metadata is None:
            self._metadata = self.load_metadata()
        return self._metadata
