"""Base Stage abstraction with deterministic disk caching and GPU tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from narezka.config import AppConfig
from narezka.project import Project
from narezka.utils.gpu import track_vram

logger = logging.getLogger(__name__)


class StageMeta(BaseModel):
    """Metadata recorded in meta.json for each executed stage."""

    stage_name: str
    code_version: str
    input_hash: str
    params_hash: str
    cache_key: str
    started_at: str
    finished_at: str
    duration_s: float
    vram_peak_mb: float = 0.0
    status: str = "completed"
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    """Execution result of a stage."""

    stage_name: str
    is_cached: bool
    output_dir: Path
    duration_s: float
    metadata: StageMeta
    outputs: dict[str, Any] = Field(default_factory=dict)


class Stage(ABC):
    """Abstract base class for all deterministic pipeline stages."""

    name: str = "base_stage"
    code_version: str = "1.0.0"
    is_gpu_stage: bool = False

    def get_stage_dir(self, project: Project) -> Path:
        """Directory where this stage stores its outputs and meta.json."""
        stage_dir = project.stages_dir / self.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir

    def get_meta_file(self, project: Project) -> Path:
        return self.get_stage_dir(project) / "meta.json"

    @abstractmethod
    def compute_input_hash(self, project: Project) -> str:
        """Compute SHA256 of all inputs (files, previous stage outputs) for this stage."""
        raise NotImplementedError

    @abstractmethod
    def compute_params_hash(self, config: AppConfig) -> str:
        """Compute SHA256 of relevant configuration parameters for this stage."""
        raise NotImplementedError

    def compute_cache_key(self, input_hash: str, params_hash: str) -> str:
        """Compute composite cache key combining stage name, code version, inputs, and parameters."""
        raw = f"{self.name}::{self.code_version}::{input_hash}::{params_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_cached(self, project: Project, force: bool = False) -> tuple[bool, StageMeta | None]:
        """Verify whether stage outputs are already valid in disk cache."""
        if force:
            return False, None

        meta_file = self.get_meta_file(project)
        if not meta_file.is_file():
            return False, None

        try:
            with open(meta_file, encoding="utf-8") as f:
                meta_dict = json.load(f)
            meta = StageMeta.model_validate(meta_dict)

            if meta.status != "completed":
                return False, None

            expected_input_hash = self.compute_input_hash(project)
            expected_params_hash = self.compute_params_hash(project.config)
            expected_key = self.compute_cache_key(expected_input_hash, expected_params_hash)

            if meta.cache_key == expected_key:
                return True, meta

            logger.debug(
                "Stage '%s' cache invalidated: key mismatch (cached=%s, current=%s)",
                self.name,
                meta.cache_key[:8],
                expected_key[:8],
            )
            return False, None
        except Exception as exc:
            logger.warning("Failed to validate cache for stage '%s': %s", self.name, exc)
            return False, None

    def execute(self, project: Project, force: bool = False) -> StageResult:
        """Run the stage with caching, timings, metadata saving, and GPU hygiene."""
        stage_dir = self.get_stage_dir(project)
        is_cached, cached_meta = self.is_cached(project, force=force)

        if is_cached and cached_meta is not None:
            logger.info("[SKIP] Стадия '%s' пропущена (результат в кеше)", self.name)
            return StageResult(
                stage_name=self.name,
                is_cached=True,
                output_dir=stage_dir,
                duration_s=cached_meta.duration_s,
                metadata=cached_meta,
            )

        logger.info("[RUN] Запуск стадии '%s'...", self.name)
        input_hash = self.compute_input_hash(project)
        params_hash = self.compute_params_hash(project.config)
        cache_key = self.compute_cache_key(input_hash, params_hash)

        started_at = datetime.now().isoformat()
        start_time = time.perf_counter()
        vram_peak_mb = 0.0

        try:
            if self.is_gpu_stage:
                with track_vram(self.name) as vram_metrics:
                    outputs = self.run(project, stage_dir)
                    vram_peak_mb = vram_metrics.get("peak_mb", 0.0)
            else:
                outputs = self.run(project, stage_dir)

            end_time = time.perf_counter()
            duration_s = round(end_time - start_time, 2)
            finished_at = datetime.now().isoformat()

            meta = StageMeta(
                stage_name=self.name,
                code_version=self.code_version,
                input_hash=input_hash,
                params_hash=params_hash,
                cache_key=cache_key,
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration_s,
                vram_peak_mb=vram_peak_mb,
                status="completed",
            )

            # Save meta.json
            with open(self.get_meta_file(project), "w", encoding="utf-8") as f:
                f.write(meta.model_dump_json(indent=2))

            logger.info(
                "[OK] Стадия '%s' завершена за %.2f с (VRAM peak: %.1f MB)",
                self.name,
                duration_s,
                vram_peak_mb,
            )

            return StageResult(
                stage_name=self.name,
                is_cached=False,
                output_dir=stage_dir,
                duration_s=duration_s,
                metadata=meta,
                outputs=outputs or {},
            )
        except Exception as exc:
            end_time = time.perf_counter()
            duration_s = round(end_time - start_time, 2)
            finished_at = datetime.now().isoformat()

            meta = StageMeta(
                stage_name=self.name,
                code_version=self.code_version,
                input_hash=input_hash,
                params_hash=params_hash,
                cache_key=cache_key,
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration_s,
                vram_peak_mb=vram_peak_mb,
                status="failed",
                error=str(exc),
            )
            # Write failed metadata so subsequent runs know it failed
            with open(self.get_meta_file(project), "w", encoding="utf-8") as f:
                f.write(meta.model_dump_json(indent=2))

            logger.error("[FAIL] Ошибка при выполнении стадии '%s': %s", self.name, exc)
            raise

    @abstractmethod
    def run(self, project: Project, output_dir: Path) -> dict[str, Any] | None:
        """Stage execution logic. Must be implemented by concrete stages."""
        raise NotImplementedError
