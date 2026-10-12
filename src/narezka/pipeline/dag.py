"""Linear and sequential DAG pipeline runner with stage filtering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narezka.pipeline.stage import Stage, StageResult
    from narezka.project import Project

logger = logging.getLogger(__name__)


class PipelineDAG:
    """Manages the registered pipeline stages and executes them sequentially."""

    def __init__(self) -> None:
        self._stages: list[Stage] = []

    def register(self, stage: Stage) -> None:
        """Register a stage in the execution order."""
        self._stages.append(stage)

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages]

    def run(
        self,
        project: Project,
        until_stage: str | None = None,
        force_stage: str | None = None,
    ) -> list[StageResult]:
        """Execute registered stages up to `until_stage`, optionally forcing rerun of `force_stage`."""
        results: list[StageResult] = []

        if until_stage and until_stage not in self.stage_names:
            raise ValueError(
                f"Целевая стадия '{until_stage}' не найдена в пайплайне. Доступные стадии: {', '.join(self.stage_names)}"
            )

        if force_stage and force_stage not in self.stage_names:
            raise ValueError(
                f"Стадия для принудительного перезапуска '{force_stage}' не найдена в пайплайне."
            )

        force_cascade = False
        for stage in self._stages:
            if force_stage and force_stage == stage.name:
                force_cascade = True
            result = stage.execute(project, force=force_cascade)
            results.append(result)

            if until_stage and stage.name == until_stage:
                logger.info("⏸ Выполнение остановлено на стадии '%s' (--until-stage)", until_stage)
                break

        return results
