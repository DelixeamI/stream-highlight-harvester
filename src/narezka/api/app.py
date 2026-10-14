"""FastAPI production service for Stream Highlight Harvester.

Provides asynchronous REST endpoints for running and monitoring VOD analysis jobs,
health check inspection, and serving generated HTML dashboards.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from narezka import __version__
from narezka.api.middleware import RequestCorrelationMiddleware
from narezka.cli import build_pipeline_dag
from narezka.config import load_config
from narezka.project import Project

logger = logging.getLogger("narezka.api")


# --- Pydantic Schemas ---

class JobSubmitRequest(BaseModel):
    """Payload for submitting a new stream highlight processing job."""

    video_path: str = Field(..., description="Absolute or relative path to broadcast video file.")
    chat_path: str | None = Field(None, description="Optional path to Twitch/Kick chat JSON file.")
    game: str | None = Field(None, description="Game title override (Counter-Strike 2, Deadlock, Dota 2).")
    until_stage: str | None = Field("s12_clips", description="Stop pipeline after stage (default: s12_clips).")
    force_stage: str | None = Field(None, description="Stage name to rerun forcefully ignoring cache.")


class JobStatusResponse(BaseModel):
    """Current state and execution metrics of an analysis job."""

    job_id: str
    video_path: str
    project_dir: str | None = None
    status: Literal["queued", "running", "completed", "failed"]
    current_stage: str | None = None
    progress_percent: float = 0.0
    moments_count: int = 0
    created_at: float
    completed_at: float | None = None
    error: str | None = None
    report_url: str | None = None


class HealthResponse(BaseModel):
    """System health diagnostics and GPU hardware inspection."""

    status: str
    version: str
    cuda_available: bool
    gpu_name: str | None
    vram_total_mb: float | None
    python_version: str
    timestamp: float


# --- In-Memory Job Store & Worker ---

class JobRecord:
    def __init__(self, job_id: str, request: JobSubmitRequest) -> None:
        self.job_id = job_id
        self.request = request
        self.status: Literal["queued", "running", "completed", "failed"] = "queued"
        self.current_stage: str | None = None
        self.progress_percent: float = 0.0
        self.project_dir: Path | None = None
        self.moments_count: int = 0
        self.created_at: float = time.time()
        self.completed_at: float | None = None
        self.error: str | None = None

    def to_response(self) -> JobStatusResponse:
        report_url = f"/api/v1/jobs/{self.job_id}/report" if self.status == "completed" else None
        return JobStatusResponse(
            job_id=self.job_id,
            video_path=self.request.video_path,
            project_dir=str(self.project_dir) if self.project_dir else None,
            status=self.status,
            current_stage=self.current_stage,
            progress_percent=self.progress_percent,
            moments_count=self.moments_count,
            created_at=self.created_at,
            completed_at=self.completed_at,
            error=self.error,
            report_url=report_url,
        )


_jobs: dict[str, JobRecord] = {}


def _execute_job_sync(job: JobRecord) -> None:
    """Synchronous pipeline execution runner invoked within a thread pool."""
    job.status = "running"
    job.progress_percent = 5.0
    vod = Path(job.request.video_path)

    if not vod.is_file():
        job.status = "failed"
        job.error = f"Video file not found: {vod}"
        job.completed_at = time.time()
        logger.error("Job %s failed: %s", job.job_id, job.error)
        return

    overrides: dict[str, Any] = {}
    if job.request.chat_path:
        chat = Path(job.request.chat_path)
        if chat.is_file():
            overrides["chat"] = {"path": str(chat.resolve())}

    detected_game = job.request.game
    if not detected_game:
        stem = vod.stem.lower()
        if "cs2" in stem or "cs" in stem or "counter" in stem:
            detected_game = "Counter-Strike 2"
        elif "deadlock" in stem:
            detected_game = "Deadlock"
        elif "dota" in stem:
            detected_game = "Dota 2"
    if detected_game:
        overrides["game"] = detected_game

    try:
        cfg = load_config(overrides=overrides if overrides else None)
        project = Project.create(vod_path=vod, config=cfg)
        job.project_dir = project.project_dir

        if job.request.chat_path:
            chat_p = Path(job.request.chat_path)
            if chat_p.is_file():
                target_chat = project.project_dir / "chat.json"
                if not target_chat.is_file() or target_chat.resolve() != chat_p.resolve():
                    shutil.copy2(chat_p, target_chat)

        dag = build_pipeline_dag()
        target_until = job.request.until_stage or "s12_clips"

        job.current_stage = "running_dag"
        job.progress_percent = 20.0

        dag.run(project, until_stage=target_until, force_stage=job.request.force_stage)

        moments_file = project.project_dir / "moments.json"
        if moments_file.is_file():
            import json

            try:
                with open(moments_file, encoding="utf-8") as f:
                    moments_data = json.load(f)
                    job.moments_count = len(moments_data.get("moments", []))
            except Exception:
                pass

        job.status = "completed"
        job.current_stage = "done"
        job.progress_percent = 100.0
        job.completed_at = time.time()
        logger.info("Job %s completed successfully with %d moments", job.job_id, job.moments_count)

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.completed_at = time.time()
        logger.exception("Job %s execution failed: %s", job.job_id, exc)


# --- FastAPI Application Factory ---

def create_app() -> FastAPI:
    """Build and configure the FastAPI ASGI application."""
    app_instance = FastAPI(
        title="Stream Highlight Harvester API",
        description="Asynchronous multimodal pipeline REST service for long-form gaming VOD highlight extraction.",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app_instance.add_middleware(RequestCorrelationMiddleware)

    @app_instance.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
    async def health_check() -> HealthResponse:
        """Inspect service health, CUDA GPU acceleration, and environment version."""
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        vram_mb = None
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            vram_mb = round(props.total_memory / (1024 * 1024), 1)

        return HealthResponse(
            status="ok",
            version=__version__,
            cuda_available=cuda_ok,
            gpu_name=gpu_name,
            vram_total_mb=vram_mb,
            python_version=platform.python_version(),
            timestamp=time.time(),
        )

    @app_instance.post(
        "/api/v1/jobs",
        response_model=JobStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Jobs"],
    )
    async def submit_job(req: JobSubmitRequest, background_tasks: BackgroundTasks) -> JobStatusResponse:
        """Submit a VOD analysis job for background execution."""
        vod_path = Path(req.video_path)
        if not vod_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video file does not exist: {req.video_path}",
            )

        job_id = uuid.uuid4().hex[:12]
        job = JobRecord(job_id=job_id, request=req)
        _jobs[job_id] = job

        background_tasks.add_task(asyncio.to_thread, _execute_job_sync, job)
        logger.info("Enqueued job %s for video %s", job_id, req.video_path)
        return job.to_response()

    @app_instance.get("/api/v1/jobs", response_model=list[JobStatusResponse], tags=["Jobs"])
    async def list_jobs() -> list[JobStatusResponse]:
        """List all submitted jobs and their current execution status."""
        return [job.to_response() for job in _jobs.values()]

    @app_instance.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
    async def get_job_status(job_id: str) -> JobStatusResponse:
        """Fetch status and progress for a specific job."""
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        return job.to_response()

    @app_instance.get("/api/v1/jobs/{job_id}/report", response_class=HTMLResponse, tags=["Jobs"])
    async def get_job_report(job_id: str) -> HTMLResponse:
        """Serve the standalone interactive HTML analytics dashboard for a completed job."""
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        if not job.project_dir:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job has not generated a project directory yet",
            )

        report_file = job.project_dir / "report.html"
        if not report_file.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="HTML report file not found. Ensure pipeline reached stage s12_clips.",
            )

        content = report_file.read_text(encoding="utf-8")
        return HTMLResponse(content=content)

    return app_instance


app = create_app()
