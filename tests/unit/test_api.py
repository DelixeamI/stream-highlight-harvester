"""Unit tests for FastAPI REST API endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from narezka.api.app import _jobs, create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    _jobs.clear()
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "cuda_available" in data
    assert "python_version" in data


def test_list_jobs_empty(client: TestClient) -> None:
    resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_nonexistent_job(client: TestClient) -> None:
    resp = client.get("/api/v1/jobs/nonexistent123")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_submit_job_missing_video(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/jobs",
        json={"video_path": "Z:/nonexistent_broadcast.mp4"},
    )
    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"]


def test_submit_job_and_poll_status(client: TestClient, tmp_path: Path) -> None:
    video_file = tmp_path / "test_vod.mp4"
    video_file.write_text("fake video content", encoding="utf-8")

    with patch("narezka.api.app._execute_job_sync"):
        resp = client.post(
            "/api/v1/jobs",
            json={"video_path": str(video_file), "game": "Deadlock"},
        )
        assert resp.status_code == 202
        data = resp.json()
        job_id = data["job_id"]
        assert data["status"] == "queued"
        assert data["video_path"] == str(video_file)

        # Verify job shows up in list and get endpoint
        get_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["job_id"] == job_id

        list_resp = client.get("/api/v1/jobs")
        assert len(list_resp.json()) == 1


def test_job_report_endpoint(client: TestClient, tmp_path: Path) -> None:
    video_file = tmp_path / "test_vod.mp4"
    video_file.write_text("fake video", encoding="utf-8")

    with patch("narezka.api.app._execute_job_sync"):
        resp = client.post(
            "/api/v1/jobs",
            json={"video_path": str(video_file)},
        )
        job_id = resp.json()["job_id"]

        # Job not finished / no project dir -> 400
        report_resp = client.get(f"/api/v1/jobs/{job_id}/report")
        assert report_resp.status_code == 400

        # Simulate finished job with report.html
        job_record = _jobs[job_id]
        job_record.project_dir = tmp_path
        report_file = tmp_path / "report.html"
        report_file.write_text("<html><body>Dashboard</body></html>", encoding="utf-8")

        ok_report = client.get(f"/api/v1/jobs/{job_id}/report")
        assert ok_report.status_code == 200
        assert "<html><body>Dashboard</body></html>" in ok_report.text


def test_correlation_id_middleware(client: TestClient) -> None:
    # Auto-generate request ID
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert "X-Process-Time-Ms" in resp.headers
    assert float(resp.headers["X-Process-Time-Ms"]) >= 0.0

    # Propagate provided request ID
    custom_id = "trace-req-778899"
    resp_custom = client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp_custom.status_code == 200
    assert resp_custom.headers["X-Request-ID"] == custom_id


def test_cli_openapi_export(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from narezka.cli import app as cli_app

    runner = CliRunner()
    out_file = tmp_path / "openapi_test.json"
    result = runner.invoke(cli_app, ["openapi", "--output", str(out_file)])
    assert result.exit_code == 0
    assert out_file.is_file()

    spec = json.loads(out_file.read_text(encoding="utf-8"))
    assert spec["openapi"].startswith("3.")
    assert "/health" in spec["paths"]
    assert "/api/v1/jobs" in spec["paths"]


def test_submit_job_forbidden_path_outside_whitelist(client: TestClient, tmp_path: Path) -> None:
    video_file = tmp_path / "test_vod.mp4"
    video_file.write_text("fake video content", encoding="utf-8")

    with patch("narezka.api.app.get_allowed_vod_roots", return_value=[Path("C:/strictly_allowed_folder")]):
        resp = client.post(
            "/api/v1/jobs",
            json={"video_path": str(video_file)},
        )
        assert resp.status_code == 403
        assert "forbidden" in resp.json()["detail"].lower()


def test_submit_job_invalid_extension(client: TestClient, tmp_path: Path) -> None:
    bad_file = tmp_path / "malicious.exe"
    bad_file.write_text("binary", encoding="utf-8")

    resp = client.post(
        "/api/v1/jobs",
        json={"video_path": str(bad_file)},
    )
    assert resp.status_code == 400
    assert "allowed extensions" in resp.json()["detail"].lower()



