"""Unit tests for Project directory management and sha1_head hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from narezka.config import AppConfig
from narezka.project import Project, compute_sha1_head


def test_compute_sha1_head(tmp_path: Path) -> None:
    sample_file = tmp_path / "test_file.bin"
    sample_data = b"Hello Narezka!" * 1000
    sample_file.write_bytes(sample_data)

    hash1 = compute_sha1_head(sample_file)
    hash2 = compute_sha1_head(sample_file)

    assert hash1 == hash2
    assert len(hash1) == 40  # SHA1 hex string length

    # Modifying the content should change the hash
    sample_file.write_bytes(b"Modified content" * 1000)
    hash3 = compute_sha1_head(sample_file)
    assert hash3 != hash1


def test_project_structure_creation(tmp_path: Path, sample_config: AppConfig) -> None:
    vod_file = tmp_path / "stream_2026_09_28.mp4"
    vod_file.write_bytes(b"\x00" * 4096)

    projects_dir = tmp_path / "projects"
    project = Project.create(
        vod_path=vod_file,
        config=sample_config,
        projects_base_dir=projects_dir,
    )

    assert project.root.exists()
    assert project.stages_dir.exists()
    assert project.clips_dir.exists()
    assert project.logs_dir.exists()
    assert project.llm_cache_dir.exists()
    assert project.meta_path.exists()

    # Verify project.json contents
    metadata = project.load_metadata()
    assert metadata.schema_version == "1.0"
    assert metadata.vod.path == str(vod_file.resolve())
    assert metadata.vod.file_size_bytes == 4096
    assert len(metadata.vod.sha1_head) == 40


def test_project_paths_with_spaces_and_cyrillic(tmp_path: Path, sample_config: AppConfig) -> None:
    # Test path robustness on Windows
    cyrillic_dir = tmp_path / "Тестовая папка со стримами"
    cyrillic_dir.mkdir(parents=True, exist_ok=True)
    vod_file = cyrillic_dir / "Запись стрима 28.09.2026.mp4"
    vod_file.write_bytes(b"\x01" * 8192)

    project = Project.create(
        vod_path=vod_file,
        config=sample_config,
        projects_base_dir=tmp_path / "Папка проектов",
    )

    assert project.root.exists()
    assert project.meta_path.is_file()
    meta = project.load_metadata()
    assert Path(meta.vod.path).exists()


def test_missing_vod_file_raises_error(tmp_path: Path, sample_config: AppConfig) -> None:
    non_existent = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        Project.create(vod_path=non_existent, config=sample_config)
