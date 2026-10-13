"""Unit tests for configuration models and loading logic."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from narezka.config import AppConfig, deep_merge, load_config


def test_deep_merge() -> None:
    base = {"a": 1, "nested": {"b": 2, "c": 3}}
    override = {"nested": {"c": 99, "d": 4}, "e": 5}
    merged = deep_merge(base, override)

    assert merged["a"] == 1
    assert merged["nested"]["b"] == 2
    assert merged["nested"]["c"] == 99
    assert merged["nested"]["d"] == 4
    assert merged["e"] == 5


def test_default_config_loading(tmp_path: Path) -> None:
    config = load_config(base_dir=tmp_path)
    assert isinstance(config, AppConfig)
    assert config.game == "deadlock"
    assert config.device in ("cuda", "cpu")
    assert config.audio.vad.threshold == 0.5
    assert config.output.top_n == 20
    assert config.judge.profile in ("claude", "cloud")


def test_config_overrides(tmp_path: Path) -> None:
    custom_yaml = tmp_path / "custom.yaml"
    custom_data = {
        "output": {"top_n": 50, "min_score": 0.8},
        "game": "custom_game",
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.dump(custom_data, f)

    config = load_config(config_path=custom_yaml, base_dir=tmp_path)
    assert config.output.top_n == 50
    assert config.output.min_score == 0.8
    assert config.game == "custom_game"
    # Defaults should remain intact
    assert config.output.alpha_judge == 0.7


def test_config_hash_determinism(tmp_path: Path) -> None:
    cfg1 = load_config(base_dir=tmp_path)
    cfg2 = load_config(base_dir=tmp_path)

    hash1 = cfg1.compute_hash()
    hash2 = cfg2.compute_hash()
    assert hash1 == hash2
    assert len(hash1) == 16

    # Modify property and check hash changes
    cfg1.output.top_n = 999
    assert cfg1.compute_hash() != hash2


def test_invalid_config_validation() -> None:
    with pytest.raises(ValidationError):
        # Invalid type for top_n
        AppConfig.model_validate({"output": {"top_n": "invalid_number"}})
