"""Configuration management using Pydantic v2 and YAML."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class AudioSeparationConfig(BaseModel):
    enabled: str | bool = "auto"
    backend: Literal["audio_separator", "demucs"] = "audio_separator"
    model: str = "mel_band_roformer_vocals"
    chunk_s: int = 600


class VadConfig(BaseModel):
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    pad_ms: int = 150


class LidConfig(BaseModel):
    enabled: bool = True
    backend: Literal["speechbrain", "whisper"] = "speechbrain"
    min_segment_s: float = 1.5
    non_ru_confidence: float = 0.8


class AudioConfig(BaseModel):
    tracks: str | dict[str, int] = "auto"
    separation: AudioSeparationConfig = Field(default_factory=AudioSeparationConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    lid: LidConfig = Field(default_factory=LidConfig)


class HallucinationFilterConfig(BaseModel):
    no_speech_prob: float = 0.6
    avg_logprob: float = -1.0
    compression_ratio: float = 2.4
    blacklist_file: str = "plugins/common/whisper_blacklist_ru.txt"


class AsrGigaamConfig(BaseModel):
    model: str = "v3_e2e"


class AsrFasterWhisperConfig(BaseModel):
    model: str = "large-v3"
    compute_type: str = "float16"
    beam_size: int = 5


class AsrConfig(BaseModel):
    backend: Literal["gigaam", "faster_whisper", "nemo"] = "faster_whisper"
    gigaam: AsrGigaamConfig = Field(default_factory=AsrGigaamConfig)
    faster_whisper: AsrFasterWhisperConfig = Field(default_factory=AsrFasterWhisperConfig)
    alignment: Literal["auto", "ctc_forced", "none"] = "auto"
    hallucination_filter: HallucinationFilterConfig = Field(default_factory=HallucinationFilterConfig)


class TimelineConfig(BaseModel):
    resolution_s: float = 0.5
    smoothing_sigma_s: float = 1.5


class FeaturesConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "loud_z": 0.20,
            "pitch_z": 0.05,
            "laugh_p": 0.25,
            "scream_p": 0.10,
            "hype_lex": 0.10,
            "chat_rate_z": 0.15,
            "chat_laugh": 0.10,
            "kill": 0.30,
            "semantic_flag": 0.20,
        }
    )
    caps: dict[str, float] = Field(
        default_factory=lambda: {
            "loud_z": 5.0,
            "pitch_z": 4.0,
            "chat_rate_z": 6.0,
        }
    )


class VisionSamplingConfig(BaseModel):
    mode: Literal["fps", "keyframes"] = "fps"
    fps: float = 2.0


class VisionMultikillConfig(BaseModel):
    window_s: float = 10.0


class VisionConfig(BaseModel):
    sampling: VisionSamplingConfig = Field(default_factory=VisionSamplingConfig)
    calibration: str = "plugins/deadlock/calibration/default.yaml"
    multikill: VisionMultikillConfig = Field(default_factory=VisionMultikillConfig)


class ChatDelayConfig(BaseModel):
    auto: bool = True
    min_delay_s: float = 3.0
    max_delay_s: float = 25.0
    default_s: float = 8.0


class ChatConfig(BaseModel):
    path: str | None = None
    delay: ChatDelayConfig = Field(default_factory=ChatDelayConfig)


class SemanticScanConfig(BaseModel):
    enabled: bool = True
    window_min: int = 8
    overlap_min: int = 1


class LLMProfileConfig(BaseModel):
    backend: Literal["openai", "anthropic"] = "openai"
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-3-5-haiku-latest"
    api_key_env: str | None = None


class JudgeConfig(BaseModel):
    profile: str = "claude"
    profiles: dict[str, LLMProfileConfig] = Field(
        default_factory=lambda: {
            "claude": LLMProfileConfig(
                backend="anthropic",
                base_url="https://api.anthropic.com",
                model="claude-3-5-haiku-latest",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            "openrouter": LLMProfileConfig(
                backend="openai",
                base_url="https://openrouter.ai/api/v1",
                model="anthropic/claude-3.5-haiku",
                api_key_env="OPENROUTER_API_KEY",
            ),
            "cloud": LLMProfileConfig(
                backend="openai",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="gemini-2.5-flash",
                api_key_env="GEMINI_API_KEY",
            ),
            "local": LLMProfileConfig(
                backend="openai",
                base_url="http://localhost:11434/v1",
                model="qwen2.5:14b",
                api_key_env=None,
            ),
        }
    )
    max_candidates: int = 120
    max_cost_usd: float = 2.0
    prompt_file: str = "prompts/judge_ru.md"
    examples_file: str = "prompts/judge_examples.yaml"


class OutputConfig(BaseModel):
    top_n: int = 20
    min_score: float = 0.50
    max_share_per_category: float = 0.5
    alpha_judge: float = 0.7
    clip_length_s: dict[str, list[int]] = Field(
        default_factory=lambda: {
            "multikill": [20, 55],
            "epic_play": [20, 60],
            "funny_joke": [15, 55],
            "funny_reaction": [15, 50],
            "fail": [18, 55],
            "other": [15, 60],
        }
    )
    pad_before_s: float = 3.0
    pad_after_s: float = 2.0


class RenderConfig(BaseModel):
    codec: Literal["h264_nvenc", "hevc_nvenc", "libx264"] = "h264_nvenc"
    preset: str = "p5"
    cq: int = 20
    audio_bitrate: str = "192k"
    max_parallel: int = 2


class AppConfig(BaseModel):
    game: str = "deadlock"
    device: Literal["cuda", "cpu"] = "cuda"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    semantic_scan: SemanticScanConfig = Field(default_factory=SemanticScanConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    def compute_hash(self) -> str:
        """Compute deterministic SHA256 hash of the configuration."""
        data = self.model_dump(mode="json")
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def load_config(
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> AppConfig:
    """Load configuration with inheritance: default.yaml -> optional config_path -> overrides -> .env."""
    load_dotenv()

    root = base_dir or Path.cwd()
    default_config_path = root / "config" / "default.yaml"

    config_dict: dict[str, Any] = {}
    if default_config_path.exists():
        with open(default_config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_dict = loaded

    if config_path is not None:
        user_path = Path(config_path)
        if not user_path.is_absolute():
            user_path = root / user_path
        if user_path.exists():
            with open(user_path, encoding="utf-8") as f:
                user_dict = yaml.safe_load(f)
                if isinstance(user_dict, dict):
                    config_dict = deep_merge(config_dict, user_dict)
        else:
            logger.warning("Config file not found: %s, continuing with defaults", user_path)

    if overrides:
        config_dict = deep_merge(config_dict, overrides)

    return AppConfig.model_validate(config_dict)
