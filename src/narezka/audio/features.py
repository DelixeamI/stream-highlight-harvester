"""Audio feature extraction and timeline unification for Stage S6.

Extracts on timeline grid (0.5s step):
- loud_z: short-term loudness (400ms window), robust z-score over 10-min rolling window
- pitch_z: median F0 on voiced frames, robust z-score
- laugh_p: maximum probability across AudioSet laughter classes
- scream_p: maximum probability across AudioSet scream classes
- speech_rate: words per second from transcript in 3s window
- hype_lex: Russian reaction lexicon matches with weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from narezka.asr.base import Segment
from narezka.asr.postprocess import normalize_russian_text
from narezka.audio.loudness import extract_loudness_features
from narezka.audio.pitch import extract_pitch_features
from narezka.audio.tagger import extract_audio_tag_features
from narezka.config import FeaturesConfig, TimelineConfig
from narezka.models import Event

logger = logging.getLogger(__name__)


@dataclass
class AudioFeaturesResult:
    """Complete extracted audio feature signals on uniform time grid."""

    timestamps: np.ndarray  # (T,) time in seconds
    loud_raw: np.ndarray  # (T,) raw loudness LUFS/dB
    loud_z: np.ndarray  # (T,) robust z-score capped
    f0_raw: np.ndarray  # (T,) raw pitch Hz
    pitch_z: np.ndarray  # (T,) robust z-score on voiced frames
    laugh_p: np.ndarray  # (T,) laughter probability 0..1
    scream_p: np.ndarray  # (T,) scream probability 0..1
    speech_rate: np.ndarray  # (T,) words per second in 3s window
    hype_lex: np.ndarray  # (T,) lexicon excitation 0..1
    events: list[Event]  # discrete detected events


def compute_speech_rate(
    segments: list[Segment],
    duration_s: float,
    resolution_s: float = 0.5,
    window_s: float = 3.0,
) -> np.ndarray:
    """Compute words per second in a sliding window from transcript."""
    num_frames = int(np.ceil(duration_s / resolution_s))
    if num_frames <= 0:
        return np.array([], dtype=np.float32)

    # Collect all word midpoints
    word_midpoints: list[float] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                word_midpoints.append((w.start + w.end) / 2.0)
        else:
            # If word timestamps were not available, estimate uniform word distribution
            words = seg.text.strip().split()
            if words and seg.end > seg.start:
                step = (seg.end - seg.start) / len(words)
                for i in range(len(words)):
                    word_midpoints.append(seg.start + (i + 0.5) * step)

    if not word_midpoints:
        return np.zeros(num_frames, dtype=np.float32)

    midpoints_arr = np.sort(np.array(word_midpoints, dtype=np.float32))
    speech_rate = np.zeros(num_frames, dtype=np.float32)
    half_win = window_s / 2.0

    for i in range(num_frames):
        t = i * resolution_s
        t_min = t - half_win
        t_max = t + half_win

        idx_left = np.searchsorted(midpoints_arr, t_min, side="left")
        idx_right = np.searchsorted(midpoints_arr, t_max, side="right")
        count = idx_right - idx_left
        speech_rate[i] = float(count / window_s)

    return speech_rate


def compute_lexicon_features(
    segments: list[Segment],
    duration_s: float,
    lexicon_path: Path,
    resolution_s: float = 0.5,
) -> tuple[np.ndarray, list[Event]]:
    """Match Russian reaction lexicon and compute excitement curve + discrete events."""
    num_frames = int(np.ceil(duration_s / resolution_s))
    if num_frames <= 0:
        return np.array([], dtype=np.float32), []

    hype_lex = np.zeros(num_frames, dtype=np.float32)
    events: list[Event] = []

    if not lexicon_path.is_file():
        logger.warning("Lexicon file not found at %s", lexicon_path)
        return hype_lex, events

    with open(lexicon_path, encoding="utf-8") as f:
        lex_data = yaml.safe_load(f) or {}

    groups = lex_data.get("groups", {})

    for seg in segments:
        norm_seg_text = normalize_russian_text(seg.text)
        if not norm_seg_text:
            continue

        best_weight = 0.0
        best_match_phrase = ""
        best_group = ""

        for grp_name, grp_info in groups.items():
            weight = float(grp_info.get("weight", 0.5))
            phrases = grp_info.get("phrases", [])
            for phrase in phrases:
                norm_phrase = normalize_russian_text(phrase)
                if norm_phrase and norm_phrase in norm_seg_text:
                    if weight > best_weight:
                        best_weight = weight
                        best_match_phrase = phrase
                        best_group = grp_name

        if best_weight > 0.0:
            start_idx = max(0, int(np.floor(seg.start / resolution_s)))
            end_idx = min(num_frames, int(np.ceil(seg.end / resolution_s)))
            hype_lex[start_idx:end_idx] = np.maximum(hype_lex[start_idx:end_idx], best_weight)

            evt_id = f"evt_lex_{int(seg.start * 10):06d}"
            if best_group == "multikill":
                evt_type = "multikill"
            elif best_group == "clutch":
                evt_type = "clutch"
            elif best_group == "lucky_shot":
                evt_type = "lucky_shot"
            elif best_group == "gameplay_fail":
                evt_type = "gameplay_fail"
            elif best_group == "laughter":
                evt_type = "laughter"
            else:
                evt_type = "hype_phrase"

            events.append(
                Event(
                    id=evt_id,
                    source="lexicon",
                    type=evt_type,
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    score=round(best_weight, 3),
                    payload={
                        "phrase": best_match_phrase,
                        "group": best_group,
                        "text": seg.text,
                        "seg_id": seg.id,
                    },
                )
            )

    return hype_lex, events


def extract_all_audio_features(
    audio_path: Path,
    segments: list[Segment],
    timeline_cfg: TimelineConfig,
    features_cfg: FeaturesConfig,
    lexicon_path: Path,
    device: str = "cuda",
    tagger_backend: str = "auto",
) -> AudioFeaturesResult:
    """Extract all audio features and events for Stage S6."""
    import soundfile as sf

    info = sf.info(str(audio_path))
    duration_s = float(info.duration)
    res_s = timeline_cfg.resolution_s

    # 1. Loudness
    raw_loudness, loud_z, loud_events = extract_loudness_features(
        audio_path_or_data=audio_path,
        sample_rate=info.samplerate,
        resolution_s=res_s,
        rolling_window_s=600.0,
        cap=features_cfg.caps.get("loud_z", 5.0),
    )

    num_frames = len(loud_z)
    timestamps = np.arange(num_frames, dtype=np.float32) * res_s

    # 2. Pitch
    f0_raw, pitch_z = extract_pitch_features(
        audio_path_or_data=audio_path,
        sample_rate=info.samplerate,
        resolution_s=res_s,
        rolling_window_s=600.0,
        cap=features_cfg.caps.get("pitch_z", 4.0),
        device="cpu",  # CPU F0 is fast and avoids VRAM contention
    )
    if len(pitch_z) < num_frames:
        pad = np.zeros(num_frames - len(pitch_z), dtype=np.float32)
        f0_raw = np.concatenate([f0_raw, pad])
        pitch_z = np.concatenate([pitch_z, pad])
    else:
        f0_raw = f0_raw[:num_frames]
        pitch_z = pitch_z[:num_frames]

    # 3. Audio tagger (laughter & screams)
    laugh_p, scream_p, tag_events = extract_audio_tag_features(
        audio_path_or_data=audio_path,
        sample_rate=info.samplerate,
        resolution_s=res_s,
        backend=tagger_backend,
        device=device,
    )
    if len(laugh_p) < num_frames:
        pad = np.zeros(num_frames - len(laugh_p), dtype=np.float32)
        laugh_p = np.concatenate([laugh_p, pad])
        scream_p = np.concatenate([scream_p, pad])
    else:
        laugh_p = laugh_p[:num_frames]
        scream_p = scream_p[:num_frames]

    # 4. Speech rate
    speech_rate = compute_speech_rate(
        segments=segments,
        duration_s=duration_s,
        resolution_s=res_s,
        window_s=3.0,
    )
    if len(speech_rate) < num_frames:
        speech_rate = np.pad(speech_rate, (0, num_frames - len(speech_rate)))
    else:
        speech_rate = speech_rate[:num_frames]

    # 5. Lexicon
    hype_lex, lex_events = compute_lexicon_features(
        segments=segments,
        duration_s=duration_s,
        lexicon_path=lexicon_path,
        resolution_s=res_s,
    )
    if len(hype_lex) < num_frames:
        hype_lex = np.pad(hype_lex, (0, num_frames - len(hype_lex)))
    else:
        hype_lex = hype_lex[:num_frames]

    # Combine all events sorted by start time
    all_events = sorted(loud_events + tag_events + lex_events, key=lambda e: e.start)

    return AudioFeaturesResult(
        timestamps=timestamps,
        loud_raw=raw_loudness[:num_frames],
        loud_z=loud_z,
        f0_raw=f0_raw,
        pitch_z=pitch_z,
        laugh_p=laugh_p,
        scream_p=scream_p,
        speech_rate=speech_rate,
        hype_lex=hype_lex,
        events=all_events,
    )
