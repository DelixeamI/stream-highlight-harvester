"""Audio tagging using AudioSet models (PANNs CNN14) and acoustic heuristics.

Detects laughter (Laughter, Giggle, Belly laugh, Chuckle, Snicker) and screams
(Screaming, Shout, Yell) in 1.0s windows with 0.5s step as specified in TZ S6.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from narezka.models import Event
from narezka.utils.gpu import cleanup_vram

logger = logging.getLogger(__name__)

LAUGHTER_INDICES = [16, 18, 19, 20, 21]  # Laughter, Giggle, Snicker, Belly laugh, Chuckle/chortle
SCREAM_INDICES = [8, 11, 14]  # Shout, Yell, Screaming


class AudioTaggerBackend(Protocol):
    """Protocol for audio taggers."""

    def predict_probs(
        self,
        audio_32k: np.ndarray,
        resolution_s: float = 0.5,
        window_s: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict laugh_p and scream_p for each frame on the resolution grid."""
        ...


def ensure_panns_checkpoint(checkpoint_path: Path | None = None) -> Path:
    """Ensure CNN14 checkpoint and class labels are downloaded safely without wget."""
    panns_dir = Path.home() / "panns_data"
    panns_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = panns_dir / "class_labels_indices.csv"
    if not labels_csv.is_file() or labels_csv.stat().st_size < 100:
        logger.info("Downloading AudioSet class labels...")
        url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
        urllib.request.urlretrieve(url, str(labels_csv))

    ckpt = checkpoint_path or (panns_dir / "Cnn14_mAP=0.431.pth")
    if not ckpt.is_file() or ckpt.stat().st_size < 3e8:
        logger.info("Downloading PANNs CNN14 checkpoint (327 MB) from Zenodo...")
        url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
        tmp_ckpt = ckpt.with_suffix(".tmp")
        with urllib.request.urlopen(url) as response, open(tmp_ckpt, "wb") as out_file:
            total_size = int(response.info().get("Content-Length", 327428481))
            chunk_size = 1024 * 1024
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="PANNs CNN14") as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
        tmp_ckpt.replace(ckpt)
        logger.info("PANNs CNN14 checkpoint downloaded to %s", ckpt)

    return ckpt


class PANNsTagger:
    """PANNs CNN14 audio tagger running on PyTorch / CUDA."""

    def __init__(self, device: str = "cuda", checkpoint_path: Path | None = None) -> None:
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.ckpt_path = ensure_panns_checkpoint(checkpoint_path)
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        from panns_inference.models import Cnn14

        logger.info("Loading PANNs CNN14 model onto %s...", self.device)
        model = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        checkpoint = torch.load(str(self.ckpt_path), map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_probs(
        self,
        audio_32k: np.ndarray,
        resolution_s: float = 0.5,
        window_s: float = 1.0,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict (laugh_p, scream_p) arrays on resolution_s grid."""
        self._load_model()
        assert self.model is not None

        num_frames = int(np.ceil((len(audio_32k) / 32000.0) / resolution_s))
        if num_frames <= 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        window_samples = int(round(window_s * 32000))
        chunks = []

        for i in range(num_frames):
            center_sample = int(round((i * resolution_s + resolution_s / 2.0) * 32000))
            start = max(0, center_sample - window_samples // 2)
            end = start + window_samples

            if end <= len(audio_32k):
                chunk = audio_32k[start:end]
            else:
                chunk = np.zeros(window_samples, dtype=np.float32)
                avail = audio_32k[start:]
                chunk[: len(avail)] = avail

            chunks.append(chunk)

        laugh_p = np.zeros(num_frames, dtype=np.float32)
        scream_p = np.zeros(num_frames, dtype=np.float32)

        # Process in batches
        with torch.no_grad():
            for b_start in range(0, num_frames, batch_size):
                b_end = min(num_frames, b_start + batch_size)
                batch_arr = np.stack(chunks[b_start:b_end])
                batch_tensor = torch.from_numpy(batch_arr).to(self.device)

                out = self.model(batch_tensor, None)
                probs = torch.sigmoid(out["clipwise_output"]).cpu().numpy()

                laugh_batch = np.max(probs[:, LAUGHTER_INDICES], axis=1)
                scream_batch = np.max(probs[:, SCREAM_INDICES], axis=1)

                laugh_p[b_start:b_end] = laugh_batch
                scream_p[b_start:b_end] = scream_batch

        # Clean VRAM after inference
        self.model = None
        cleanup_vram()

        return laugh_p, scream_p


class HeuristicAudioTagger:
    """Fast acoustic heuristic audio tagger for tests and lightweight CPU fallback.

    Detects laughter modulation (intermittent burst envelope at 4-8 Hz)
    and screaming (high energy + high spectral centroid + high F0).
    """

    def predict_probs(
        self,
        audio_32k: np.ndarray,
        resolution_s: float = 0.5,
        window_s: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_frames = int(np.ceil((len(audio_32k) / 32000.0) / resolution_s))
        if num_frames <= 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        window_samples = int(round(window_s * 32000))
        laugh_p = np.zeros(num_frames, dtype=np.float32)
        scream_p = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            center_sample = int(round((i * resolution_s + resolution_s / 2.0) * 32000))
            start = max(0, center_sample - window_samples // 2)
            end = min(len(audio_32k), start + window_samples)
            chunk = audio_32k[start:end]

            if len(chunk) < 32000 * 0.2:
                continue

            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            if rms < 1e-3:
                continue

            # Spectral analysis via FFT
            fft_mag = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / 32000)

            total_mag = float(np.sum(fft_mag) + 1e-12)
            centroid = float(np.sum(freqs * fft_mag) / total_mag)

            # High energy scream indicator
            if rms > 0.08 and centroid > 1800:
                score = min(1.0, (rms / 0.25) * (centroid / 3000))
                scream_p[i] = float(score)

            # Laughter modulation indicator (amplitude variance of sub-blocks)
            sub_blocks = [
                np.sqrt(np.mean(chunk[j : j + 1600] ** 2) + 1e-12)
                for j in range(0, len(chunk) - 1600, 1600)
            ]
            if len(sub_blocks) >= 4:
                burstiness = float(np.std(sub_blocks) / (np.mean(sub_blocks) + 1e-6))
                if burstiness > 0.8 and 800 < centroid < 3500 and rms > 0.02:
                    score = min(1.0, burstiness * 0.45)
                    laugh_p[i] = float(score)

        return laugh_p, scream_p


def extract_audio_tag_features(
    audio_path_or_data: Path | np.ndarray,
    sample_rate: int = 16000,
    resolution_s: float = 0.5,
    backend: str = "auto",
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, list[Event]]:
    """Extract laugh_p, scream_p series and generate discrete laughter/scream events.

    Args:
        audio_path_or_data: File path or numpy audio array.
        sample_rate: Input sample rate.
        resolution_s: Timeline resolution step.
        backend: "auto", "panns", or "heuristic".
        device: "cuda" or "cpu".

    Returns:
        (laugh_p, scream_p, events)
    """
    if isinstance(audio_path_or_data, Path):
        data, sr = sf.read(str(audio_path_or_data), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    else:
        data = audio_path_or_data
        sr = sample_rate

    # Resample to 32000 for PANNs if needed
    if sr != 32000:
        import scipy.signal as signal

        num_target_samples = int(round(len(data) * 32000 / sr))
        res_arr = signal.resample(data, num_target_samples)
        if isinstance(res_arr, tuple):
            res_arr = res_arr[0]
        audio_32k = np.asarray(res_arr, dtype=np.float32)
    else:
        audio_32k = data.astype(np.float32)

    tagger: AudioTaggerBackend
    if backend == "panns" or (backend == "auto" and device == "cuda" and torch.cuda.is_available()):
        try:
            tagger = PANNsTagger(device=device)
            laugh_p, scream_p = tagger.predict_probs(audio_32k, resolution_s=resolution_s)
        except Exception as e:
            logger.warning("PANNs tagger initialization failed (%s), using heuristic tagger", e)
            tagger = HeuristicAudioTagger()
            laugh_p, scream_p = tagger.predict_probs(audio_32k, resolution_s=resolution_s)
    else:
        tagger = HeuristicAudioTagger()
        laugh_p, scream_p = tagger.predict_probs(audio_32k, resolution_s=resolution_s)

    # Detect discrete events
    events: list[Event] = []

    # 1. Laughter events: laugh_p >= 0.6 sustained for >= 1.0s
    in_laugh = False
    l_start = 0.0
    l_scores: list[float] = []
    for i, p in enumerate(laugh_p):
        t = i * resolution_s
        if p >= 0.6:
            if not in_laugh:
                in_laugh = True
                l_start = t
                l_scores = [float(p)]
            else:
                l_scores.append(float(p))
        else:
            if in_laugh:
                dur = t - l_start
                if dur >= 1.0:
                    max_p = max(l_scores)
                    evt_id = f"evt_laugh_{int(l_start * 10):06d}"
                    events.append(
                        Event(
                            id=evt_id,
                            source="audio_tagger",
                            type="laughter",
                            start=round(l_start, 2),
                            end=round(t, 2),
                            score=round(max_p, 3),
                            payload={"max_p": round(max_p, 3), "duration_s": round(dur, 2)},
                        )
                    )
                in_laugh = False

    # 2. Scream events: scream_p >= 0.6 sustained for >= 0.5s
    in_scream = False
    s_start = 0.0
    s_scores: list[float] = []
    for i, p in enumerate(scream_p):
        t = i * resolution_s
        if p >= 0.6:
            if not in_scream:
                in_scream = True
                s_start = t
                s_scores = [float(p)]
            else:
                s_scores.append(float(p))
        else:
            if in_scream:
                dur = t - s_start
                if dur >= 0.5:
                    max_p = max(s_scores)
                    evt_id = f"evt_scream_{int(s_start * 10):06d}"
                    events.append(
                        Event(
                            id=evt_id,
                            source="audio_tagger",
                            type="scream",
                            start=round(s_start, 2),
                            end=round(t, 2),
                            score=round(max_p, 3),
                            payload={"max_p": round(max_p, 3), "duration_s": round(dur, 2)},
                        )
                    )
                in_scream = False

    return laugh_p, scream_p, events
