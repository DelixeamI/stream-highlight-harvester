"""Video frame sampling utilities using OpenCV VideoCapture."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSampler:
    """Samples video frames at a fixed FPS rate (default 2.0 FPS)."""

    def __init__(self, vod_path: Path | str, fps: float = 2.0) -> None:
        self.vod_path = Path(vod_path)
        self.target_fps = float(fps)
        self.step_s = 1.0 / self.target_fps if self.target_fps > 0 else 0.5
        self._cap: cv2.VideoCapture | None = None
        self.video_fps: float = 30.0
        self.total_frames: int = 0
        self.duration_s: float = 0.0

    def open(self) -> None:
        if not self.vod_path.exists():
            raise FileNotFoundError(f"VOD file not found: {self.vod_path}")
        self._cap = cv2.VideoCapture(str(self.vod_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.vod_path}")
        self.video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_s = self.total_frames / self.video_fps if self.video_fps > 0 else 0.0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def sample_frames(
        self,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> Generator[tuple[float, np.ndarray], None, None]:
        """Iterate over frames at target_fps rate, yielding (timestamp_s, frame_bgr)."""
        if self._cap is None:
            raise RuntimeError("Sampler is not open. Use 'with FrameSampler(...):'")

        max_t = self.duration_s if end_s is None else min(self.duration_s, end_s)
        curr_t = float(start_s)

        # Seek once at start if start_s > 0
        if start_s > 0.0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

        frame_interval = max(1, int(round(self.video_fps / self.target_fps))) if self.target_fps > 0 else 1
        actual_step_s = frame_interval / self.video_fps

        while curr_t <= max_t:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield curr_t, frame
            # Skip subsequent frames in interval using fast grab() without full decoding
            for _ in range(frame_interval - 1):
                if not self._cap.grab():
                    return
            curr_t += actual_step_s
