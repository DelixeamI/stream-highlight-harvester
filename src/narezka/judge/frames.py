"""Frame extraction and image serialization for Video-VLM Judge."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_candidate_keyframes(
    vod_path: Path | str,
    start_s: float,
    end_s: float,
    peak_s: float,
    num_frames: int = 8,
    target_height: int = 360,
    jpeg_quality: int = 75,
) -> list[tuple[float, str]]:
    """Extract evenly spaced keyframes across candidate window and encode to base64 JPEG.

    Args:
        vod_path: Path to video file.
        start_s: Candidate start time in seconds.
        end_s: Candidate end time in seconds.
        peak_s: Candidate peak timestamp.
        num_frames: Number of frames to extract (default 8).
        target_height: Output height for downscaling (default 360p, aspect ratio preserved).
        jpeg_quality: JPEG compression quality (default 75).

    Returns:
        List of (timestamp_s, base64_jpeg_str) tuples.
    """
    vod_path = Path(vod_path)
    if not vod_path.is_file():
        logger.warning("VOD file not found for frame extraction: %s", vod_path)
        return []

    # Compute timestamps with extra sample right around the peak
    raw_timestamps = np.linspace(start_s + 0.5, end_s - 0.5, num_frames).tolist()
    # Ensure peak is represented
    if not any(abs(t - peak_s) < 1.5 for t in raw_timestamps):
        raw_timestamps.append(peak_s)
    raw_timestamps = sorted(list(set(round(t, 2) for t in raw_timestamps)))
    if len(raw_timestamps) > num_frames:
        raw_timestamps = raw_timestamps[:num_frames]

    cap = cv2.VideoCapture(str(vod_path))
    if not cap.isOpened():
        logger.warning("Failed to open VOD with OpenCV: %s", vod_path)
        return []

    extracted: list[tuple[float, str]] = []
    try:
        for t_s in raw_timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s * 1000.0))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            if h > target_height:
                new_w = int(w * (target_height / float(h)))
                frame_resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame

            ok, buf = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
                extracted.append((t_s, b64_str))
    finally:
        cap.release()

    return extracted
