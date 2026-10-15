from pathlib import Path

import cv2
import numpy as np

from narezka.judge.frames import extract_candidate_keyframes


def test_extract_candidate_keyframes(tmp_path: Path):
    # Create a synthetic 5-second test video
    vod_file = tmp_path / "test_vod.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vod_file), fourcc, 30.0, (640, 360))
    for _ in range(150):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    frames = extract_candidate_keyframes(
        vod_path=vod_file,
        start_s=1.0,
        end_s=4.0,
        peak_s=2.5,
        num_frames=4,
        target_height=180,
    )
    assert len(frames) >= 3
    for ts, b64_img in frames:
        assert isinstance(ts, float)
        assert isinstance(b64_img, str)
        assert len(b64_img) > 50
