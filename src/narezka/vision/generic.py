"""Generic fallback computer vision plugin for games without specialized HUD parsers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from narezka.models import Event
from narezka.vision.base import BaseGameVisionPlugin

if TYPE_CHECKING:
    from narezka.vision.sampler import FrameSampler

logger = logging.getLogger(__name__)


class GenericVisionPlugin(BaseGameVisionPlugin):
    """Fallback vision plugin analyzing killfeed ROI delta and combat motion dynamics."""

    game_name: str = "generic"

    def __init__(
        self,
        killfeed_diff_threshold: float = 24.0,
        action_diff_threshold: float = 38.0,
    ) -> None:
        self.killfeed_diff_threshold = killfeed_diff_threshold
        self.action_diff_threshold = action_diff_threshold

    def detect_events(
        self,
        sampler: FrameSampler,
        duration_s: float,
        config: Any = None,
    ) -> list[Event]:
        """Scan frames for killfeed changes and high-intensity combat action spikes."""
        events: list[Event] = []
        prev_kf_gray: np.ndarray | None = None
        prev_center_gray: np.ndarray | None = None
        last_kf_event_t: float = -999.0
        last_action_event_t: float = -999.0

        logger.info("Starting generic vision analysis (killfeed delta & combat motion)...")

        for curr_t, frame in sampler.sample_frames(start_s=0.0, end_s=duration_s):
            h, w, _ = frame.shape

            # Standard Killfeed ROI: top-right corner
            kf = frame[int(h * 0.04):int(h * 0.25), int(w * 0.70):int(w * 0.995)]
            kf_gray = cv2.cvtColor(kf, cv2.COLOR_BGR2GRAY)

            # Center Action ROI: middle screen combat area
            center = frame[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
            center_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)

            if prev_kf_gray is not None and prev_center_gray is not None:
                # 1. Killfeed difference
                kf_diff = float(np.mean(cv2.absdiff(kf_gray, prev_kf_gray)))
                if kf_diff >= self.killfeed_diff_threshold:
                    if curr_t - last_kf_event_t > 3.0:
                        last_kf_event_t = curr_t
                        evt_id = f"evt_gen_kf_{int(curr_t * 10):06d}"
                        events.append(
                            Event(
                                id=evt_id,
                                source="vision_killfeed",
                                type="kill",
                                start=round(curr_t, 2),
                                end=round(curr_t + 1.0, 2),
                                score=0.70,
                                payload={
                                    "game": "generic",
                                    "method": "killfeed_delta",
                                    "diff": round(kf_diff, 1),
                                },
                            )
                        )

                # 2. Combat motion spike
                action_diff = float(np.mean(cv2.absdiff(center_gray, prev_center_gray)))
                if action_diff >= self.action_diff_threshold:
                    if curr_t - last_action_event_t > 8.0:
                        last_action_event_t = curr_t
                        evt_id = f"evt_gen_act_{int(curr_t * 10):06d}"
                        events.append(
                            Event(
                                id=evt_id,
                                source="vision_motion",
                                type="epic_play",
                                start=round(curr_t, 2),
                                end=round(curr_t + 2.0, 2),
                                score=0.65,
                                payload={
                                    "game": "generic",
                                    "method": "center_motion_spike",
                                    "diff": round(action_diff, 1),
                                },
                            )
                        )

            prev_kf_gray = kf_gray
            prev_center_gray = center_gray

        logger.info("Generic vision analysis complete: found %d visual events", len(events))
        return events
