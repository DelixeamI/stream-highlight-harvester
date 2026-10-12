"""Computer Vision plugin for Counter-Strike 2 (CS2).

Detects streamer's personal kills and multikills via red rectangular killfeed borders
and cash reward notifications.
"""

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


class CS2VisionPlugin(BaseGameVisionPlugin):
    """CS2 vision plugin detecting personal kills and multikills via killfeed HUD analysis."""

    game_name: str = "cs2"

    def __init__(
        self,
        multikill_window_s: float = 12.0,
        kill_cooldown_s: float = 3.5,
    ) -> None:
        self.multikill_window_s = multikill_window_s
        self.kill_cooldown_s = kill_cooldown_s

    def detect_events(
        self,
        sampler: FrameSampler,
        duration_s: float,
        config: Any = None,
    ) -> list[Event]:
        """Scan frames for CS2 personal kill boxes and aggregate into kill/multikill events."""
        events: list[Event] = []
        confirmed_kills: list[float] = []
        active_kills: list[dict[str, float]] = []  # [{'last_seen': t, 'last_y': y}]

        multikill_win = (
            getattr(getattr(config, "vision", None), "multikill", None).window_s
            if hasattr(config, "vision")
            else self.multikill_window_s
        )

        logger.info("Starting CS2 vision killfeed analysis...")

        for curr_t, frame in sampler.sample_frames(start_s=0.0, end_s=duration_s):
            h, w, _ = frame.shape

            # CS2 Killfeed ROI: top-right corner, avoiding center player icons
            roi_y1, roi_y2 = int(h * 0.05), int(h * 0.28)
            roi_x1, roi_x2 = int(w * 0.65), int(w * 0.995)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # Precise red border mask: high red channel dominance
            b = roi[:, :, 0].astype(int)
            g = roi[:, :, 1].astype(int)
            r = roi[:, :, 2].astype(int)
            red_mask = (r >= 135) & ((r - g) >= 65) & ((r - b) >= 75)
            mask_u8 = (red_mask * 255).astype(np.uint8)

            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_boxes: list[tuple[float, int, int]] = []
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                # CS2 personal kill box dimensions: wide rectangle around ~220-750px wide, 18-65px high
                if 220 <= bw <= 750 and 18 <= bh <= 65 and (bw / max(1, bh)) >= 3.8:
                    y_c = by + bh / 2.0
                    valid_boxes.append((y_c, bw, bh))

            # Deduplicate multiple contours on the same frame that share similar y
            deduped_frame_boxes: list[tuple[float, int, int]] = []
            for y_c, bw, bh in sorted(valid_boxes, key=lambda b: b[0]):
                if not any(abs(y_c - dy) < 25.0 for dy, _, _ in deduped_frame_boxes):
                    deduped_frame_boxes.append((y_c, bw, bh))

            # Match against actively tracked killfeed notifications
            for y_c, bw, bh in deduped_frame_boxes:
                matched_kill = None
                for k in active_kills:
                    if abs(y_c - k["last_y"]) <= 45.0 and (curr_t - k["last_seen"]) <= 1.5:
                        matched_kill = k
                        break

                if matched_kill is not None:
                    matched_kill["last_seen"] = curr_t
                    matched_kill["last_y"] = y_c
                else:
                    # New distinct kill event (cooldown >= 0.5s)
                    if not confirmed_kills or (curr_t - confirmed_kills[-1]) >= 0.5:
                        confirmed_kills.append(curr_t)
                        active_kills.append({"last_seen": curr_t, "last_y": y_c})
                        m, s = int(curr_t // 60), int(curr_t % 60)
                        logger.info("🎯 [CS2 Vision] Личный фраг стримера на %02d:%02d (%.1f с)", m, s, curr_t)
                        evt_id = f"evt_vis_kill_{int(curr_t * 10):06d}"
                        events.append(
                            Event(
                                id=evt_id,
                                source="vision_killfeed",
                                type="kill",
                                start=round(curr_t, 2),
                                end=round(curr_t + 1.0, 2),
                                score=1.0,
                                payload={
                                    "game": "cs2",
                                    "method": "red_killfeed_contour",
                                    "box_w": bw,
                                    "box_h": bh,
                                },
                            )
                        )

            # Purge active tracked kills that disappeared from screen (> 2.0s without sighting)
            active_kills = [k for k in active_kills if (curr_t - k["last_seen"]) <= 2.0]

            # Periodic progress log every 600s
            if int(curr_t) % 600 == 0 and int(curr_t) > 0 and (curr_t - int(curr_t)) < 0.5:
                dur_m, dur_s = int(duration_s // 60), int(duration_s % 60)
                m, s = int(curr_t // 60), int(curr_t % 60)
                logger.info(
                    "👁️ Прогресс анализа видео: %02d:%02d / %02d:%02d (найдено фрагов: %d)",
                    m,
                    s,
                    dur_m,
                    dur_s,
                    len(confirmed_kills),
                )

        # Aggregate multikills
        multikill_events = self._aggregate_multikills(confirmed_kills, multikill_win)
        events.extend(multikill_events)

        logger.info(
            "CS2 vision analysis complete: found %d personal kills, %d multikills",
            len(confirmed_kills),
            len(multikill_events),
        )
        return events

    def _aggregate_multikills(
        self,
        kills: list[float],
        window_s: float,
    ) -> list[Event]:
        """Cluster kills into multikill events (Double, Triple, Quad, Ace)."""
        multikill_events: list[Event] = []
        labels = {
            2: "Double Kill",
            3: "Triple Kill",
            4: "Quad Kill",
            5: "Ace / Team Wipe",
        }

        i = 0
        while i < len(kills):
            streak = [kills[i]]
            j = i + 1
            # In CS2 competitive, max team wipe is 5 kills and max round action is ~22s
            while (
                j < len(kills)
                and kills[j] - streak[-1] <= window_s
                and len(streak) < 5
                and (kills[j] - streak[0]) <= 22.0
            ):
                streak.append(kills[j])
                j += 1

            count = len(streak)
            if count >= 2:
                label = labels.get(count, f"{count}-Kill Streak")
                score = min(1.0, 0.75 + 0.05 * count)
                evt_id = f"evt_vis_mk_{int(streak[0] * 10):06d}"
                multikill_events.append(
                    Event(
                        id=evt_id,
                        source="vision_multikill",
                        type="multikill",
                        start=round(streak[0], 2),
                        end=round(streak[-1], 2),
                        score=score,
                        payload={
                            "game": "cs2",
                            "count": count,
                            "label": label,
                            "kills": [round(t, 2) for t in streak],
                            "duration_s": round(streak[-1] - streak[0], 2),
                        },
                    )
                )
            i = j

        return multikill_events
