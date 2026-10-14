"""Computer Vision plugin for Valve's Deadlock.

Detects hero eliminations, center kill confirmations, teamfights, and multikills
via top roster status and center HUD indicators.
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


class DeadlockVisionPlugin(BaseGameVisionPlugin):
    """Deadlock vision plugin detecting hero deaths, kill confirmations and multikills."""

    game_name: str = "deadlock"

    def __init__(
        self,
        multikill_window_s: float = 15.0,
        streamer_team: str = "Amber",  # "Amber" (slots 0-5) or "Sapphire" (slots 6-11)
    ) -> None:
        self.multikill_window_s = multikill_window_s
        self.streamer_team = streamer_team

    def detect_events(
        self,
        sampler: FrameSampler,
        duration_s: float,
        config: Any = None,
    ) -> list[Event]:
        """Scan frames for Deadlock hero deaths, center kill confirmations, and multikills."""
        events: list[Event] = []
        confirmed_kills: list[float] = []
        hero_deaths: list[tuple[float, int, str]] = []  # [(time, slot_idx, team)]
        last_center_kill_t: float = -999.0
        prev_slot_sats: list[float] | None = None

        multikill_win = (
            getattr(getattr(config, "vision", None), "multikill", None).window_s
            if hasattr(config, "vision")
            else self.multikill_window_s
        )

        logger.info("Starting Deadlock vision HUD analysis...")

        for curr_t, frame in sampler.sample_frames(start_s=0.0, end_s=duration_s):
            h, w, _ = frame.shape

            # 1. Top Bar Hero Roster (12 hero slots across x: 0.25..0.75, y: 0.005..0.06)
            tb = frame[int(h * 0.005):int(h * 0.06), int(w * 0.25):int(w * 0.75)]
            tb_hsv = cv2.cvtColor(tb, cv2.COLOR_BGR2HSV)
            tb_w = tb.shape[1]
            slot_w = tb_w / 12.0
            slot_sats: list[float] = []
            for i in range(12):
                x1 = int(i * slot_w)
                x2 = int((i + 1) * slot_w)
                slot = tb_hsv[:, x1:x2, 1]
                slot_sats.append(float(np.mean(slot)))

            # Check hero death transitions (living saturation > 65 to dead < 55)
            if prev_slot_sats is not None:
                dropped_slots = [
                    i for i in range(12)
                    if prev_slot_sats[i] > 65.0 and slot_sats[i] < 55.0
                ]
                # If 3 or more hero slots drop simultaneously in the exact same frame,
                # it is a scene transition (menu, scoreboard, death screen, map load), NOT kills!
                if len(dropped_slots) < 3:
                    for i in dropped_slots:
                        team = "Amber" if i < 6 else "Sapphire"
                        hero_deaths.append((curr_t, i, team))
                        # Is this an enemy hero death?
                        is_enemy = (team != self.streamer_team)
                        if is_enemy:
                            # If recently confirmed in center or standalone enemy death
                            is_recent_center = (curr_t - last_center_kill_t) <= 3.0
                            score = 1.0 if is_recent_center else 0.75
                            confirmed_kills.append(curr_t)
                            m, s = int(curr_t // 60), int(curr_t % 60)
                            logger.info("🎯 [Deadlock Vision] Фраг врага (%s, слот %d) на %02d:%02d (%.1f с)", team, i, m, s, curr_t)
                            evt_id = f"evt_dl_kill_{int(curr_t * 10):06d}"
                            events.append(
                                Event(
                                    id=evt_id,
                                    source="vision_hud",
                                    type="kill",
                                    start=round(curr_t, 2),
                                    end=round(curr_t + 1.0, 2),
                                    score=score,
                                    payload={
                                        "game": "deadlock",
                                        "slot": i,
                                        "team": team,
                                        "is_enemy": True,
                                        "confirmed_by_center": is_recent_center,
                                    },
                                )
                            )

            prev_slot_sats = slot_sats

            # 2. Top-Left Elimination Feed Banner (x: 0.0..0.22, y: 0.16..0.27)
            kf = frame[int(h * 0.16):int(h * 0.27), :int(w * 0.22)]
            kf_hsv = cv2.cvtColor(kf, cv2.COLOR_BGR2HSV)
            kf_red = cv2.inRange(kf_hsv, (0, 140, 120), (10, 255, 255)) | cv2.inRange(kf_hsv, (170, 140, 120), (180, 255, 255))
            kf_red_px = int(np.count_nonzero(kf_red))
            kf_green = cv2.inRange(kf_hsv, (40, 60, 140), (90, 255, 255))
            kf_green_px = int(np.count_nonzero(kf_green))

            # A hero kill banner in Deadlock has a red victim block (>1200 px) and green soul bounty (>250 px)
            if kf_red_px > 1200 and kf_green_px > 250:
                if curr_t - last_center_kill_t > 3.0:
                    last_center_kill_t = curr_t
                    if not confirmed_kills or (curr_t - confirmed_kills[-1]) > 2.5:
                        confirmed_kills.append(curr_t)
                        m, s = int(curr_t // 60), int(curr_t % 60)
                        logger.info("🎯 [Deadlock Vision] Киллфид: подтверждено устранение героя на %02d:%02d (%.1f с)", m, s, curr_t)
                        evt_id = f"evt_dl_kf_{int(curr_t * 10):06d}"
                        events.append(
                            Event(
                                id=evt_id,
                                source="vision_killfeed",
                                type="kill",
                                start=round(curr_t, 2),
                                end=round(curr_t + 1.0, 2),
                                score=0.85,
                                payload={
                                    "game": "deadlock",
                                    "method": "top_left_killfeed",
                                    "red_pixels": kf_red_px,
                                    "green_pixels": kf_green_px,
                                },
                            )
                        )

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

        # Aggregate multikills and teamfights
        multikills = self._aggregate_multikills(confirmed_kills, multikill_win)
        events.extend(multikills)

        teamfights = self._detect_teamfights(hero_deaths, window_s=20.0)
        events.extend(teamfights)

        logger.info(
            "Deadlock vision analysis complete: %d kills, %d multikills, %d teamfights",
            len(confirmed_kills),
            len(multikills),
            len(teamfights),
        )
        return events

    def _aggregate_multikills(
        self,
        kills: list[float],
        window_s: float,
    ) -> list[Event]:
        """Aggregate rapid kills into multikill events."""
        events: list[Event] = []
        labels = {
            2: "Double Kill",
            3: "Triple Kill",
            4: "Quadra Kill",
            5: "Team Wipe",
            6: "Full Team Wipe",
        }

        i = 0
        while i < len(kills):
            streak = [kills[i]]
            j = i + 1
            while j < len(kills) and kills[j] - streak[-1] <= window_s:
                streak.append(kills[j])
                j += 1

            count = len(streak)
            if count >= 2:
                label = labels.get(count, f"{count}-Hero Streak")
                score = min(1.0, 0.75 + 0.05 * count)
                evt_id = f"evt_dl_mk_{int(streak[0] * 10):06d}"
                events.append(
                    Event(
                        id=evt_id,
                        source="vision_multikill",
                        type="multikill",
                        start=round(streak[0], 2),
                        end=round(streak[-1], 2),
                        score=score,
                        payload={
                            "game": "deadlock",
                            "count": count,
                            "label": label,
                            "kills": [round(t, 2) for t in streak],
                            "duration_s": round(streak[-1] - streak[0], 2),
                        },
                    )
                )
            i = j

        return events

    def _detect_teamfights(
        self,
        deaths: list[tuple[float, int, str]],
        window_s: float = 20.0,
    ) -> list[Event]:
        """Detect massive teamfights where 3+ total heroes from both teams die in a short window."""
        events: list[Event] = []
        if len(deaths) < 3:
            return events

        i = 0
        while i < len(deaths):
            cluster = [deaths[i]]
            j = i + 1
            while j < len(deaths) and deaths[j][0] - deaths[i][0] <= window_s:
                cluster.append(deaths[j])
                j += 1

            if len(cluster) >= 3:
                t_start = cluster[0][0]
                t_end = cluster[-1][0]
                evt_id = f"evt_dl_tf_{int(t_start * 10):06d}"
                events.append(
                    Event(
                        id=evt_id,
                        source="vision_teamfight",
                        type="epic_play",
                        start=round(t_start, 2),
                        end=round(t_end, 2),
                        score=0.85,
                        payload={
                            "game": "deadlock",
                            "total_deaths": len(cluster),
                            "duration_s": round(t_end - t_start, 2),
                        },
                    )
                )
                i = j
            else:
                i += 1

        return events
