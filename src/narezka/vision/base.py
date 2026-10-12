"""Base protocol and classes for game-specific computer vision plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from narezka.models import Event

if TYPE_CHECKING:
    from narezka.vision.sampler import FrameSampler


class BaseGameVisionPlugin(ABC):
    """Abstract base class for game-specific vision analysis (killfeed, HUD, events)."""

    game_name: str = "generic"

    @abstractmethod
    def detect_events(
        self,
        sampler: FrameSampler,
        duration_s: float,
        config: Any = None,
    ) -> list[Event]:
        """Process video frames from sampler and return discrete highlight events (kills, multikills, etc.)."""
        raise NotImplementedError

    def get_feature_curves(
        self,
        events: list[Event],
        duration_s: float,
        resolution_s: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """Convert detected events into continuous 1D timeline feature tracks (e.g. 'kill')."""
        num_frames = int(np.ceil(duration_s / resolution_s)) if duration_s > 0 else 0
        if num_frames <= 0:
            return {"kill": np.array([], dtype=np.float32)}

        kill_track = np.zeros(num_frames, dtype=np.float32)

        for evt in events:
            if evt.type in ("kill", "multikill"):
                idx = int(round(evt.start / resolution_s))
                if 0 <= idx < num_frames:
                    # Multikills have higher intensity; single kills set 1.0
                    intensity = 1.0 if evt.type == "kill" else 1.5
                    kill_track[idx] = max(kill_track[idx], intensity)
                    # Gaussian-like spread to neighboring 2-3 frames (+-1.5s)
                    for offset, weight in [(-2, 0.4), (-1, 0.75), (1, 0.75), (2, 0.4)]:
                        n_idx = idx + offset
                        if 0 <= n_idx < num_frames:
                            kill_track[n_idx] = max(kill_track[n_idx], intensity * weight)

        return {"kill": kill_track}
