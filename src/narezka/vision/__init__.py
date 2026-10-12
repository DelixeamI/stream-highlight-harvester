"""Computer vision package for esports game HUD and killfeed highlight extraction."""

from __future__ import annotations

from narezka.vision.base import BaseGameVisionPlugin
from narezka.vision.cs2 import CS2VisionPlugin
from narezka.vision.deadlock import DeadlockVisionPlugin
from narezka.vision.generic import GenericVisionPlugin
from narezka.vision.registry import get_vision_plugin
from narezka.vision.sampler import FrameSampler

__all__ = [
    "BaseGameVisionPlugin",
    "CS2VisionPlugin",
    "DeadlockVisionPlugin",
    "GenericVisionPlugin",
    "FrameSampler",
    "get_vision_plugin",
]
