"""Registry and factory for game vision plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from narezka.vision.base import BaseGameVisionPlugin
from narezka.vision.cs2 import CS2VisionPlugin
from narezka.vision.deadlock import DeadlockVisionPlugin
from narezka.vision.generic import GenericVisionPlugin

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_PLUGIN_REGISTRY: dict[str, type[BaseGameVisionPlugin]] = {
    "cs2": CS2VisionPlugin,
    "csgo": CS2VisionPlugin,
    "counterstrike": CS2VisionPlugin,
    "counter-strike": CS2VisionPlugin,
    "counter_strike": CS2VisionPlugin,
    "deadlock": DeadlockVisionPlugin,
    "generic": GenericVisionPlugin,
}


def get_vision_plugin(game: str | None = None) -> BaseGameVisionPlugin:
    """Instantiate and return the appropriate vision plugin for the target game."""
    if not game:
        return GenericVisionPlugin()

    norm_game = game.strip().lower()
    plugin_cls = _PLUGIN_REGISTRY.get(norm_game)
    if plugin_cls is not None:
        return plugin_cls()

    # Partial match check
    if "cs" in norm_game or "counter" in norm_game:
        return CS2VisionPlugin()
    if "deadlock" in norm_game:
        return DeadlockVisionPlugin()

    logger.info("No specific vision plugin for game '%s', falling back to GenericVisionPlugin", game)
    return GenericVisionPlugin()
