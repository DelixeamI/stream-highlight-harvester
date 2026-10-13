"""Unit tests for Computer Vision subsystem and Stage S07."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from narezka.models import Event
from narezka.vision.cs2 import CS2VisionPlugin
from narezka.vision.deadlock import DeadlockVisionPlugin
from narezka.vision.generic import GenericVisionPlugin
from narezka.vision.registry import get_vision_plugin
from narezka.vision.sampler import FrameSampler


def test_vision_registry():
    assert isinstance(get_vision_plugin("cs2"), CS2VisionPlugin)
    assert isinstance(get_vision_plugin("csgo"), CS2VisionPlugin)
    assert isinstance(get_vision_plugin("counter-strike"), CS2VisionPlugin)
    assert isinstance(get_vision_plugin("deadlock"), DeadlockVisionPlugin)
    assert isinstance(get_vision_plugin("unknown_game"), GenericVisionPlugin)
    assert isinstance(get_vision_plugin(None), GenericVisionPlugin)


def test_cs2_killfeed_red_box_detection():
    plugin = CS2VisionPlugin(multikill_window_s=12.0, kill_cooldown_s=3.0)

    # Frame 1: Blank frame (no kills)
    f_blank = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Frame 2: Contains red kill box in killfeed ROI (x: 0.60..0.995, y: 0.04..0.28)
    # 1920 * 0.70 = 1344, 1080 * 0.10 = 108
    f_kill = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # In BGR, bright red is [0, 0, 255]
    f_kill[100:135, 1300:1600] = [0, 0, 240]

    # Mock sampler
    sampler = MagicMock(spec=FrameSampler)
    sampler.sample_frames.return_value = [
        (10.0, f_blank),
        (15.0, f_kill),
        (15.5, f_kill),  # Same kill box persisting on screen
        (20.0, f_blank),
    ]

    events = plugin.detect_events(sampler, duration_s=30.0)
    kills = [e for e in events if e.type == "kill"]

    # Should detect exactly 1 kill at t=15.0s, not duplicated on t=15.5s
    assert len(kills) == 1
    assert kills[0].start == 15.0
    assert kills[0].source == "vision_killfeed"
    assert kills[0].payload["game"] == "cs2"


def test_cs2_multikill_aggregation():
    plugin = CS2VisionPlugin(multikill_window_s=12.0)
    # 3 kills within 12 seconds: 10.0, 14.0, 20.0
    kills = [10.0, 14.0, 20.0, 50.0]
    multikills = plugin._aggregate_multikills(kills, window_s=12.0)

    assert len(multikills) == 1
    mk = multikills[0]
    assert mk.type == "multikill"
    assert mk.payload["count"] == 3
    assert mk.payload["label"] == "Triple Kill"
    assert mk.start == 10.0
    assert mk.end == 20.0


def test_deadlock_hero_death_and_teamfight():
    plugin = DeadlockVisionPlugin(multikill_window_s=15.0, streamer_team="Amber")

    # Slot 6 is sapphire (enemy). Top bar: y: 0.005..0.06, x: 0.25..0.75
    # tb_w = 1920 * 0.50 = 960. slot_w = 960 / 12 = 80.
    # Slot 6 x is: 1920*0.25 + 6*80 = 480 + 480 = 960..1040.
    f_alive = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Living hero: high saturation (HSV S=150 -> BGR with high chroma)
    f_alive[5:60, 480:1440] = [200, 100, 50]

    f_dead = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Other slots stay alive, slot 6 turns grayscale (S=0)
    f_dead[5:60, 480:1440] = [200, 100, 50]
    f_dead[5:60, 960:1040] = [120, 120, 120]  # grayscale / dead

    sampler = MagicMock(spec=FrameSampler)
    sampler.sample_frames.return_value = [
        (10.0, f_alive),
        (10.5, f_dead),
    ]

    events = plugin.detect_events(sampler, duration_s=20.0)
    enemy_kills = [e for e in events if e.type == "kill" and e.payload.get("slot") == 6]

    assert len(enemy_kills) == 1
    assert enemy_kills[0].start == 10.5
    assert enemy_kills[0].payload["team"] == "Sapphire"


def test_generic_vision_plugin_killfeed_delta():
    plugin = GenericVisionPlugin(killfeed_diff_threshold=20.0)

    f1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Sudden brightness change in killfeed ROI
    f2[50:200, 1400:1800] = 200

    sampler = MagicMock(spec=FrameSampler)
    sampler.sample_frames.return_value = [
        (10.0, f1),
        (10.5, f2),
    ]

    events = plugin.detect_events(sampler, duration_s=20.0)
    assert len(events) >= 1
    assert events[0].source == "vision_killfeed"


def test_feature_curves_generation():
    plugin = CS2VisionPlugin()
    events = [
        Event(
            id="evt_1",
            source="vision_killfeed",
            type="kill",
            start=10.0,
            end=11.0,
            score=1.0,
        ),
        Event(
            id="evt_2",
            source="vision_multikill",
            type="multikill",
            start=25.0,
            end=28.0,
            score=1.0,
            payload={"count": 2},
        ),
    ]

    curves = plugin.get_feature_curves(events, duration_s=40.0, resolution_s=0.5)
    assert "kill" in curves
    kill_arr = curves["kill"]
    assert len(kill_arr) == 80  # 40.0 / 0.5 = 80
    # Peak at frame 20 (t=10s)
    assert kill_arr[20] >= 1.0
    # Higher peak at frame 50 (t=25s) for multikill
    assert kill_arr[50] >= 1.4
