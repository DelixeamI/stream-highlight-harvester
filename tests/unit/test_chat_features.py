"""Unit tests for Stage S08: Twitch chat features, delay estimation and alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from narezka.chat.base import ChatMessage
from narezka.chat.features import (
    estimate_chat_delay,
    extract_and_align_chat,
    extract_raw_chat_signals,
    shift_signal_backward,
)
from narezka.chat.twitch_downloader import TwitchDownloaderJSON
from narezka.config import ChatConfig, FeaturesConfig, TimelineConfig


def test_twitch_downloader_json_parser(tmp_path: Path):
    chat_file = tmp_path / "chat.json"
    content = {
        "streamer": {"name": "streamer_x"},
        "comments": [
            {
                "content_offset_seconds": 10.5,
                "commenter": {"display_name": "Viewer1"},
                "message": {"body": "KEKW nice shot"},
            },
            {
                "content_offset_seconds": 11.2,
                "commenter": {"display_name": "Viewer2"},
                "message": {"body": "Pog PogChamp"},
            },
            {
                "content_offset_seconds": 12.0,
                "commenter": {"display_name": "Viewer3"},
                "message": {"body": "ахаха ору"},
            },
        ],
    }
    chat_file.write_text(json.dumps(content), encoding="utf-8")

    parser = TwitchDownloaderJSON()
    msgs = parser.load_messages(chat_file)
    assert len(msgs) == 3
    assert msgs[0].timestamp_s == 10.5
    assert msgs[0].author == "Viewer1"
    assert msgs[0].text == "KEKW nice shot"


def test_extract_raw_chat_signals(tmp_path: Path):
    markers_file = tmp_path / "markers.yaml"
    markers_file.write_text(
        """
laugh_markers: ["kekw", "ору", "ахах"]
hype_markers: ["pog", "имба"]
""",
        encoding="utf-8",
    )

    msgs = [
        ChatMessage(timestamp_s=5.0, author="u1", text="KEKW"),
        ChatMessage(timestamp_s=5.2, author="u2", text="ору"),
        ChatMessage(timestamp_s=5.5, author="u3", text="Pog"),
        ChatMessage(timestamp_s=5.8, author="u4", text="обычный текст"),
    ]

    rate, rate_z, laugh, hype = extract_raw_chat_signals(
        messages=msgs,
        duration_s=10.0,
        markers_path=markers_file,
        resolution_s=0.5,
        window_s=2.0,
    )

    assert len(rate) == 20
    # Around t=5.0s (frame 10), count=4 in window -> rate = 2.0 msg/s
    assert rate[10] > 0
    # 2 out of 4 are laugh markers
    assert laugh[10] == 0.5
    # 1 out of 4 is hype
    assert hype[10] == 0.25


def test_estimate_chat_delay_cross_correlation():
    res_s = 0.5
    num_frames = 200  # 100 seconds

    # Stream excitation at t = 20s (frame 40)
    audio_excitation = np.zeros(num_frames, dtype=np.float32)
    audio_excitation[38:43] = 4.0

    # Chat responds 10 seconds later at t = 30s (frame 60)
    chat_rate_z = np.zeros(num_frames, dtype=np.float32)
    chat_rate_z[58:63] = 5.0

    delay = estimate_chat_delay(
        chat_rate_z=chat_rate_z,
        audio_excitation=audio_excitation,
        resolution_s=res_s,
        min_delay_s=3.0,
        max_delay_s=25.0,
        default_s=8.0,
    )

    # Delay should be discovered as approximately 10.0s
    assert abs(delay - 10.0) <= 1.0


def test_shift_signal_backward():
    res_s = 0.5
    sig = np.array([0, 0, 0, 10, 20, 30, 0, 0], dtype=np.float32)
    # Shift backward by 1.0s = 2 frames
    shifted = shift_signal_backward(sig, shift_seconds=1.0, resolution_s=res_s)
    # The value 10 at index 3 should move to index 1
    assert shifted[1] == 10.0
    assert shifted[2] == 20.0


def test_extract_and_align_chat_empty(tmp_path: Path):
    markers = tmp_path / "markers.yaml"
    markers.write_text("laugh_markers: []\nhype_markers: []", encoding="utf-8")

    res = extract_and_align_chat(
        messages=[],
        duration_s=10.0,
        audio_excitation=None,
        chat_cfg=ChatConfig(),
        features_cfg=FeaturesConfig(),
        timeline_cfg=TimelineConfig(),
        markers_path=markers,
    )

    assert len(res.timestamps) == 20
    assert np.all(res.chat_rate == 0)
    assert len(res.events) == 0
