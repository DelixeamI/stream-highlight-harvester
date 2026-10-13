"""Unit tests for pipeline stages S00 to S04."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from narezka.asr.base import Segment, Word
from narezka.pipeline.dag import PipelineDAG
from narezka.pipeline.stages.s00_probe import S00ProbeStage
from narezka.pipeline.stages.s01_audio_extract import S01AudioExtractStage
from narezka.pipeline.stages.s02_voice_isolation import S02VoiceIsolationStage
from narezka.pipeline.stages.s03_vad_lid import S03VadLidStage
from narezka.pipeline.stages.s04_asr import S04AsrStage
from narezka.project import Project


def test_stages_s00_to_s04_mocked_pipeline(test_project: Project) -> None:
    dag = PipelineDAG()
    dag.register(S00ProbeStage())
    dag.register(S01AudioExtractStage())
    dag.register(S02VoiceIsolationStage())
    dag.register(S03VadLidStage())
    dag.register(S04AsrStage())

    # Mock separation backend to avoid downloading weights during unit tests
    mock_sep_backend = MagicMock()

    def fake_separate(input_audio: Path, output_voice: Path, output_accompaniment: Path, **kwargs: object) -> None:
        import shutil
        shutil.copy2(input_audio, output_voice)
        shutil.copy2(input_audio, output_accompaniment)

    mock_sep_backend.separate.side_effect = fake_separate

    # Mock VAD
    mock_vad_backend = MagicMock()
    mock_vad_backend.detect_speech.return_value = [(0.0, 1.5)]

    # Mock LID
    mock_lid_backend = MagicMock()
    mock_lid_backend.identify_language.return_value = [
        {"start": 0.0, "end": 1.5, "lang": "ru", "confidence": 0.99}
    ]

    # Mock ASR backend
    mock_asr_backend = MagicMock()
    mock_asr_backend.transcribe.return_value = [
        Segment(
            id=0,
            start=0.0,
            end=1.5,
            text="мы победили",
            speaker="streamer",
            lang="ru",
            words=[
                Word(text="мы", start=0.0, end=0.6, prob=0.99),
                Word(text="победили", start=0.6, end=1.5, prob=0.98),
            ],
            source_track="voice_sep",
        )
    ]

    with patch("narezka.pipeline.stages.s02_voice_isolation.get_separation_backend", return_value=mock_sep_backend):
        with patch("narezka.pipeline.stages.s03_vad_lid.get_vad_backend", return_value=mock_vad_backend):
            with patch("narezka.pipeline.stages.s03_vad_lid.get_lid_backend", return_value=mock_lid_backend):
                with patch("narezka.pipeline.stages.s04_asr.get_asr_backend", return_value=mock_asr_backend):
                    results = dag.run(test_project)

                    assert len(results) == 5
                    assert results[0].stage_name == "s00_probe"
                    assert results[1].stage_name == "s01_audio"
                    assert results[2].stage_name == "s02_voice"
                    assert results[3].stage_name == "s03_vad_lid"
                    assert results[4].stage_name == "s04_asr"

                    # Check artifacts created
                    s04_dir = test_project.stages_dir / "s04_asr"
                    assert (s04_dir / "transcript.json").is_file()
                    assert (s04_dir / "transcript.srt").is_file()
                    assert (s04_dir / "transcript.txt").is_file()

                    # Verify transcript.json content
                    with open(s04_dir / "transcript.json", encoding="utf-8") as f:
                        data = json.load(f)
                    assert len(data) == 1
                    assert data[0]["text"] == "мы победили"
                    assert len(data[0]["words"]) == 2
