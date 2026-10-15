"""Unit tests for LLM Judge client, context builder, and anti-hallucination validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from narezka.asr.base import Segment
from narezka.config import LLMProfileConfig
from narezka.judge.context_builder import build_candidate_prompt_context, format_seconds_time
from narezka.judge.validation import validate_judge_verdict
from narezka.llm.client import BudgetExceededError, LLMClient
from narezka.llm.schemas import JudgeVerdict
from narezka.models import Candidate, Event


class SimpleEchoSchema(BaseModel):
    message: str
    number: int


def test_format_seconds_time():
    assert format_seconds_time(0.0) == "00:00:00.0"
    assert format_seconds_time(65.5) == "00:01:05.5"
    assert format_seconds_time(3661.0) == "01:01:01.0"


def test_llm_client_generation_and_caching(tmp_path: Path):
    profile = LLMProfileConfig(
        backend="openai",
        base_url="https://api.mock.test/v1",
        model="mock-gemini-2.5-flash",
        api_key_env="TEST_API_KEY",
    )
    cache_dir = tmp_path / "llm_cache"
    client = LLMClient(profile_config=profile, cache_dir=cache_dir, max_cost_usd=1.0)

    # Mock OpenAI chat completion response
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"message": "привет", "number": 42})
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 500
    mock_usage.completion_tokens = 100

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    messages = [{"role": "user", "content": "Тестовый запрос"}]

    assert client.client is not None
    with patch.object(client.client.chat.completions, "create", return_value=mock_resp) as mock_create:
        # First call: cache miss, calls client
        res1, meta1 = client.generate_json(messages, SimpleEchoSchema)
        assert res1.message == "привет"
        assert res1.number == 42
        assert meta1["cached"] is False
        assert meta1["cost_usd"] > 0
        assert mock_create.call_count == 1

        # Second call with same parameters: cache hit!
        res2, meta2 = client.generate_json(messages, SimpleEchoSchema)
        assert res2.message == "привет"
        assert res2.number == 42
        assert meta2["cached"] is True
        assert meta2["cost_usd"] == 0.0
        # Call count should still be 1 because it loaded from cache
        assert mock_create.call_count == 1


def test_llm_client_anthropic_claude(tmp_path: Path):
    profile = LLMProfileConfig(
        backend="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-3-5-haiku-latest",
        api_key_env="TEST_ANTHROPIC_KEY",
    )
    cache_dir = tmp_path / "claude_cache"
    client = LLMClient(profile_config=profile, cache_dir=cache_dir, max_cost_usd=1.0)
    assert client.anthropic_client is not None

    mock_block = MagicMock()
    mock_block.text = '```json\n{"message": "claude", "number": 77}\n```'

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage = MagicMock(input_tokens=200, output_tokens=50)

    messages = [{"role": "user", "content": "Привет, Claude"}]

    with patch.object(client.anthropic_client.messages, "create", return_value=mock_resp) as mock_create:
        res, meta = client.generate_json(messages, SimpleEchoSchema)
        assert res.message == "claude"
        assert res.number == 77
        assert meta["cached"] is False
        assert meta["cost_usd"] > 0
        assert mock_create.call_count == 1


def test_llm_client_budget_exceeded(tmp_path: Path):
    profile = LLMProfileConfig(
        base_url="https://api.mock.test/v1",
        model="mock-model",
    )
    client = LLMClient(profile_config=profile, cache_dir=tmp_path, max_cost_usd=0.00001)

    # Artificially consume budget
    client.total_cost_usd = 0.00002

    messages = [{"role": "user", "content": "Запрос"}]
    with pytest.raises(BudgetExceededError, match="budget limit reached"):
        client.generate_json(messages, SimpleEchoSchema)


def test_llm_client_retry_on_invalid_json(tmp_path: Path):
    profile = LLMProfileConfig(
        base_url="https://api.mock.test/v1",
        model="mock-model",
    )
    client = LLMClient(profile_config=profile, cache_dir=tmp_path, max_cost_usd=1.0)

    # First attempt: invalid JSON string. Second attempt: valid schema JSON.
    bad_choice = MagicMock()
    bad_choice.message.content = "Not JSON at all!"
    bad_resp = MagicMock()
    bad_resp.choices = [bad_choice]
    bad_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    good_choice = MagicMock()
    good_choice.message.content = json.dumps({"message": "recovered", "number": 100})
    good_resp = MagicMock()
    good_resp.choices = [good_choice]
    good_resp.usage = MagicMock(prompt_tokens=20, completion_tokens=10)

    messages = [{"role": "user", "content": "Запрос"}]
    assert client.client is not None
    with patch.object(client.client.chat.completions, "create", side_effect=[bad_resp, good_resp]) as mock_create:
        res, meta = client.generate_json(messages, SimpleEchoSchema, max_retries=2)
        assert res.message == "recovered"
        assert res.number == 100
        assert mock_create.call_count == 2


def test_validate_judge_verdict_hallucinated_quote():
    cand = Candidate(
        id="c_0001",
        start=10.0,
        end=30.0,
        peak=20.0,
        signal_score=0.9,
    )
    segments = [
        Segment(id=1, start=11.0, end=15.0, text="Убей его быстрее на базе!", speaker="streamer"),
        Segment(id=2, start=16.0, end=20.0, text="Хорош, минус один!", speaker="streamer"),
    ]
    events: list[Event] = []

    # Verdict contains hallucinated quote not present in segments
    verdict = JudgeVerdict(
        candidate_id="c_0001",
        verdict="keep",
        category="funny_reaction",
        title_ru="Смешной момент",
        reason_ru="Стример пошутил про банан",
        start_seg=1,
        end_seg=2,
        score=8,
        confidence=0.8,
        evidence_quotes=["Я обожаю бананы и апельсины в этой игре"],
    )

    sanitized, flags = validate_judge_verdict(verdict, cand, segments, events)
    assert "ungrounded" in flags
    # Hard reject on hallucinated quote
    assert sanitized.verdict == "reject"
    assert sanitized.score == 0.0
    assert sanitized.confidence == 0.0


def test_validate_judge_verdict_grounded_quote():
    cand = Candidate(
        id="c_0001",
        start=10.0,
        end=30.0,
        peak=20.0,
        signal_score=0.9,
    )
    segments = [
        Segment(id=1, start=11.0, end=15.0, text="Убей его быстрее на базе!", speaker="streamer"),
        Segment(id=2, start=16.0, end=20.0, text="Хорош минус один!", speaker="streamer"),
    ]
    events: list[Event] = []

    # Verdict contains verbatim quote
    verdict = JudgeVerdict(
        candidate_id="c_0001",
        verdict="keep",
        category="funny_reaction",
        title_ru="«Хорош минус один!»",
        reason_ru="Стример хвалит тиммейта",
        start_seg=1,
        end_seg=2,
        score=8,
        confidence=0.8,
        evidence_quotes=["Хорош минус один"],
    )

    sanitized, flags = validate_judge_verdict(verdict, cand, segments, events)
    assert "ungrounded" not in flags
    assert sanitized.confidence == pytest.approx(0.80)


def test_validate_judge_verdict_multikill_downgrade():
    cand = Candidate(
        id="c_0001",
        start=10.0,
        end=30.0,
        peak=20.0,
        signal_score=0.9,
    )
    segments = [
        Segment(id=1, start=11.0, end=20.0, text="Один готов, второй готов", speaker="streamer"),
    ]
    # No multikill event present in events list
    events = [
        Event(id="e1", source="audio", type="loud_z", start=15.0, end=18.0, score=0.8),
    ]

    verdict = JudgeVerdict(
        candidate_id="c_0001",
        verdict="keep",
        category="multikill",
        title_ru="Мультикилл на линии",
        reason_ru="Сделал триплкилл",
        start_seg=1,
        end_seg=1,
        score=9,
        confidence=0.9,
        evidence_quotes=["Один готов, второй готов"],
    )

    sanitized, flags = validate_judge_verdict(verdict, cand, segments, events)
    assert "multikill_downgraded" in flags
    assert sanitized.category == "epic_play"


def test_validate_judge_verdict_swapped_segments():
    cand = Candidate(
        id="c_0001",
        start=10.0,
        end=30.0,
        peak=20.0,
        signal_score=0.9,
    )
    segments = [
        Segment(id=10, start=12.0, end=15.0, text="Начало фразы", speaker="streamer"),
        Segment(id=20, start=21.0, end=25.0, text="Конец фразы", speaker="streamer"),
    ]
    events: list[Event] = []

    # Inverted segments: start_seg=20, end_seg=10
    verdict = JudgeVerdict(
        candidate_id="c_0001",
        verdict="keep",
        category="fail",
        title_ru="Фейл",
        reason_ru="Ошибся с кнопкой",
        start_seg=20,
        end_seg=10,
        score=7,
        confidence=0.8,
        evidence_quotes=["Конец фразы"],
    )

    sanitized, flags = validate_judge_verdict(verdict, cand, segments, events)
    assert "segment_order_swapped" in flags
    assert sanitized.start_seg == 10
    assert sanitized.end_seg == 20


def test_build_candidate_prompt_context():
    cand = Candidate(
        id="c_0042",
        start=100.0,
        end=140.0,
        peak=120.0,
        signal_score=0.95,
        suspected_categories=["multikill", "funny_reaction"],
        features_summary={"loud_z": 3.4, "chat_hype": 0.8},
    )
    segments = [
        Segment(id=5, start=105.0, end=112.0, text="Они пушат мид!", speaker="streamer"),
        Segment(id=6, start=113.0, end=125.0, text="Минус три, я гений!", speaker="streamer"),
    ]
    events = [
        Event(id="e1", source="audio", type="scream_p", start=118.0, end=122.0, score=0.9),
        Event(id="e2", source="chat", type="chat_hype", start=120.0, end=125.0, score=0.85),
    ]

    prompt_ctx = build_candidate_prompt_context(cand, segments, events, game_name="Deadlock")
    assert "КАНДИДАТ c_0042" in prompt_ctx
    assert "ИГРА: Deadlock" in prompt_ctx
    assert "Они пушат мид!" in prompt_ctx
    assert "Минус три, я гений!" in prompt_ctx
    assert "[seg 0005" in prompt_ctx
    assert "CHAT_HYPE" in prompt_ctx


def test_validate_judge_verdict_anti_afk_and_tech_check():
    cand = Candidate(id="c_0099", start=10.0, end=40.0, peak=25.0, signal_score=0.7)

    # 1. Tech check rejection
    tech_segs = [
        Segment(id=1, start=12.0, end=18.0, text="Слышно меня в дискорде? Микрофон поправил", speaker="streamer"),
    ]
    verdict = JudgeVerdict(
        candidate_id="c_0099",
        verdict="keep",
        category="other",
        title_ru="Настройка микрофона",
        reason_ru="Стример говорит про микрофон",
        start_seg=1,
        end_seg=1,
        score=7,
        confidence=0.8,
        evidence_quotes=["Слышно меня в дискорде"],
    )
    sanitized, flags = validate_judge_verdict(verdict, cand, tech_segs, [])
    assert sanitized.verdict == "reject"
    assert "tech_check_rejected" in flags

    # 2. AFK multikill rejection (silent streamer during match killfeed)
    silent_segs = []  # No speech
    mk_verdict = JudgeVerdict(
        candidate_id="c_0099",
        verdict="keep",
        category="multikill",
        title_ru="Серия фрагов",
        reason_ru="Мультикилл в игре",
        start_seg=0,
        end_seg=0,
        score=8,
        confidence=0.9,
        evidence_quotes=[],
    )
    sanitized_mk, mk_flags = validate_judge_verdict(mk_verdict, cand, silent_segs, [])
    assert sanitized_mk.verdict == "reject"
    assert "afk_multikill_rejected" in mk_flags

