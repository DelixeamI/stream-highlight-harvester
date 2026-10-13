"""Unit tests for ASR postprocessing: normalization, hallucination filter, and glossary."""

from __future__ import annotations

from pathlib import Path

from narezka.asr.base import Segment, Word
from narezka.asr.postprocess import (
    GlossaryCorrector,
    HallucinationFilter,
    check_repeated_ngrams,
    normalize_russian_text,
)
from narezka.config import HallucinationFilterConfig


def test_normalize_russian_text() -> None:
    raw = "Привет, Мир! Это тест: всё ещё работает?"
    norm = normalize_russian_text(raw)
    assert norm == "привет мир это тест все еще работает"


def test_check_repeated_ngrams() -> None:
    # 3 repetitions of 1-gram
    words_rep = ["да", "да", "да", "хорошо"]
    assert check_repeated_ngrams(words_rep, max_n=3, max_repeats=3)

    # 3 repetitions of 2-gram
    words_2gram = ["вот", "так", "вот", "так", "вот", "так", "конец"]
    assert check_repeated_ngrams(words_2gram, max_n=3, max_repeats=3)

    # Normal sentence
    normal_words = ["мы", "пошли", "на", "мид", "и", "взяли", "урну"]
    assert not check_repeated_ngrams(normal_words, max_n=3, max_repeats=3)


def test_hallucination_filter_thresholds() -> None:
    cfg = HallucinationFilterConfig(
        no_speech_prob=0.6,
        avg_logprob=-1.0,
        compression_ratio=2.4,
        blacklist_file="dummy_blacklist.txt",
    )
    filter_inst = HallucinationFilter(config=cfg)

    # High no_speech_prob
    seg1 = Segment(id=1, start=0.0, end=1.0, text="Привет", speaker="streamer", lang="ru", no_speech_prob=0.85)
    is_h1, reason1 = filter_inst.is_hallucination(seg1)
    assert is_h1
    assert "no_speech_prob" in str(reason1)

    # Low avg_logprob
    seg2 = Segment(id=2, start=0.0, end=1.0, text="Привет", speaker="streamer", lang="ru", avg_logprob=-1.8)
    is_h2, reason2 = filter_inst.is_hallucination(seg2)
    assert is_h2
    assert "avg_logprob" in str(reason2)

    # High compression ratio
    seg3 = Segment(id=3, start=0.0, end=1.0, text="Привет", speaker="streamer", lang="ru", compression_ratio=3.0)
    is_h3, reason3 = filter_inst.is_hallucination(seg3)
    assert is_h3
    assert "compression_ratio" in str(reason3)


def test_hallucination_filter_blacklist(tmp_path: Path) -> None:
    bl_file = tmp_path / "blacklist.txt"
    bl_file.write_text("Спасибо за просмотр!\nСубтитры сделал DimaTorzok\n", encoding="utf-8")

    cfg = HallucinationFilterConfig(
        no_speech_prob=0.9,
        avg_logprob=-2.0,
        compression_ratio=5.0,
        blacklist_file=str(bl_file),
    )
    filter_inst = HallucinationFilter(config=cfg, blacklist_path=bl_file)

    seg_bl = Segment(id=1, start=0.0, end=2.0, text="Спасибо за просмотр", speaker="streamer", lang="ru")
    is_h, reason = filter_inst.is_hallucination(seg_bl)
    assert is_h
    assert "blacklist_match" in str(reason)

    seg_ok = Segment(id=2, start=2.0, end=4.0, text="Я забрал урну", speaker="streamer", lang="ru")
    is_h_ok, _ = filter_inst.is_hallucination(seg_ok)
    assert not is_h_ok


def test_glossary_corrector(tmp_path: Path) -> None:
    glossary_file = tmp_path / "glossary.yaml"
    glossary_content = """
heroes:
  Haze: [Хейз, Хэйз]
  Lady Geist: [Леди Гейст, Гейст]
terms:
  Urn: [урна, урну]
"""
    glossary_file.write_text(glossary_content, encoding="utf-8")

    corrector = GlossaryCorrector(glossary_file=glossary_file, similarity_threshold=85.0)

    seg = Segment(
        id=1,
        start=0.0,
        end=2.0,
        text="хейз забрал урну",
        speaker="streamer",
        lang="ru",
        words=[
            Word(text="хейз", start=0.0, end=0.8),
            Word(text="забрал", start=0.8, end=1.4),
            Word(text="урну", start=1.4, end=2.0),
        ],
    )

    corrected = corrector.apply([seg])
    assert corrected[0].words[0].text == "Haze"
    assert corrected[0].words[2].text == "Urn"
    assert "Haze забрал Urn" == corrected[0].text
