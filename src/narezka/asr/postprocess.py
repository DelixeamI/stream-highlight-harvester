"""Transcription postprocessing: hallucination filtering, glossary substitution, and text normalization."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from narezka.asr.base import Segment, Word
from narezka.config import HallucinationFilterConfig

logger = logging.getLogger(__name__)

PUNCTUATION_REGEX = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_russian_text(text: str) -> str:
    """Normalize Russian text for evaluation and fuzzy matching: lowercased, ё->е, no punctuation."""
    if not text:
        return ""
    # Lowercase
    t = text.lower()
    # ё -> е
    t = t.replace("ё", "е")
    # Strip punctuation
    t = PUNCTUATION_REGEX.sub(" ", t)
    # Collapse multiple whitespaces
    t = re.sub(r"\s+", " ", t).strip()
    return t


def check_repeated_ngrams(words: list[str], max_n: int = 4, max_repeats: int = 3) -> bool:
    """Detect if any n-gram of length 1..max_n repeats max_repeats or more times consecutively."""
    if len(words) < max_repeats:
        return False

    for n in range(1, max_n + 1):
        if len(words) < n * max_repeats:
            continue
        for i in range(len(words) - n * max_repeats + 1):
            pattern = words[i : i + n]
            repeats = 1
            idx = i + n
            while idx + n <= len(words) and words[idx : idx + n] == pattern:
                repeats += 1
                idx += n
                if repeats >= max_repeats:
                    return True
    return False


class HallucinationFilter:
    """Filters typical Whisper hallucinations on silence, music, or repetitive loops."""

    def __init__(self, config: HallucinationFilterConfig, blacklist_path: Path | None = None) -> None:
        self.config = config
        self.blacklist: list[str] = []

        path = blacklist_path or Path(config.blacklist_file)
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                self.blacklist = [
                    normalize_russian_text(line.strip())
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

    def is_hallucination(self, segment: Segment) -> tuple[bool, str | None]:
        # 1. Check metric thresholds if available
        if segment.no_speech_prob is not None and segment.no_speech_prob > self.config.no_speech_prob:
            return True, f"no_speech_prob ({segment.no_speech_prob:.2f} > {self.config.no_speech_prob})"

        if segment.avg_logprob is not None and segment.avg_logprob < self.config.avg_logprob:
            return True, f"avg_logprob ({segment.avg_logprob:.2f} < {self.config.avg_logprob})"

        if segment.compression_ratio is not None and segment.compression_ratio > self.config.compression_ratio:
            return True, f"compression_ratio ({segment.compression_ratio:.2f} > {self.config.compression_ratio})"

        norm_text = normalize_russian_text(segment.text)
        if not norm_text:
            return True, "empty_text"

        # 2. Check blacklist with fuzzy partial matching
        for phrase in self.blacklist:
            if not phrase:
                continue
            ratio = fuzz.partial_ratio(phrase, norm_text)
            if ratio >= 85:
                return True, f"blacklist_match ('{phrase}', ratio={ratio:.1f})"

        # 3. Check for repetitive n-grams (e.g. loops like "да да да да...")
        tokens = norm_text.split()
        if check_repeated_ngrams(tokens, max_n=3, max_repeats=3):
            return True, "repeated_ngrams"

        return False, None

    def filter_segments(self, segments: list[Segment]) -> list[Segment]:
        clean: list[Segment] = []
        for s in segments:
            is_hallu, reason = self.is_hallucination(s)
            if is_hallu:
                logger.debug("Отброшен галлюцинаторный сегмент [%.1f - %.1f]: '%s' (причина: %s)", s.start, s.end, s.text, reason)
            else:
                clean.append(s)
        return clean


class GlossaryCorrector:
    """Fuzzy dictionary substitution for gaming slang and hero names without distorting other words."""

    def __init__(self, glossary_file: Path | str | None = None, similarity_threshold: float = 85.0) -> None:
        self.threshold = similarity_threshold
        self.replacements: dict[str, str] = {}  # variant_normalized -> canonical

        if glossary_file:
            path = Path(glossary_file)
            if path.is_file():
                self.load_glossary(path)

    def load_glossary(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return

        for section in ("heroes", "terms", "items"):
            entries = data.get(section, {})
            if isinstance(entries, dict):
                for canonical, variants in entries.items():
                    norm_canon = normalize_russian_text(canonical)
                    self.replacements[norm_canon] = canonical
                    if isinstance(variants, list):
                        for var in variants:
                            norm_var = normalize_russian_text(str(var))
                            if norm_var:
                                self.replacements[norm_var] = canonical

    def correct_word(self, word: str) -> str:
        norm = normalize_russian_text(word)
        if not norm or len(norm) < 3:
            return word

        if norm in self.replacements:
            return self.replacements[norm]

        best_match = None
        best_score = 0.0

        for candidate, canonical in self.replacements.items():
            score = fuzz.ratio(norm, candidate)
            if score >= self.threshold and score > best_score:
                best_score = score
                best_match = canonical

        return best_match if best_match else word

    def apply(self, segments: list[Segment]) -> list[Segment]:
        if not self.replacements:
            return segments

        for seg in segments:
            new_words: list[Word] = []
            for w in seg.words:
                corrected_text = self.correct_word(w.text)
                new_words.append(Word(
                    text=corrected_text,
                    start=w.start,
                    end=w.end,
                    prob=w.prob,
                ))
            seg.words = new_words
            seg.text = " ".join(w.text for w in new_words) if new_words else seg.text

        return segments
