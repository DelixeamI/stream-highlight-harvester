"""Logging setup with rich console formatting, file handler, and API key masking."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()

# Masking patterns for sensitive keys
SENSITIVE_PATTERNS = [
    re.compile(r"(AIza[0-9A-Za-z-_]{35})"),
    re.compile(r"(sk-[a-zA-Z0-9_-]{20,})"),
    re.compile(r"(bearer\s+)([a-zA-Z0-9_\-\.]{15,})", re.IGNORECASE),
]


class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive tokens and API keys in log records."""

    def __init__(self, extra_secrets: list[str] | None = None) -> None:
        super().__init__()
        self.extra_secrets = [s for s in (extra_secrets or []) if s and len(s) > 4]
        # Also include known env secrets if present
        for env_var in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            val = os.getenv(env_var)
            if val and len(val) > 4:
                self.extra_secrets.append(val)

    def mask_text(self, text: str) -> str:
        for secret in self.extra_secrets:
            if secret in text:
                masked = secret[:3] + "..." + secret[-3:] if len(secret) > 8 else "***"
                text = text.replace(secret, masked)
        for pattern in SENSITIVE_PATTERNS:
            text = pattern.sub(r"***MASKED_KEY***", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.mask_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (self.mask_text(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(
                    (self.mask_text(a) if isinstance(a, str) else a)
                    for a in record.args
                )
        return True


def setup_logging(
    log_dir: Path | None = None,
    log_level: int = logging.INFO,
    debug: bool = False,
    extra_secrets: list[str] | None = None,
) -> logging.Logger:
    """Configure root logger with RichHandler for console and optional file handler."""
    effective_level = logging.DEBUG if debug else log_level
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    # Clear existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    filter_instance = SensitiveDataFilter(extra_secrets=extra_secrets)

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=debug,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(effective_level)
    rich_handler.addFilter(filter_instance)
    root_logger.addHandler(rich_handler)

    # Optional file handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(filter_instance)
        root_logger.addHandler(file_handler)

    # Silence overly verbose external loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger
