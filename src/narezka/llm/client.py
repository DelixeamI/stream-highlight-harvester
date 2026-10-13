"""OpenAI and Anthropic compatible LLM client with disk caching, token tracking and cost limits."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TypeVar

import anthropic
import openai
from pydantic import BaseModel, ValidationError

from narezka.config import LLMProfileConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default estimated pricing per 1M tokens (Gemini / Claude Haiku baseline)
PRICE_PER_M_INPUT_USD = 0.80
PRICE_PER_M_OUTPUT_USD = 4.00


class BudgetExceededError(RuntimeError):
    """Raised when cumulative LLM call cost exceeds configured budget limit."""
    pass


class LLMClient:
    """LLM client supporting OpenAI, OpenRouter and Anthropic APIs with caching and cost guardrails."""

    def __init__(
        self,
        profile_config: LLMProfileConfig,
        cache_dir: Path | None = None,
        max_cost_usd: float = 2.0,
    ) -> None:
        self.profile = profile_config
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_cost_usd = max_cost_usd
        self.total_cost_usd = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Retrieve API key
        api_key = "dummy_key_for_local"
        if profile_config.api_key_env:
            val = os.getenv(profile_config.api_key_env)
            if val:
                api_key = val
            else:
                logger.warning(
                    "Переменная окружения %s не установлена. Запросы к облачной LLM могут завершиться ошибкой.",
                    profile_config.api_key_env,
                )

        if profile_config.backend == "anthropic":
            base_url = None if "api.anthropic.com" in profile_config.base_url else profile_config.base_url
            self.anthropic_client: anthropic.Anthropic | None = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
            )
            self.client: openai.OpenAI | None = None
        else:
            self.anthropic_client = None
            self.client = openai.OpenAI(
                base_url=profile_config.base_url,
                api_key=api_key,
                timeout=45.0,
            )

    def _compute_cache_key(self, messages: list[dict[str, str]], temperature: float) -> str:
        raw = f"{self.profile.model}::{temperature}::{json.dumps(messages, sort_keys=True)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate_json(
        self,
        messages: list[dict[str, str]],
        schema_cls: type[T],
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> tuple[T, dict[str, Any]]:
        """Call LLM ensuring structured JSON output matching schema_cls.

        Returns:
            (validated_instance, metadata_dict)
        """
        if self.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"LLM budget limit reached: ${self.total_cost_usd:.3f} >= ${self.max_cost_usd:.2f}"
            )

        cache_key = self._compute_cache_key(messages, temperature)
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.is_file():
                try:
                    with open(cache_file, encoding="utf-8") as f:
                        cached_data = json.load(f)
                    instance = schema_cls.model_validate(cached_data["data"])
                    logger.debug("Cache hit for LLM query %s", cache_key[:8])
                    return instance, {"cached": True, "cost_usd": 0.0}
                except Exception as e:
                    logger.warning("Cache read failed for %s: %s, executing query", cache_key[:8], e)

        current_messages = list(messages)
        last_error = ""

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info("Retrying LLM generation (attempt %d/%d)...", attempt, max_retries)
                current_messages.append(
                    {
                        "role": "user",
                        "content": f"Предыдущий ответ не прошел валидацию схемы: {last_error}. Пожалуйста, исправь ошибки и верни строго корректный JSON.",
                    }
                )

            try:
                schema_doc = json.dumps(schema_cls.model_json_schema(), ensure_ascii=False)
                schema_instruction = (
                    f"\n\nВАЖНО: Ответ должен быть СТРОГО валидным JSON объектом по следующей схеме JSON Schema:\n{schema_doc}\n"
                    "Не добавляй никаких вводных слов, пояснений или markdown-тегов вне JSON."
                )

                if self.anthropic_client:
                    # Anthropic Claude API call
                    system_text = ""
                    claude_messages: list[dict[str, Any]] = []
                    for m in current_messages:
                        if m["role"] == "system":
                            system_text += m["content"] + "\n\n"
                        else:
                            claude_messages.append({"role": m["role"], "content": m["content"]})

                    system_prompt_full = (
                        (system_text.strip() + "\n\n" if system_text else "")
                        + schema_instruction.strip()
                    )

                    response = self.anthropic_client.messages.create(
                        model=self.profile.model,
                        system=system_prompt_full,
                        messages=claude_messages,  # type: ignore
                        temperature=temperature,
                        max_tokens=4096,
                    )
                    content = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            content += block.text

                    p_tokens = response.usage.input_tokens
                    c_tokens = response.usage.output_tokens

                    model_low = self.profile.model.lower()
                    if "haiku" in model_low:
                        call_cost = (p_tokens / 1_000_000.0) * 0.80 + (c_tokens / 1_000_000.0) * 4.00
                    elif "sonnet" in model_low:
                        call_cost = (p_tokens / 1_000_000.0) * 3.00 + (c_tokens / 1_000_000.0) * 15.00
                    else:
                        call_cost = (p_tokens / 1_000_000.0) * 1.00 + (c_tokens / 1_000_000.0) * 5.00
                else:
                    # OpenAI-compatible API call
                    assert self.client is not None
                    openai_messages: list[dict[str, Any]] = []
                    has_system = False
                    for m in current_messages:
                        if m["role"] == "system":
                            openai_messages.append({
                                "role": "system",
                                "content": m["content"] + schema_instruction,
                            })
                            has_system = True
                        else:
                            openai_messages.append(dict(m))
                    if not has_system:
                        openai_messages.insert(0, {
                            "role": "system",
                            "content": schema_instruction.strip(),
                        })

                    api_resp = self.client.chat.completions.create(
                        model=self.profile.model,
                        messages=openai_messages,  # type: ignore
                        temperature=temperature,
                        max_tokens=1200,
                        response_format={"type": "json_object"},
                    )
                    choice = api_resp.choices[0]
                    content = choice.message.content or "{}"

                    usage = api_resp.usage
                    p_tokens = usage.prompt_tokens if usage else 0
                    c_tokens = usage.completion_tokens if usage else 0
                    call_cost = (p_tokens / 1_000_000.0) * PRICE_PER_M_INPUT_USD + (
                        c_tokens / 1_000_000.0
                    ) * PRICE_PER_M_OUTPUT_USD

                self.total_prompt_tokens += p_tokens
                self.total_completion_tokens += c_tokens
                self.total_cost_usd += call_cost

                # Clean thinking tags (<think>...</think>) from reasoning models
                clean_content = content.strip()
                clean_content = re.sub(r"<think>.*?</think>", "", clean_content, flags=re.DOTALL).strip()

                # Clean markdown fences if LLM wrapped JSON in ```json ... ```
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                elif clean_content.startswith("```"):
                    clean_content = clean_content[3:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                clean_content = clean_content.strip()

                # Parse JSON
                try:
                    parsed_raw = json.loads(clean_content)
                except json.JSONDecodeError as jde:
                    last_error = f"Invalid JSON syntax: {jde}"
                    continue

                # Validate Pydantic schema
                try:
                    instance = schema_cls.model_validate(parsed_raw)

                    # Save to cache
                    if self.cache_dir:
                        cache_file = self.cache_dir / f"{cache_key}.json"
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "model": self.profile.model,
                                    "data": instance.model_dump(),
                                    "tokens": {"prompt": p_tokens, "completion": c_tokens},
                                    "cost_usd": call_cost,
                                },
                                f,
                                indent=2,
                                ensure_ascii=False,
                            )

                    return instance, {
                        "cached": False,
                        "cost_usd": call_cost,
                        "prompt_tokens": p_tokens,
                        "completion_tokens": c_tokens,
                    }
                except ValidationError as ve:
                    last_error = f"Pydantic validation error: {ve}"
                    continue

            except (openai.AuthenticationError, anthropic.AuthenticationError) as auth_err:
                raise RuntimeError(
                    f"Ошибка аутентификации LLM API ({auth_err}). Проверьте наличие корректного API-ключа в переменной окружения {self.profile.api_key_env}."
                ) from auth_err
            except (openai.APIConnectionError, anthropic.APIConnectionError) as conn_err:
                raise RuntimeError(
                    f"Ошибка подключения к LLM API ({conn_err}). Проверьте доступность URL: {self.profile.base_url}"
                ) from conn_err
            except Exception as exc:
                last_error = str(exc)
                logger.warning("LLM API call exception: %s", exc)

        raise ValueError(f"Failed to obtain valid LLM structured output after {max_retries} retries: {last_error}")
