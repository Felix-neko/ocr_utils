"""Клиент OpenRouter поверх ``requests``: один вызов chat/completions с ретраями и учётом цены.

Стоимость запроса OpenRouter теперь кладёт в каждый ответ (``usage.cost``, доллары), так
что считать её по прайсу не нужно — и не стоит: у одной модели десяток провайдеров с
разными ценами, а маршрутизатор выбирает между ними сам.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_KEY = "OPENROUTER_API_KEY"
DEFAULT_TIMEOUT = 300.0
DEFAULT_ATTEMPTS = 5
# Ретраить имеет смысл только на перегрузе и сетевых сбоях; 400/401/402/403 — наша ошибка
# или кончились деньги, повтор их не вылечит.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ChatResponse:
    text: str
    finish_reason: str | None
    provider: str | None
    model: str | None
    request_id: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    latency_s: float
    attempts: int
    raw_usage: dict = field(default_factory=dict)


def api_key_from(option: str | None) -> str:
    """Ключ: явная опция → переменная окружения → ошибка. В логи и meta ключ не попадает."""
    key = option or os.environ.get(ENV_KEY)
    if not key:
        raise OpenRouterError(f"нет ключа OpenRouter: передайте --api-key или задайте ${ENV_KEY}")
    return key


def _content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # некоторые провайдеры отдают список частей
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT,
        attempts: int = DEFAULT_ATTEMPTS,
        post: Callable[..., requests.Response] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Заголовки-визитка OpenRouter: попадают в их статистику, на маршрутизацию не влияют.
            "HTTP-Referer": "https://github.com/Felix-Neko/ocr_utils",
            "X-Title": "ocr_utils external_ocr_services",
        }
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self._post = post or requests.post
        self._sleep = sleep

    def chat(self, payload: dict[str, Any]) -> ChatResponse:
        started = time.monotonic()
        last_error: OpenRouterError | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                response = self._post(API_URL, headers=self._headers, json=payload, timeout=self.timeout)
            except requests.RequestException as error:
                last_error = OpenRouterError(f"сеть: {error}")
            else:
                if response.status_code in RETRY_STATUSES:
                    last_error = OpenRouterError(
                        f"HTTP {response.status_code}", response.status_code, response.text[:500]
                    )
                elif response.status_code != 200:
                    raise OpenRouterError(f"HTTP {response.status_code}", response.status_code, response.text[:2000])
                else:
                    body = response.json()
                    if "error" in body and not body.get("choices"):
                        # OpenRouter умеет отдать 200 с ошибкой провайдера внутри тела.
                        error = body["error"]
                        code = int(error.get("code") or 0) or None
                        message = f"провайдер: {error.get('message')}"
                        if code in RETRY_STATUSES:
                            last_error = OpenRouterError(message, code, str(error)[:500])
                        else:
                            raise OpenRouterError(message, code, str(error)[:2000])
                    else:
                        return self._parse(body, time.monotonic() - started, attempt)
            if attempt < self.attempts:
                delay = min(60.0, 2.0 * 2**attempt) + random.uniform(0, 1)
                logger.warning("попытка %d/%d не удалась (%s), пауза %.0f с", attempt, self.attempts, last_error, delay)
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _parse(body: dict, latency: float, attempts: int) -> ChatResponse:
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterError(f"в ответе нет choices: {str(body)[:500]}")
        choice = choices[0]
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        cost = usage.get("cost")
        return ChatResponse(
            text=_content_text(choice.get("message") or {}),
            finish_reason=choice.get("finish_reason") or choice.get("native_finish_reason"),
            provider=body.get("provider"),
            model=body.get("model"),
            request_id=body.get("id"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            cost_usd=float(cost) if cost is not None else None,
            latency_s=latency,
            attempts=attempts,
            raw_usage=usage,
        )


def credits(api_key: str, timeout: float = 30.0) -> dict:
    """Баланс ключа: ``{"total_credits": ..., "total_usage": ...}`` — для сверки суммы прогона."""
    response = requests.get(
        "https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
    )
    response.raise_for_status()
    return response.json().get("data") or {}
