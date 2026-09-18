"""Клиент OpenRouter поверх штатного SDK ``openrouter``: один вызов chat/completions с ретраями и учётом цены.

SDK (пакет ``openrouter``, генерируется Speakeasy) даёт типизированный ``chat.send`` и ``credits``;
здесь остаётся своё, чего в нём нет или что он делает не так, как нам нужно:

* ретраи — SDK повторяет только 5xx, а нам нужны ещё 429/408 и «200 с ошибкой провайдера в теле»,
  поэтому у SDK ретраи выключены и цикл с backoff свой;
* ``reasoning: {"enabled": false}`` из старых payload SDK молча выбрасывает (его модель знает только
  ``effort``), а без выключенного thinking DeepSeek стоит втрое — адаптер переводит в ``effort: none``;
* имя провайдера, обслужившего запрос, в модели ответа SDK нет — оно берётся из ``openrouter_metadata``
  (запрашивается заголовком, выбранный endpoint помечен ``selected``).

Стоимость запроса OpenRouter кладёт в каждый ответ (``usage.cost``, доллары), так что считать её по
прайсу не нужно — и не стоит: у одной модели десяток провайдеров с разными ценами, а маршрутизатор
выбирает между ними сам.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from openrouter import OpenRouter, errors
from openrouter.components import ChatResult
from openrouter.utils import BackoffStrategy, RetryConfig

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_KEY = "OPENROUTER_API_KEY"
DEFAULT_TIMEOUT = 300.0
DEFAULT_ATTEMPTS = 5
# Ретраить имеет смысл только на перегрузе и сетевых сбоях; 400/401/402/403 — наша ошибка
# или кончились деньги, повтор их не вылечит.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# Заголовки-визитка OpenRouter: попадают в их статистику, на маршрутизацию не влияют.
APP_REFERER = "https://github.com/Felix-Neko/ocr_utils"
APP_TITLE = "ocr_utils external_ocr_services"
# Ретраи самого SDK выключены: он повторяет только 5xx, а наш цикл ниже покрывает и 429, и сеть,
# и ошибку провайдера внутри 200; два цикла друг над другом дали бы до 25 попыток.
_NO_SDK_RETRIES = RetryConfig("none", BackoffStrategy(0, 0, 1.0, 0), False)


class OpenRouterError(RuntimeError):
    """Сбой запроса: HTTP-статус (если был), начало тела ответа; сеть — без статуса.

    Args:
        message: Текст ошибки для лога и meta (``error``).
        status: HTTP-статус или код ошибки провайдера из тела; ``None`` — сетевой сбой.
        body: Начало тела ответа (обрезано) — чтобы понять причину без повторного запроса.
    """

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ChatResponse:
    """Разобранный ответ chat/completions: текст, кто обслужил, токены, цена, время, число попыток.

    Args:
        text: Текст ответа модели (обычно JSON страницы).
        finish_reason: Почему модель остановилась (``stop``, ``length`` …); ``length`` — упёрлась в потолок.
        provider: Провайдер OpenRouter, обслуживший запрос (из ``openrouter_metadata``).
        model: Идентификатор модели, который вернул сервер.
        request_id: Идентификатор генерации у OpenRouter — для сверки в их кабинете.
        prompt_tokens: Токенов входа (с картинками).
        completion_tokens: Токенов выхода.
        reasoning_tokens: Токенов рассуждений внутри выхода; при выключенном thinking — 0.
        cost_usd: Стоимость по ``usage.cost``; ``None`` — провайдер не сообщил.
        latency_s: Время от первой попытки до ответа, с (с паузами между попытками).
        attempts: Номер удачной попытки (1 — с первого раза).
        raw_usage: Блок ``usage`` ответа как есть — на случай новых полей.
        cached_tokens: Токены входа, взятые провайдером из кэша префикса; по ним видно, работает ли кэш.
    """

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
    cached_tokens: int = 0


def api_key_from(option: str | None) -> str:
    """Ключ: явная опция → переменная окружения → ошибка. В логи и meta ключ не попадает.

    Args:
        option: Значение ``--api-key``; ``None`` или пусто — взять из ``$OPENROUTER_API_KEY``.
    """
    key = option or os.environ.get(ENV_KEY)
    if not key:
        raise OpenRouterError(f"нет ключа OpenRouter: передайте --api-key или задайте ${ENV_KEY}")
    return key


def _content_text(content: Any) -> str:
    """Текст ответа: строка или список частей (некоторые провайдеры отдают части).

    Args:
        content: ``choices[0].message.content`` из модели ответа SDK.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
    return ""


def adapt_reasoning(reasoning: dict | None) -> dict | None:
    """Поле ``reasoning`` под модель SDK: ``{"enabled": false}`` → ``{"effort": "none"}``.

    SDK знает только ``effort`` и ``summary``; ``enabled`` и ``max_tokens`` он молча отбросил бы,
    и thinking у DeepSeek включился бы обратно. ``effort: none`` по докам OpenRouter выключает
    рассуждения целиком. ``max_tokens`` (потолок thinking у Gemini) через SDK не передать — только
    предупреждение.

    Args:
        reasoning: Поле ``reasoning`` payload в форме HTTP API; ``None`` — не слать.

    Returns:
        Словарь для аргумента ``reasoning`` SDK (только ``effort`` / ``summary``) или ``None``, если
        слать нечего.
    """
    if reasoning is None:
        return None
    result = {key: value for key, value in reasoning.items() if key in ("effort", "summary")}
    if reasoning.get("enabled") is False:
        result["effort"] = "none"
    if "max_tokens" in reasoning:
        logger.warning(
            "reasoning.max_tokens=%s SDK не поддерживает — потолок thinking не передан", reasoning["max_tokens"]
        )
    return result or None


def _provider_from(result: ChatResult) -> str | None:
    """Имя провайдера из ``openrouter_metadata``: endpoint с ``selected``; иначе последний из попыток.

    Args:
        result: Модель ответа SDK; без metadata (заголовок не слали) — ``None``.
    """
    metadata = result.openrouter_metadata
    if metadata is None:
        return None
    for endpoint in metadata.endpoints.available:
        if endpoint.selected:
            return endpoint.provider
    if metadata.attempts:
        return metadata.attempts[-1].provider
    return None


def _error_from_body(body: str) -> tuple[str, int | None] | None:
    """OpenRouter умеет отдать 200 с ``{"error": {...}}`` вместо ответа; SDK на этом падает разбором.

    Возвращает (сообщение, код) или ``None``, если тело — не такая ошибка.

    Args:
        body: Сырой текст ответа из ``ResponseValidationError.body``.
    """
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or "error" not in parsed or parsed.get("choices"):
        return None
    error = parsed["error"] if isinstance(parsed["error"], dict) else {"message": str(parsed["error"])}
    code = int(error.get("code") or 0) or None
    return f"провайдер: {error.get('message')}", code


class OpenRouterClient:
    """Синхронный клиент: ``chat(payload)`` с ретраями; ``http_client`` — подмена httpx в тестах.

    Args:
        api_key: Ключ OpenRouter (из ``api_key_from``); в логи и repr не попадает.
        timeout: Таймаут одного запроса, с (``--timeout``); полоса с 4 тайлами читается до минуты.
        attempts: Попыток на запрос при 429/5xx/сети (``--attempts``).
        http_client: Готовый ``httpx.Client`` (в тестах — с ``MockTransport``); ``None`` — свой у SDK.
        sleep: Функция паузы между попытками; в тестах подменяется, чтобы не ждать.
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT,
        attempts: int = DEFAULT_ATTEMPTS,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self._sleep = sleep
        self._sdk = OpenRouter(
            api_key=api_key,
            timeout_ms=int(timeout * 1000),
            retry_config=_NO_SDK_RETRIES,
            http_referer=APP_REFERER,
            x_open_router_title=APP_TITLE,
            client=http_client,
        )

    def chat(self, payload: dict[str, Any]) -> ChatResponse:
        """Один запрос chat/completions по «сырому» payload (как у HTTP API) с ретраями по RETRY_STATUSES.

        Args:
            payload: Тело запроса в форме HTTP API (``ocr.build_payload``): ``model``, ``messages``,
                ``temperature``, ``max_tokens``, ``provider``, ``response_format``, ``reasoning``.

        Returns:
            ``ChatResponse`` удачной попытки; исчерпаны попытки или ошибка без ретрая — ``OpenRouterError``.
        """
        kwargs = self._kwargs(payload)
        started = time.monotonic()
        last_error: OpenRouterError | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                result = self._sdk.chat.send(**kwargs)
            except errors.ResponseValidationError as error:
                # Тело не легло в модель SDK. При 200 это ошибка провайдера внутри ответа (код — в
                # теле); при другом статусе — просто ответ с ошибкой не по схеме SDK, статус главнее.
                body = error.body or ""
                found = _error_from_body(body)
                if error.status_code != 200:
                    message, code = (found[0] if found else f"HTTP {error.status_code}"), error.status_code
                elif found is None:
                    raise OpenRouterError(f"ответ не разобран: {error.message}", error.status_code, body[:2000])
                else:
                    message, code = found
                if code in RETRY_STATUSES:
                    last_error = OpenRouterError(message, code, body[:500])
                else:
                    raise OpenRouterError(message, code, body[:2000])
            except errors.OpenRouterError as error:
                status = error.status_code
                if status in RETRY_STATUSES:
                    last_error = OpenRouterError(f"HTTP {status}", status, (error.body or "")[:500])
                else:
                    raise OpenRouterError(f"HTTP {status}", status, (error.body or "")[:2000])
            except httpx.HTTPError as error:  # таймаут, обрыв, DNS — статуса нет
                last_error = OpenRouterError(f"сеть: {error!r}")
            else:
                return self._parse(result, time.monotonic() - started, attempt)
            if attempt < self.attempts:
                delay = min(60.0, 2.0 * 2**attempt) + random.uniform(0, 1)
                logger.warning("попытка %d/%d не удалась (%s), пауза %.0f с", attempt, self.attempts, last_error, delay)
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _kwargs(payload: dict[str, Any]) -> dict[str, Any]:
        """Payload в стиле HTTP API → именованные аргументы ``chat.send``; metadata запрашивается всегда.

        Поле, которого адаптер не знает, — ошибка, а не молчаливая потеря: SDK строгий, и
        неизвестный параметр до модели всё равно не дошёл бы.

        Args:
            payload: Тело запроса в форме HTTP API.
        """
        kwargs: dict[str, Any] = {"x_open_router_metadata": "enabled"}
        for key in ("model", "messages", "temperature", "max_tokens", "provider", "response_format"):
            if key in payload:
                kwargs[key] = payload[key]
        reasoning = adapt_reasoning(payload.get("reasoning"))
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        unknown = set(payload) - set(kwargs) - {"reasoning"}
        if unknown:
            raise OpenRouterError(f"payload содержит поля, которые адаптер не передаёт: {sorted(unknown)}")
        return kwargs

    @staticmethod
    def _parse(result: ChatResult, latency: float, attempts: int) -> ChatResponse:
        """Модель ответа SDK → ``ChatResponse``; ответ без ``choices`` — ошибка.

        Args:
            result: ``ChatResult`` из ``chat.send``.
            latency: Время от первой попытки до этого ответа, с.
            attempts: Номер удачной попытки.
        """
        if not result.choices:
            raise OpenRouterError(f"в ответе нет choices: {result.model_dump_json()[:500]}")
        choice = result.choices[0]
        usage = result.usage
        # У SDK «не пришло» — это None или UNSET; приводим к простым числам, как раньше из JSON.
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
        cost = getattr(usage, "cost", None) if usage else None
        return ChatResponse(
            text=_content_text(choice.message.content),
            finish_reason=choice.finish_reason,
            provider=_provider_from(result),
            model=result.model,
            request_id=result.id,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            latency_s=latency,
            attempts=attempts,
            raw_usage=usage.model_dump(exclude_unset=True) if usage is not None else {},
            cached_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
        )


def credits(api_key: str, timeout: float = 30.0) -> dict:
    """Баланс ключа: ``{"total_credits": ..., "total_usage": ...}`` — для сверки суммы прогона.

    Args:
        api_key: Ключ OpenRouter.
        timeout: Таймаут запроса, с.

    Returns:
        ``{"total_credits": куплено, "total_usage": потрачено}`` в долларах за всё время ключа.
    """
    sdk = OpenRouter(api_key=api_key, timeout_ms=int(timeout * 1000), retry_config=_NO_SDK_RETRIES)
    data = sdk.credits.get_credits().data
    return {"total_credits": data.total_credits, "total_usage": data.total_usage}
