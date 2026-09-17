"""Клиент OpenRouter переехал в боевой пакет ``ocr_utils.external_ocr_services.client``; здесь — реэкспорт.

Исследовательский стенд остаётся рабочим без правки импортов, а код клиента живёт в одном месте.
"""

from ocr_utils.external_ocr_services.client import (  # noqa: F401
    API_URL,
    DEFAULT_ATTEMPTS,
    DEFAULT_TIMEOUT,
    ENV_KEY,
    RETRY_STATUSES,
    ChatResponse,
    OpenRouterClient,
    OpenRouterError,
    api_key_from,
    credits,
)
