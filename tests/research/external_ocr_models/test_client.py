"""Реэкспорт клиента OpenRouter в стенде: сами тесты клиента — в tests/ocr_utils/external_ocr_services/test_client.py."""

from ocr_utils.external_ocr_services import client as production
from research.external_ocr_models import client as stand


def test_stand_reexports_production_client():
    for name in ("OpenRouterClient", "OpenRouterError", "ChatResponse", "api_key_from", "credits", "RETRY_STATUSES"):
        assert getattr(stand, name) is getattr(production, name)
