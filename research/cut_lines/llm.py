"""Текст срезанной строки через DeepSeek (OpenRouter): картинка строки донора, контекст строк над ней и черновик tesseract → строка; ответы кэшируются."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Модель (реестр ``external_ocr_services.models``) и потолок токенов ответа.
MODEL = "deepseek-v41-flash"
MAX_TOKENS = 600
# Версия промпта: правка текста поднимает версию (кэш ответов разделяется по ней).
PROMPT_VERSION = 2

SYSTEM = (
    "You restore one printed line of a Russian economics journal from 1991. You get a photo crop of a page "
    "column: the line to restore is the LAST (bottom-most) text line in the crop; the lines above are context. "
    "The crop may be damaged near the book spine (smeared or missing letters at one end of the lines). You also "
    "get a draft OCR of the bottom line and the OCR of the lines above (may contain errors). Return the exact "
    "text of the whole bottom line as printed, from its left end to its right end: keep the original spelling, "
    "hyphenation at the line end (a trailing '-'), punctuation and spaces. Where letters are unreadable, infer "
    "them from the context so that the words are correct Russian and continue the sentence of the line above. "
    'Do not add words that do not fit the line. Answer JSON: {"text": "..."}.'
)


def data_url(img: np.ndarray) -> str:
    """Картинка как data-URL PNG.

    Args:
        img: BGR или серое.

    Returns:
        Строка ``data:image/png;base64,...``.
    """
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def cache_key(strip: np.ndarray, context: str, draft: str) -> str:
    """Ключ кэша: хэш картинки, контекста, черновика и версии промпта."""
    h = hashlib.sha256()
    h.update(cv2.imencode(".png", strip)[1].tobytes())
    h.update(f"{context}\n{draft}\n{MODEL}\n{PROMPT_VERSION}".encode())
    return h.hexdigest()[:24]


def parse_text(answer: str) -> str | None:
    """Текст строки из ответа модели: JSON ``{"text": ...}``, в том числе внутри ```-блока или с мусором вокруг.

    Args:
        answer: Ответ модели.

    Returns:
        Текст или None.
    """
    match = re.search(r"\{.*\}", answer, flags=re.S)
    if not match:
        return None
    try:
        return str(json.loads(match.group(0))["text"]).strip()
    except (ValueError, KeyError, TypeError):
        return None


def line_text(strip: np.ndarray, context: str, draft: str, cache_dir: Path) -> str:
    """Текст строки по DeepSeek (с кэшем); при ошибке — черновик tesseract.

    Args:
        strip: Выпрямленная полоса строки донора (BGR), лучше крупно (буквы ≥ 25 px).
        context: Текст строк над ней (основа), через перевод строки.
        draft: Черновик tesseract.
        cache_dir: Папка кэша ответов.

    Returns:
        Текст строки.
    """
    from ocr_utils.external_ocr_services.client import OpenRouterClient, api_key_from
    from ocr_utils.external_ocr_services.models import resolve

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(strip, context, draft)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]
    spec = resolve(MODEL)
    user = f"OCR of the lines above (may be wrong):\n{context}\n\nDraft OCR of the bottom line:\n{draft}\n\nThe crop is attached."
    payload = {
        "model": spec.openrouter_id,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": user},
                                         {"type": "image_url", "image_url": {"url": data_url(strip)}}]},  # fmt: skip
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        # Рассуждения выключены: иначе весь потолок токенов уходит на thinking, и ответ приходит пустым.
        "reasoning": {"enabled": False},
        "provider": {"order": list(spec.provider_order), "ignore": list(spec.provider_ignore)},
    }
    try:
        resp = OpenRouterClient(api_key_from(None)).chat(payload)
    except Exception as exc:  # сеть — строка остаётся по tesseract
        log.warning("DeepSeek: %r — беру черновик tesseract", exc)
        return draft
    text = parse_text(resp.text)
    if text is None:
        log.warning("DeepSeek: ответ не разобран (%r) — беру черновик tesseract", resp.text[:300])
        return draft
    path.write_text(json.dumps({"text": text, "draft": draft, "context": context}, ensure_ascii=False), "utf-8")
    return text
