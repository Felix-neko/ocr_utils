#!/usr/bin/env bash
#
# Пересобрать выпуски пака-1 в целиковые markdown из готовых полос внешнего OCR. К модели уходят
# только сомнительные стыки полос — один запрос на выпуск (README, «Проверка стыков моделью»),
# ответы кэшируются. Обычно не нужен: `run` собирает каждый выпуск сам; сюда — после правки правил
# сборки (assemble.py, boundary.py) или ручной правки .json полос.
#
# Аргументы: необязательно год/номер (один выпуск); остальное уходит команде assemble
# (например, --no-join-paragraphs, --no-check-boundaries).
# ЧИТАЕТ  EXTERNAL_OCR_SERVICES_PAGES/<год>/<номер>/*.json (и .meta.json — сбойные полосы, toc.json),
#         SHARPENED_DIR/<год>/<номер> (полоски строк для проверки стыков), ключ $OPENROUTER_API_KEY.
# ПИШЕТ   EXTERNAL_OCR_SERVICES_OUT/<год>/<год>_<номер>.md (только md), EXTERNAL_OCR_SERVICES_PAGES/<год>/<год>_<номер>.pages.json,
#         EXTERNAL_OCR_SERVICES_CACHE/…/_boundaries и _headings.
# ОРИЕНТИР (замер 2026-09-19): 1966/03 — 97 полос за ~2 с (словарь pymorphy3 грузится 0.05 с) плюс
# один запрос по стыкам ≈ 0.1 ¢; на весь пак ~5 мин и ≈ $0.1. Однопоточный: чтение JSON и регулярки.
set -euo pipefail
source "$(dirname "$0")/common.sh"

ISSUE="${1:-}"
if [[ -n "$ISSUE" && "$ISSUE" != --* ]]; then
    shift
    ONLY=(--only-year "${ISSUE%%/*}" --only-issue "${ISSUE#*/}")
else
    ONLY=()
fi

uv run python -m ocr_utils.external_ocr_services assemble \
    --pages-dir "$EXTERNAL_OCR_SERVICES_PAGES" \
    --issues-dir "$EXTERNAL_OCR_SERVICES_OUT" \
    --in-dir "$SHARPENED_DIR" \
    --cache-dir "$EXTERNAL_OCR_SERVICES_CACHE" \
    "${ONLY[@]}" \
    "$@"
