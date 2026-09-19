#!/usr/bin/env bash
#
# Пересобрать выпуски пака-1 в целиковые markdown из готовых полос внешнего OCR — без запросов
# к модели. Обычно не нужен: `run` собирает каждый выпуск сам; сюда — после правки правил сборки
# (assemble.py) или ручной правки .json полос.
#
# Аргументы: необязательно год/номер (один выпуск); остальное уходит команде assemble
# (например, --no-join-paragraphs).
# ЧИТАЕТ  EXTERNAL_OCR_SERVICES_OUT/<год>/<номер>/*.json (и .meta.json — сбойные полосы).
# ПИШЕТ   EXTERNAL_OCR_SERVICES_OUT/<год>/<номер>.md и <номер>.pages.json.
# ОРИЕНТИР (замер 2026-09-19): 1966/03 — 97 полос за ~2 с (словарь pymorphy3 грузится 0.05 с);
# на весь пак ~5 мин. Однопоточный: работа — чтение JSON и регулярки, диск SSD.
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
    --out-dir "$EXTERNAL_OCR_SERVICES_OUT" \
    "${ONLY[@]}" \
    "$@"
