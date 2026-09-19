#!/usr/bin/env bash
#
# Один выпуск заострённых полос через DeepSeek V4.1 Flash в два этапа: оглавление (теги из
# DB_REVIEWED) -> список статей -> остальные полосы со списком в промпте.
#
# Аргументы: выпуск как год/номер (по умолчанию 1966/03); остальное уходит команде run.
# ЧИТАЕТ  SHARPENED_DIR/<выпуск>, DB_REVIEWED (только чтение).
# ПИШЕТ   EXTERNAL_OCR_SERVICES_OUT/<выпуск>/… (полосы; рядом <год>/<номер>.md — весь выпуск одним
#         файлом и <номер>.pages.json), EXTERNAL_OCR_SERVICES_DEBUG/<выпуск>/…,
#         EXTERNAL_OCR_SERVICES_CACHE/<выпуск>/… (кэш запросов: повтор с тем же промптом бесплатен).
# Тайлы: шаг 4500 px исходника (обычная полоса 1×2, разворот 2×2), 2200 px модели — замер по базе
# пака-1, см. docstring ocr_utils/external_ocr_services/tiling.py. Оглавление, найденное
# моделью вне базы, перераспознаётся автоматически (--on-missed-toc redo по умолчанию).
# ОРИЕНТИР (замер 2026-09-17): 1966/03 — 97 полос, $0.10, ~4 мин; 1975/12 — 100 полос (9 полос
# оглавления и указателя), $0.10, 4 мин. DeepSeek в трети ответов на длинный промпт со списком
# отвечает эхом response_format — код сам повторяет запрос без него (лишние ~0.03 ¢ на полосу).
set -euo pipefail
source "$(dirname "$0")/common.sh"

ISSUE="${1:-1966/03}"; shift || true
YEAR="${ISSUE%%/*}"
NUMBER="${ISSUE#*/}"

uv run python -m ocr_utils.external_ocr_services run \
    --in-dir "$SHARPENED_DIR" \
    --out-dir "$EXTERNAL_OCR_SERVICES_OUT" \
    --debug-dir "$EXTERNAL_OCR_SERVICES_DEBUG" \
    --cache-dir "$EXTERNAL_OCR_SERVICES_CACHE" \
    --db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --only-year "$YEAR" \
    --only-issue "$NUMBER" \
    --source "$EXTERNAL_OCR_SOURCE" \
    --jobs "$JOBS" \
    --skip-done \
    "$@"
uv run python -m ocr_utils.external_ocr_services balance
