#!/usr/bin/env bash
#
# Общие параметры боевого внешнего OCR по паку-2 (1977, выпуски 01-06; ocr_utils.external_ocr_services).
# Сам ничего не запускает — подключается через `source` из соседних run_*.sh. Схема та же, что у пака-1
# (../common.sh), пути — свои.
#
# Ключ OpenRouter берётся из $OPENROUTER_API_KEY; в скриптах и логах он не появляется.

# Пути пака (BLURRED_DIR, DB_REVIEWED и прочие) — из общего файла разметки пака-2.
source "$(dirname "${BASH_SOURCE[0]}")/../../scan_markup/pack2/common.sh"

# ВХОД — очищенные полосы без заострения (BLURRED_DIR, TIFF RGB), а не выгрузка Capture One: в заострённых
# копиях пака-2 повреждены растровые фотографии. Имена с отпечатком (0030_2R.e7eb772f.tif) сопоставляются
# с базой по cleaned_rel_path (pages.flags_from_db).
EXTERNAL_OCR_IN_DIR="$BLURRED_DIR"

# Выходы — на SSD, раскладка как у пака-1: полосы (pages), md выпусков (out), отладка, кэш запросов (cache:
# не удалять между прогонами — это и есть «продолжить с места»).
EXTERNAL_OCR_SERVICES_ROOT="/mnt/hotstore/scan_processing/mts/pack2_external_ocr_services"
EXTERNAL_OCR_SERVICES_PAGES="$EXTERNAL_OCR_SERVICES_ROOT/pages"
EXTERNAL_OCR_SERVICES_OUT="$EXTERNAL_OCR_SERVICES_ROOT/out"
EXTERNAL_OCR_SERVICES_DEBUG="$EXTERNAL_OCR_SERVICES_ROOT/debug"
EXTERNAL_OCR_SERVICES_CACHE="$EXTERNAL_OCR_SERVICES_ROOT/cache"

# Описание издания для промпта — без года выпуска (см. ../common.sh: общий кэшируемый префикс).
EXTERNAL_OCR_SOURCE="журнал «Материально-техническое снабжение», Москва, 1977"

# Сетка тайлов. Полосы пака-2 — 600 dpi, ~3350x5850 px (scan_markup/pack2/common.sh): при шаге 4500 px,
# как у пака-1 (4165x6733), обычная полоса режется так же, 1x2.
MAX_SRC_TILE_SIZE=4500

# Параллельных запросов — сеть, а не CPU; 12, как у пака-1 (429 у DeepSeek на них не было, см. ../common.sh).
JOBS=12
