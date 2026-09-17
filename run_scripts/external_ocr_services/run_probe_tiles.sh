#!/usr/bin/env bash
#
# Проба на полосах из probe_tiles_pages.txt плюс весь выпуск 1975/12 (двухполосное «Содержание»
# и указатель за год на 5+ полосах): проверить, что разворот 1967/10/IMG_0041 ушёл четырьмя
# тайлами (debug/…/IMG_0041.tile_{00,01,10,11}.jpg), а toc.json 1975/12 содержит contents из двух
# полос со слитой рубрикой и index из всех полос указателя.
#
# ПИШЕТ в отдельный корень EXTERNAL_OCR_SERVICES_ROOT/probe, чтобы не смешивать с боевым выходом.
# ОРИЕНТИР (замер 2026-09-17): 102 полосы, $0.11, 4 минуты; разворот ушёл 4 тайлами 2200×1822,
# «Содержание» 1975/12 слилось из 2 полос (11 рубрик, 27 статей), указатель — из 7 (19 рубрик, 299 статей).
set -euo pipefail
source "$(dirname "$0")/common.sh"

PROBE_ROOT="$EXTERNAL_OCR_SERVICES_ROOT/probe"
LIST="$(dirname "$0")/probe_tiles_pages.txt"

uv run python -m ocr_utils.external_ocr_services run \
    --in-dir "$SHARPENED_DIR" \
    --out-dir "$PROBE_ROOT/out" \
    --debug-dir "$PROBE_ROOT/debug" \
    --db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --pages "$LIST" \
    --source "$EXTERNAL_OCR_SOURCE" \
    --jobs "$JOBS" \
    --skip-done --on-missed-toc skip \
    "$@"
uv run python -m ocr_utils.external_ocr_services run \
    --in-dir "$SHARPENED_DIR" \
    --out-dir "$PROBE_ROOT/out" \
    --debug-dir "$PROBE_ROOT/debug" \
    --db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --only-year 1975 --only-issue 12 \
    --source "$EXTERNAL_OCR_SOURCE" \
    --jobs "$JOBS" \
    --skip-done --on-missed-toc skip \
    "$@"
uv run python -m ocr_utils.external_ocr_services balance
echo "Тайлы разворота: $(ls "$PROBE_ROOT/debug/1967/10/" | grep -c 'IMG_0041.tile_')"
