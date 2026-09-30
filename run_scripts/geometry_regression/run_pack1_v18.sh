#!/usr/bin/env bash
#
# Боевой детектор порчи геометрии v18 по паку-1 (ocr_utils.geometry_regression.quality run): для каждой страницы с
# разбором page_layout обоих PDF — кэш v16 (при промахе v16 меряет страницу, ~6 с), меры v17–v18 (строки и края
# блоков по разбору, строки без пары в A — перенос оси полем, меры по плотному полю), verdicts.csv по порогам в коде.
# Его кэш читает сборщик финальных PDF (run_scripts/scan_markup/pack1/run_final_pdfs.sh).
#
# Сначала — разбор page_layout обоих PDF (run_scripts/page_layout/run_pack1_analysis_v6_fr.sh, GPU, ~4 ч): здесь он
# не строится. Кэш surya — $LAYOUT_CACHE_DIR (набивает тот же разбор).
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR, $GEOMETRY_LAYOUT_ROOT, $LAYOUT_CACHE_DIR, $GEOMETRY_V16_DIR/cache.
# Пишет: $GEOMETRY_V16_DIR/cache (только промахи), $GEOMETRY_V18_DIR/{cache/,verdicts.csv}.
# Время: с готовым кэшем v16 — ~4 мин на пак (16 воркеров); с пустым — ~1.5 ч (v16 ~6 с на страницу).
# Задача пула — один PDF (оба PDF открываются один раз на ~100 страниц).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/geometry_regression/run_pack1_v18.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

JOBS="${JOBS:-16}"

uv run python -m ocr_utils.geometry_regression.quality run --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" \
    --layout-root "$GEOMETRY_LAYOUT_ROOT" --v16-dir "$GEOMETRY_V16_DIR" --run-dir "$GEOMETRY_V18_DIR" \
    --layout-cache "$LAYOUT_CACHE_DIR" --jobs "$JOBS" "$@"
