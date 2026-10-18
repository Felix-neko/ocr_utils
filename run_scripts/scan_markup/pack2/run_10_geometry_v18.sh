#!/usr/bin/env bash
#
# Боевой детектор порчи геометрии FineReader v18 по паку-2 (ocr_utils.geometry_regression.quality run): по каждой
# странице с разбором обоих PDF — мера v16 (кэш пуст, промахи меряются здесь же, ~6 с на страницу), меры v17–v18
# (строки и края блоков по разбору, строки без пары в A — перенос оси полем), verdicts.csv. Пороги — по умолчанию
# в коде (ocr_utils/geometry_regression/quality/scoring.py, калибровка по паку-1: reports/geometry_regression_v18.md).
# Его кэш читает сборщик финальных PDF (run_11_final_pdfs.sh).
#
# Сначала — разбор page_layout обоих PDF (run_9_page_layout_fr.sh).
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR, $GEOMETRY_LAYOUT_ROOT, $LAYOUT_CACHE_DIR.
# Пишет:  $GEOMETRY_V16_DIR/cache, $GEOMETRY_V18_DIR/{cache/,verdicts.csv}.
# Время: ~600 страниц × ~6 с v16 / 6 задач (по PDF на задачу) — около 10 мин; повтор с готовым кэшем — секунды.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/scan_markup/pack2/run_10_geometry_v18.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

# Задача пула — один PDF, а их 6: больше воркеров не займётся.
JOBS="${JOBS:-6}"

mkdir -p "$GEOMETRY_V16_DIR" "$GEOMETRY_V18_DIR"
uv run python -m ocr_utils.geometry_regression.quality run --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" \
    --layout-root "$GEOMETRY_LAYOUT_ROOT" --v16-dir "$GEOMETRY_V16_DIR" --run-dir "$GEOMETRY_V18_DIR" \
    --layout-cache "$LAYOUT_CACHE_DIR" --jobs "$JOBS" "$@"
