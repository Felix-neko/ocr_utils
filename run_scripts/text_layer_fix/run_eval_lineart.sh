#!/usr/bin/env bash
#
# Точность и полнота источников line art против 221 ручной области line_art_schema
# (pack1_reviewed.sqlite) на PDF без коррекции геометрии: картинки FineReader, детектор
# таблиц (виды «схема»/«рисунок»), связные пятна line_art_detection, блоки Figure surya из
# кэша разметки по сканам и их объединение. Плюс 200 контрольных страниц без эталона —
# ложные срабатывания. → $TEXT_LAYER_RUN_DIR/{lineart_eval.csv,lineart_eval.json}.
#
# Читает: $TEXT_LAYER_NOGEO_PDF_DIR, $PROBE_DB, $DB_REVIEWED, $LAYOUT_CACHE_DIR, sample.csv прогона.
# Идёт ~1–2 с на страницу на воркер (~400 страниц): минута-две на 12 воркерах.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

JOBS="${JOBS:-16}"
RESERVE_CPU_CORES=4

uv run python -m research.text_layer_fix eval-lineart \
    --nogeo-dir "$TEXT_LAYER_NOGEO_PDF_DIR" \
    --out-dir "$TEXT_LAYER_RUN_DIR" \
    --probe-db "$PROBE_DB" \
    --markup-db "$DB_REVIEWED" \
    --layout-dir "$LAYOUT_CACHE_DIR" \
    --controls 200 \
    --jobs "$JOBS" \
    --reserve-cpu-cores "$RESERVE_CPU_CORES" \
    "$@"
