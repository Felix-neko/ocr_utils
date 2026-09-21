#!/usr/bin/env bash
#
# Геометрия таблиц, которых FineReader не увидел: отбор, диагноз, выпрямление, листы.
#
# Замер по паку (что уже получено этими же командами):
#   find-broken       — 132 находки на страницах без таблицы в DOCX: 99 пропущенных таблиц,
#                       30 с боковым текстом (отдельная задача), 3 на боковых полосах;
#   diagnose-geometry — из 99: «геометрия ни при чём» 79, «крива уже на скане» 15,
#                       «FineReader выпрямил» 5, «FineReader искорёжил» НИ ОДНОЙ;
#   fix-geometry      — на 15 кривых: rules_separable снижает сагитту в 3.7 раза (медиана
#                       отношения 0.27) и улучшает 14 из 15; остальные четыре способа
#                       не меняют ничего.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# PDF без коррекции геометрии: нужен, чтобы отличить «было криво» от «искорёжил FineReader».
NOGEO_PDF_DIR="/mnt/system/raw/mts/pack1_pdf/full_pdfs_binary_no_bg_brightening_no_geometry_correction"

step() {
    uv run python -m research.legacy.table_processing "$@" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do sleep 10; done
    wait "$pid"
}

step find-broken --docx-dir "$DOCX_DIR" --pdf-dir "$RECOGNIZED_PDF_DIR" \
    --sharpened-dir "$SHARPENED_DIR" --db "$DB" --pack "$PACK" --out-dir "$OUT_DIR" --jobs "$JOBS"

step diagnose-geometry --out-dir "$OUT_DIR" --sharpened-dir "$SHARPENED_DIR" \
    --pdf-dir "$RECOGNIZED_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --jobs 10

step fix-geometry --out-dir "$OUT_DIR" --sharpened-dir "$SHARPENED_DIR" --jobs 8

step geometry-pairs --out-dir "$OUT_DIR" --per-sheet 3

echo "листы «было-стало»: $OUT_DIR/sheets_geometry"
echo "листы трёх источников: $OUT_DIR/geometry_pairs"
