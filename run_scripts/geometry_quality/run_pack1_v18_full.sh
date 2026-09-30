#!/usr/bin/env bash
#
# Боевой прогон детектора порчи геометрии v18 по паку-1 и выгрузки для просмотра — после свежего разбора page_layout
# обоих PDF ($GEOMETRY_LAYOUT_ROOT) и свежего прогона v16 ($GEOMETRY_V16_DIR):
#
# 1. мера всех страниц v18 и отчёт с эталоном (run_pack1_v18.sh) → $GEOMETRY_V18_DIR;
# 2. смены вердикта против прошлого прогона $PREV_RUN_DIR (по умолчанию pack1_v18f — v18 на разборе v6 и прежнем
#    кэше v16) → $DIFF_DIR/<было>_to_<стало>/{lineart,other}/, diff.csv, diff.md;
# 3. все страницы с line art или формулами по вердиктам good/mixed/bad и годам → $EXPORT_DIR/lineart_formula/;
# 4. страницы без line art с вердиктом bad и mixed по худшей метрике и поясам её score → $EXPORT_DIR/damage_belts/.
#
# Время: мера ~4 мин (16 воркеров), выгрузки — секунда на картинку на воркер (рендер двух PDF), 8 воркеров.
# Ждать по сохранённому PID:
#     setsid bash run_scripts/geometry_quality/run_pack1_v18_full.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

PREV_RUN_DIR="${PREV_RUN_DIR:-$QUALITY_ROOT/pack1_v18f}"
STAMP="$(basename "$GEOMETRY_V18_DIR")"
DIFF_DIR="${DIFF_DIR:-$MARKUP_ROOT/pack1_geometry_${STAMP}_diff}"
EXPORT_DIR="${EXPORT_DIR:-$MARKUP_ROOT/pack1_geometry_${STAMP}_export}"
DRAW_JOBS="${DRAW_JOBS:-8}"

bash "$(dirname "$0")/run_pack1_v18.sh"
QUALITY_RUN_DIR="$PREV_RUN_DIR" QUALITY_RUN_DIR_V18="$GEOMETRY_V18_DIR" DIFF_DIR="$DIFF_DIR" \
    bash "$(dirname "$0")/run_diff_v17_v18.sh"
for mode in lineart_formula damage_belts; do
    uv run python -m research.geometry_quality export --run-dir "$GEOMETRY_V18_DIR" --geo-dir "$GEO_PDF_DIR" \
        --nogeo-dir "$NOGEO_PDF_DIR" --out-dir "$EXPORT_DIR/$mode" --layout-root "$QUALITY_LAYOUT_ROOT" --mode "$mode" \
        --jobs "$DRAW_JOBS"
done
