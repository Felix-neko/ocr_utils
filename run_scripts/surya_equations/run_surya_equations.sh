#!/usr/bin/env bash
#
# Стенд research/surya_equations: как surya layout (класс Equation) находит выносные формулы пака-1.
#
# ЧТО ДЕЛАЕТ. collect — по всем 12 128 полосам с markdown внешнего OCR считает боксы surya Equation
# (кэш page_layout, вариант sharpened — те же заострённые JPEG, что видел DeepSeek) и выносные $$…$$
# DeepSeek V4.1 Flash, раскладывает полосы по слоям. evaluate — сверка с эталоном глазами
# (research/surya_equations/labels: 100 полос выборки по слоям, 89 выносных блоков): полнота и точность
# с пересчётом на пак, качество рамки. overlays — картинки размеченных полос по трём папкам.
# Разметка (view / snap / verdict) — руками, не из этого скрипта. Отчёт — reports/surya_equations.md.
#
# ЧИТАЕТ  LAYOUT_CACHE_DIR/sharpened, SHARPENED_DIR, EXTERNAL_OCR_SERVICES_PAGES (только чтение).
# ПИШЕТ   OUT (pages.csv, eval_*.csv, summary.json), OVERLAYS (reports/surya_equations, вне git).
# ВРЕМЯ   collect — секунды, evaluate — ~1 мин (бутстрэп), overlays — ~1 мин на 16 воркерах.
#
#     setsid bash run_scripts/surya_equations/run_surya_equations.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 10; done

set -euo pipefail
set -m
trap 'kill -- -$$' EXIT INT TERM
source "$(dirname "$0")/../external_ocr_services/common.sh"

OUT="/mnt/system/raw/mts/pack1_surya_equations"
OVERLAYS="$(dirname "$0")/../../reports/surya_equations"
JOBS=16

uv run python -m research.surya_equations collect \
    --layout-cache "$LAYOUT_CACHE_DIR" --pages-dir "$EXTERNAL_OCR_SERVICES_PAGES" --out-dir "$OUT" --jobs "$JOBS"
uv run python -m research.surya_equations evaluate --layout-cache "$LAYOUT_CACHE_DIR" --out-dir "$OUT"
uv run python -m research.surya_equations overlays \
    --sharpened-dir "$SHARPENED_DIR" --layout-cache "$LAYOUT_CACHE_DIR" --out-dir "$OVERLAYS" --jobs "$JOBS"
trap - EXIT
