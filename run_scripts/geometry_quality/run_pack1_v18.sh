#!/usr/bin/env bash
#
# Стенд v18 по паку-1: v17 плюс форма line art сверх подобия (`lineart_shape_mm` и части) и строки без пары в A
# (ось A — перенос оси B плотным полем). Мера всех страниц (кэш v18), отчёт с эталоном.
#
# Читает: $QUALITY_LAYOUT_ROOT (разбор v6 по обоим PDF), $QUALITY_V16_DIR (кэш v16), оба PDF, эталон $GEOMETRY_LABELS.
# Пишет: $QUALITY_RUN_DIR_V18/{cache/,report.md,verdicts.csv}.
# Время: каждая страница рендерится из обоих PDF (строки без пары в A), страницы с line art ещё меряются плотным
# полем — около секунды на страницу; 12 135 страниц / 16 воркеров — минуты.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/geometry_quality/run_pack1_v18.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

JOBS="${JOBS:-16}"
EXTRA="${EXTRA:-}"
# По умолчанию — боевой прогон v18 из common.sh пака ($GEOMETRY_V18_DIR). pack1_v18 … pack1_v18f — прогоны
# отладки стенда (кэш той же версии, но старого кода и разбора v6): не использовать.
QUALITY_RUN_DIR_V18="${QUALITY_RUN_DIR_V18:-$GEOMETRY_V18_DIR}"

uv run python -m research.geometry_quality measure --layout-root "$QUALITY_LAYOUT_ROOT" --v16-dir "$QUALITY_V16_DIR" \
    --run-dir "$QUALITY_RUN_DIR_V18" --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --jobs "$JOBS"
# shellcheck disable=SC2086
uv run python -m research.geometry_quality report --run-dir "$QUALITY_RUN_DIR_V18" --v14-csv "$QUALITY_V14_CSV" \
    --v16-csv "$QUALITY_V16_DIR/metrics.csv" --labels "$GEOMETRY_LABELS" $EXTRA
