#!/usr/bin/env bash
#
# Стенд v17 по паку-1: мера всех страниц с разбором обоих вариантов (кэш), отчёт с вердиктами, сменами против
# v14/v16 и эталоном, оверлеи эталона «было | стало» с метриками и порогами в шапке.
#
# Читает: $QUALITY_LAYOUT_ROOT (разбор v6 по обоим PDF), $QUALITY_V16_DIR (кэш v16), эталон $GEOMETRY_LABELS.
# Пишет: $QUALITY_RUN_DIR/{cache/,report.md,verdicts.csv,sheets/}.
# Время: мера — доли секунды на страницу (разбор уже готов); страницы с line art ещё рендерятся из обоих PDF и
# меряются плотным полем (секунды). 12 135 страниц / 16 воркеров — минуты.
# Пороги — стартовые (research/geometry_quality/scoring.py); перекрыть: EXTRA="--thr line_quality_mm=1.0 --total 3".
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/geometry_quality/run_pack1.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

JOBS="${JOBS:-16}"
EXTRA="${EXTRA:-}"

uv run python -m research.geometry_quality measure --layout-root "$QUALITY_LAYOUT_ROOT" --v16-dir "$QUALITY_V16_DIR" \
    --run-dir "$QUALITY_RUN_DIR" --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --jobs "$JOBS"
# shellcheck disable=SC2086
uv run python -m research.geometry_quality report --run-dir "$QUALITY_RUN_DIR" --v14-csv "$QUALITY_V14_CSV" \
    --v16-csv "$QUALITY_V16_DIR/metrics.csv" --labels "$GEOMETRY_LABELS" $EXTRA
# shellcheck disable=SC2086
uv run python -m research.geometry_quality sheets --run-dir "$QUALITY_RUN_DIR" --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" --out-dir "$QUALITY_RUN_DIR/sheets/labels" --select labels --labels "$GEOMETRY_LABELS" \
    --v16-csv "$QUALITY_V16_DIR/metrics.csv" --limit 100 $EXTRA
