#!/usr/bin/env bash
#
# Стенд v15 детектора «FineReader ухудшил геометрию» по всему паку-1 (движок
# research.geometry_regression.v15: перекос блоков по полю и кромкам, наклон строк по проекции,
# наклон штрихов по краске, выигрыш по строкам таблиц, деcкью, снимки по одному).
#
# Читает: $GEO_PDF_DIR и $NOGEO_PDF_DIR (129 выпусков, ~12 тыс. страниц), кэш surya $LAYOUT_CACHE_DIR.
# Пишет: $GEOMETRY_RUN_DIR (по умолчанию …/pack1_v15)/{run.json,metrics.csv,cache/<pdf>/pNNN.json}.
# ~3–4 с на страницу на воркер: 12 воркеров → ~1 ч на пак. Повторный запуск берёт готовые JSON (--skip-done).
#
# Запуск в фон: setsid ./run_pack1_v15.sh > лог 2>&1 < /dev/null & PID=$!; ждать по PID.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

GEOMETRY_RUN_DIR="${GEOMETRY_RUN_DIR_V15:-$GEOMETRY_REGRESSION_ROOT/pack1_v15}"
JOBS="${JOBS:-16}"
RESERVE_CPU_CORES=4

uv run python -m research.geometry_regression run \
    --engine v15 \
    --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" \
    --out-dir "$GEOMETRY_RUN_DIR" \
    --jobs "$JOBS" \
    --reserve-cpu-cores "$RESERVE_CPU_CORES" \
    --stroke-min-mm 4 \
    --line-min-mm 25 \
    --layout-cache "$LAYOUT_CACHE_DIR" \
    "$@"
