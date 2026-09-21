#!/usr/bin/env bash
#
# Детектор «FineReader ухудшил геометрию» по всему паку-1: пары страниц из PDF с коррекцией
# геометрии и без неё → поле смещений, штрихи, строки, кромки → metrics.csv и JSON на страницу.
#
# Читает: $GEO_PDF_DIR и $NOGEO_PDF_DIR (129 выпусков, ~12 тыс. страниц).
# Пишет: $GEOMETRY_RUN_DIR/{run.json,metrics.csv,cache/<pdf>/pNNN.json}.
# Идёт ~1–2 с на страницу на воркер: 16 воркеров → 20–30 минут на пак. Повторный запуск берёт
# готовые JSON из кэша (--skip-done). Размеры (штрих от 4 мм — дробные черты формул 5–8 мм,
# строка от 25 мм) — в мм бумаги, те же, что у регрессии на эталоне (`regress`),
# из замера на эталонных страницах (research/geometry_regression/README.md).
#
# Запуск в фон: setsid ./run_pack1.sh > лог 2>&1 < /dev/null & PID=$!; ждать по PID.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
# Абсолютный путь до sourcing: common.sh пака (через common.sh направления) сам делает cd в корень
# репо, после чего относительный dirname указывал бы мимо.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

# Упирается в CPU (рендер JBIG2 + LSD + корреляция), диск SSD. Оставить ядра под другое: JOBS=12 ./run_pack1.sh
JOBS="${JOBS:-16}"
RESERVE_CPU_CORES=4  # столько физических ядер оставить машине: 16 воркеров на 16 ядрах душат всё остальное

uv run python -m research.geometry_regression run \
    --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" \
    --out-dir "$GEOMETRY_RUN_DIR" \
    --jobs "$JOBS" \
    --reserve-cpu-cores "$RESERVE_CPU_CORES" \
    --stroke-min-mm 4 \
    --line-min-mm 25 \
    --layout-cache "$LAYOUT_CACHE_DIR" \
    "$@"
