#!/usr/bin/env bash
#
# Детектор «FineReader ухудшил геометрию» по всему паку-1: пары страниц из PDF с коррекцией
# геометрии и без неё → поле смещений, штрихи, строки, кромки → metrics.csv и JSON на страницу.
#
# Читает: $GEO_PDF_DIR и $NOGEO_PDF_DIR (129 выпусков, ~12 тыс. страниц).
# Пишет: $GEOMETRY_RUN_DIR/{run.json,metrics.csv,cache/<pdf>/pNNN.json}.
# Идёт ~1–2 с на страницу на воркер: 16 воркеров → 20–30 минут на пак. Повторный запуск берёт
# готовые JSON из кэша (--skip-done). Размеры (штрих от 8 мм, строка от 25 мм) — в мм бумаги,
# из замера на эталонных страницах (research/geometry_regression/README.md).
#
# Запуск в фон: setsid ./run_pack1.sh > лог 2>&1 < /dev/null & PID=$!; ждать по PID.

set -euo pipefail
set -m
trap 'kill -- -$$' EXIT INT TERM
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

JOBS=16  # упирается в CPU (рендер JBIG2 + LSD + корреляция), диск SSD

uv run python -m research.geometry_regression run \
    --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" \
    --out-dir "$GEOMETRY_RUN_DIR" \
    --jobs "$JOBS" \
    --stroke-min-mm 8 \
    --line-min-mm 25 \
    "$@"
