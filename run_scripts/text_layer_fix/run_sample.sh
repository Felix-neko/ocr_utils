#!/usr/bin/env bash
#
# Разбор выборки страниц пака-1: зоны повёрнутого текста (ячейки таблиц, подписи на схемах,
# отдельный текст), вердикты по словам текстового слоя FineReader, чтение зон tesseract-ом
# → JSON на страницу в cache/, pages.csv, words.csv, zones.csv, sample.csv.
#
# Выборка: страницы таблиц с боковыми ячейками (summary.csv прогона rotated_text), полосы
# с ручной разметкой line_art_schema, полосы с rotate_cw=90, страницы с покрытием line art
# ≥ 5 % и 200 случайных контрольных — около 700 страниц.
# Читает: $TEXT_LAYER_PDF_DIR, $PROBE_DB, $DB_REVIEWED, $ROTATED_TABLES_DIR/summary.csv, $LINE_ART_CSV.
# Пишет: $TEXT_LAYER_RUN_DIR/{sample.csv,pages.csv,words.csv,zones.csv,cache/<pdf>/pNNNN.json}.
# Идёт 2–8 с на страницу на воркер (сетка и tesseract по ячейкам): 12 воркеров → 5–10 минут.
#
# Запуск в фон: setsid ./run_sample.sh > лог 2>&1 < /dev/null & PID=$!; ждать по PID.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

JOBS="${JOBS:-16}"
RESERVE_CPU_CORES=4  # tesseract на каждую ячейку в каждом воркере: оставить ядра машине

uv run python -m research.text_layer_fix run \
    --pdf-dir "$TEXT_LAYER_PDF_DIR" \
    --out-dir "$TEXT_LAYER_RUN_DIR" \
    --probe-db "$PROBE_DB" \
    --markup-db "$DB_REVIEWED" \
    --rotated-summary "$ROTATED_TABLES_DIR/summary.csv" \
    --line-art-csv "$LINE_ART_CSV" \
    --min-coverage 0.05 \
    --controls 200 \
    --jobs "$JOBS" \
    --reserve-cpu-cores "$RESERVE_CPU_CORES" \
    "$@"
