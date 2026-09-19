#!/usr/bin/env bash
#
# Обзор текстового слоя всех страниц пака-1: формы спанов FineReader, повёрнутые матрицы,
# растяжения, привязка глифов к символам → survey.csv. Только чтение PDF.
#
# Читает: $TEXT_LAYER_PDF_DIR (123 выпуска, ~12 тыс. страниц).
# Пишет: $TEXT_LAYER_RUN_DIR/survey.csv.
# Идёт ~0.05–0.1 с на страницу на воркер (разбор потока + rawdict): 16 воркеров → 1–2 минуты.
#
# Запуск в фон: setsid ./run_survey.sh > лог 2>&1 < /dev/null & PID=$!; ждать по PID.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

JOBS="${JOBS:-16}"
RESERVE_CPU_CORES=4  # оставить ядра машине: прогон короткий, но пусть не душит остальное

uv run python -m ocr_utils.text_layer_fix survey \
    --pdf-dir "$TEXT_LAYER_PDF_DIR" \
    --out-dir "$TEXT_LAYER_RUN_DIR" \
    --jobs "$JOBS" \
    --reserve-cpu-cores "$RESERVE_CPU_CORES" \
    "$@"
