#!/usr/bin/env bash
#
# Шаг 3. Сравнение алгоритмов по ручной разметке (research/legacy/table_processing/labels).
# Дёшево и повторяемо: считает по 132 размеченным ячейкам восьми таблиц.
#
# GPU-детекторы (surya_lines, doctr) идут в том же прогоне, но последовательно — видеопамять
# одна на всех, в пул их заворачивать нельзя (CLAUDE.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

uv run python -m research.legacy.table_processing compare-rotation \
    --out-dir "$OUT_DIR" --report reports/table_processing_rotation.md

# PaddleOCR живёт в отдельном окружении; если оно не синхронизировано, движок просто
# не попадёт в сравнение:
#   uv sync --project research/legacy/table_processing/paddle_env
uv run python -m research.legacy.table_processing compare-ocr \
    --out-dir "$OUT_DIR" --report reports/table_processing_ocr.md
