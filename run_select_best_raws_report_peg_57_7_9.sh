#!/usr/bin/env bash
#
# Отчёт по группам дубликатов RAF (без копирования файлов) для папки
# «ЭГ 1957_07_12 чётная»: ищем повторные кадры одной полосы
# методом AKAZE-признаков и показываем, какой файл в группе самый резкий.
# Печатаются только группы из двух и более файлов.
#
set -euo pipefail

cd "$(dirname "$0")"

INPUT_DIR="/mnt/system/raw/2026-07-09 ЭГ/1957_07_12 четная"

echo "select_best_raws (report):"
echo "  input  = $INPUT_DIR"

uv run select_best_raws.py \
    "$INPUT_DIR" \
    --mode report \
    --method local \
    --n-search 5 \
    --min-match-ratio 0.2 \
    --max-scale-change 1.15
