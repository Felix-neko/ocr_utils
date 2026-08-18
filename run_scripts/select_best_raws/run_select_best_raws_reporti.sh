#!/usr/bin/env bash
#
# Отчёт по группам дубликатов RAF (без копирования файлов) для папки
# «198_FUJI 1986 4-6 нечётная сторона»: ищем повторные кадры одной полосы
# методом AKAZE-признаков и показываем, какой файл в группе самый резкий.
# Печатаются только группы из двух и более файлов.
#
set -euo pipefail

# Скрипт лежит в run_scripts/<подсистема>/, а пути внутри отсчитываются от корня
# репозитория — поднимаемся на два уровня.
cd "$(dirname "$0")/../.."

INPUT_DIR="/mnt/system/raw/ЭГ/PEG 1959 07-12 even"
REPORT_FILE="59_712_even_report.txt"

echo "select_best_raws (report):"
echo "  input  = $INPUT_DIR"
echo "  report = $REPORT_FILE"

uv run select_best_raws.py \
    "$INPUT_DIR" \
    --mode report \
    --method local \
    --n-search 5 \
    --min-match-ratio 0.2 \
    --max-scale-change 1.15 \
    2>&1 | tee "$REPORT_FILE"
