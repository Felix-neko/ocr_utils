#!/usr/bin/env bash
#
# Отчёт по группам дубликатов RAF (без копирования файлов) : ищем повторные кадры одной полосы
# методом AKAZE-признаков и показываем, какой файл в группе самый резкий.
# Печатаются только группы из двух и более файлов.
# Папка съёмки 2026-09-23 разложена по подпапкам 279–283 (ЭГ 1962 и 1985, сентябрь): обходим
# их рекурсивно (--recursive), дубликаты ищутся внутри каждой подпапки отдельно.
#
set -euo pipefail

# Скрипт лежит в run_scripts/<подсистема>/, а пути внутри отсчитываются от корня
# репозитория — поднимаемся на два уровня.
cd "$(dirname "$0")/../.."

INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/неразобранное/2026-09-23 ЭГ/"
REPORT_FILE="eg_2026_09_23.txt"
echo "select_best_raws (report):"
echo "  input  = $INPUT_DIR"
echo "  report = $REPORT_FILE"
uv run select_best_raws.py \
    "$INPUT_DIR" \
    --mode report \
    --recursive \
    --method local \
    --n-search 5 \
    --min-match-ratio 0.2 \
    --max-scale-change 1.15 \
    2>&1 | tee "$REPORT_FILE"
