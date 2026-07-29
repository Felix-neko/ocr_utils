#!/usr/bin/env bash
#
# Экспорт страниц журнала «Плановое хозяйство» (пак Сафронова, переименованное)
# из PDF в картинки: рекурсивный обход всех PDF, для каждого — своя папка.
# Одиночные встроенные изображения вынимаются без перекодирования (JPEG остаётся
# JPEG), MRC-страницы и JPEG-2000 рендерятся/пересохраняются в PNG.
#
set -euo pipefail

cd "$(dirname "$0")"

INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/пак Сафронова/переименованное"
OUTPUT_DIR="/mnt/system/raw/planhoz/pack_1/exported"

echo "extract_images:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

uv run python -m ocr_utils.pdf_utils \
    "$INPUT_DIR" \
    "$OUTPUT_DIR"