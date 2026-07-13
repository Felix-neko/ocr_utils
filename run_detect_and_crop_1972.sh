#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
set -euo pipefail

cd "$(dirname "$0")"

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/1972 готово"
INPUT_DIR="/mnt/system/raw/плохие сканы  ВЭ/06"
OUTPUT_DIR="/mnt/system/raw/mts/ve_cropped/1958/06_6"
DEBUG_DIR="/mnt/system/raw/mts/ve_debug/1958/06_6"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.detect_and_crop \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --x-margins -140 \
    --y-margins -140 \
    --output-format png \
    --compensate-levels \
    --finger-dilate-px=60 \
    --finger-zone-light-increment=20,40 \
    --remove-fingers

