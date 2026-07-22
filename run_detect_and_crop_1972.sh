#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
set -euo pipefail

cd "$(dirname "$0")"

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе"
#INPUT_DIR="/mnt/system/raw/плохие сканы  ВЭ/06"
#INPUT_DIR="/mnt/system/raw/ve_80s/in"
#INPUT_DIR="/mnt/system/raw/ve_80s/in/1989/06 проверить зональный пересвет"
#OUTPUT_DIR="/mnt/system/raw/ve_80s/test_896_tiff_9/out"
#DEBUG_DIR="/mnt/system/raw/ve_80s/test_896_tiff_9/debug"

INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/1966 готово/04 готово"
OUTPUT_DIR="/mnt/system/raw/mts/iter_8/out/1966 готово/04 готово"
DEBUG_DIR="/mnt/system/raw/mts/iter_8/debug/1966 готово/04 готово"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.detect_and_crop \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin -120 \
    --bottom-margin -120 \
    --left-margin -180 \
    --right-margin -180 \
    --output-format tiff \
    --force-dpi=300 \
    --compensate-levels \
    --finger-dilate-px=60 \
    --max-asymmetric-dilation-ratio=2 \
    --finger-zone-light-increment=20,20 \
    --extra-erosion-px=110 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-px=12,48 \
    --bg-fill-method=nearest \
    --bg-fill-blur-px=16 \
    --remove-fingers

