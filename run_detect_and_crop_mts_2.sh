#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
set -euo pipefail

cd "$(dirname "$0")"

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе"
INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/1977 - 1978-03 (300 dpi, лёгкие расфокусы)"
OUTPUT_DIR="/mnt/system/raw/mts/77_78_defocus/cropped"
DEBUG_DIR="/mnt/system/raw/mts/77_78_defocus/debug"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.scan_cropping \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin -120 \
    --bottom-margin -180 \
    --left-margin -230 \
    --right-margin -230 \
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
    --log-level=INFO \
    --remove-fingers \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-px=32 \
    --crop-fill-fade=0.0 \

