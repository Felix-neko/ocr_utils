#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
set -euo pipefail

cd "$(dirname "$0")"

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе"
INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-4 1978-04 - 1985 (450 DPI)"
OUTPUT_DIR="/media/felix/SYSTEM/raw/mts/pack_4_v_1/cropped"
DEBUG_DIR="/media/felix/SYSTEM/raw/mts/pack_4_v_1/debug"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.scan_cropping \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin -180 \
    --bottom-margin -270 \
    --left-margin -360 \
    --right-margin -360 \
    --output-format tiff \
    --force-dpi=450 \
    --compensate-levels \
    --finger-dilate-px=120 \
    --max-asymmetric-dilation-ratio=1.
    --finger-zone-light-increment=20,20 \
    --extra-erosion-px=120 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-px=18,72 \
    --bg-fill-method=nearest \
    --bg-fill-blur-px=16 \
    --log-level=INFO \
    --remove-fingers \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-px=32 \
    --crop-fill-fade=0.0 \

