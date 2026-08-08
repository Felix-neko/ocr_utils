#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
set -euo pipefail

# Скрипт лежит в run_scripts/<подсистема>/, а пути внутри отсчитываются от корня
# репозитория — поднимаемся на два уровня.
cd "$(dirname "$0")/../.."

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе"
#INPUT_DIR="/mnt/system/raw/плохие сканы  ВЭ/06"
#INPUT_DIR="/mnt/system/raw/ve_80s/in"
#INPUT_DIR="/mnt/system/raw/ve_80s/in/1989/06 проверить зональный пересвет"
#OUTPUT_DIR="/mnt/system/raw/ve_80s/test_896_tiff_9/out"
#DEBUG_DIR="/mnt/system/raw/ve_80s/test_896_tiff_9/debug"

INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Экономист/пак-2 (450 dpi)/1991/01"
OUTPUT_DIR="/mnt/system/raw/economist/pak2/iter_1/out/1991/01"
DEBUG_DIR="/mnt/system/raw/economist/pak2/iter_1/debug/1991/01"

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
    --left-margin -310 \
    --right-margin -310 \
    --output-format tiff \
    --force-dpi=450 \
    --compensate-levels \
    --finger-dilate-px=120 \
    --max-asymmetric-dilation-ratio=1.6 \
    --finger-zone-light-increment=20,40 \
    --extra-erosion-px=165 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-px=18,72 \
    --bg-fill-method=nearest \
    --bg-fill-blur-px=16 \
    --remove-fingers

