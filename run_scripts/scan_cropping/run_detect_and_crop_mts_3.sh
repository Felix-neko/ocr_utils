#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
# РАЗМЕРЫ — В ММ БУМАГИ (2026-10-19): прежние пиксели пересчитаны при --dpi 450 (то же, что --force-dpi:
# при этом разрешении мм дают ровно прежние пиксели). Неизменяемые из CLI умолчания v1 (эрозия книги,
# контекст LaMa, растушёвка) названы при 450 dpi.

set -euo pipefail

# Скрипт лежит в run_scripts/<подсистема>/, а пути внутри отсчитываются от корня
# репозитория — поднимаемся на два уровня.
cd "$(dirname "$0")/../.."

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе"
INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-5 (1985 и далее + немножко пересканов) (450 DPI)/"
OUTPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/нарезка сканов/пак-5 (1985 и далее + немножко пересканов) (450 DPI)/v4/cropped"
DEBUG_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/нарезка сканов/пак-5 (1985 и далее + немножко пересканов) (450 DPI)/v4/debug"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.scan_cropping.v1 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin-mm -10.16 \
    --bottom-margin-mm -15.24 \
    --left-margin-mm -20.32 \
    --right-margin-mm -20.32 \
    --output-format tiff \
    --force-dpi=450 \
    --dpi=450 \
    --compensate-levels \
    --finger-dilate-mm=5.08 \
    --max-asymmetric-dilation-ratio=2. \
    --finger-zone-light-increment=20,20 \
    --extra-erosion-mm=6.7733 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-mm=1.016,4.064 \
    --bg-fill-method=nearest \
    --bg-fill-blur-mm=0.903111 \
    --log-level=INFO \
    --remove-fingers \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-mm=1.806222 \
    --crop-fill-fade=0.0 \

