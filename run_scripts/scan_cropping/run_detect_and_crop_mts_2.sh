#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «1972 готово» (620 JPG в 12 подпапках):
# детекция разворота → поворот → кроп с припусками, апскейл ×2, компенсация
# уровней, сохранение в PNG. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
# РАЗМЕРЫ — В ММ БУМАГИ (2026-10-19): прежние пиксели пересчитаны при --dpi 300 (то же, что --force-dpi:
# при этом разрешении мм дают ровно прежние пиксели). Неизменяемые из CLI умолчания v1 (эрозия книги,
# контекст LaMa, растушёвка) названы при 450 dpi — у пака 300 dpi они теперь на треть мельче в пикселях,
# как и должно быть на бумаге (прежде были одинаковыми при 300 и 450).

set -euo pipefail

# Скрипт лежит в run_scripts/<подсистема>/, а пути внутри отсчитываются от корня
# репозитория — поднимаемся на два уровня.
cd "$(dirname "$0")/../.."

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе"
INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/1977 - 1978-03 (300 dpi, лёгкие расфокусы)"
OUTPUT_DIR="/mnt/hotstore/scan_processing/mts/77_78_defocus/cropped"
DEBUG_DIR="/mnt/hotstore/scan_processing/mts/77_78_defocus/debug"

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
    --left-margin-mm -19.4733 \
    --right-margin-mm -19.4733 \
    --output-format tiff \
    --force-dpi=300 \
    --dpi=300 \
    --compensate-levels \
    --finger-dilate-mm=5.08 \
    --max-asymmetric-dilation-ratio=2 \
    --finger-zone-light-increment=20,20 \
    --extra-erosion-mm=9.3133 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-mm=1.016,4.064 \
    --bg-fill-method=nearest \
    --bg-fill-blur-mm=1.354667 \
    --log-level=INFO \
    --remove-fingers \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-mm=2.709333 \
    --crop-fill-fade=0.0 \

