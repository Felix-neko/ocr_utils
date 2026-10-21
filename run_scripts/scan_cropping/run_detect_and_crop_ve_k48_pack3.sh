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
#INPUT_DIR="/mnt/hotstore/scan_processing/плохие сканы  ВЭ/06"
#INPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/in"
#INPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/in/1989/06 проверить зональный пересвет"
#OUTPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/test_896_tiff_9/out"
#DEBUG_DIR="/mnt/hotstore/scan_processing/ve_80s/test_896_tiff_9/debug"

#INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Вопросы экономики/в работе/пак-2 (450 DPI)"
#OUTPUT_DIR="/mnt/hotstore/scan_processing/ve_test/pack2/iter_1/cropped"
#DEBUG_DIR="/mnt/hotstore/scan_processing/ve_test/pack2/iter_1/debug"

INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Вопросы экономики/в работе/пак-3 (450 DPI улучшили фокус)"
OUTPUT_DIR="/mnt/hotstore/scan_processing/ve/pack_3/v_1/cropped"
DEBUG_DIR="/mnt/hotstore/scan_processing/ve/pack_3/v_1/debug"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.scan_cropping.v1 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin-mm -10.7244 \
    --bottom-margin-mm -15.24 \
    --left-margin-mm -16.9333 \
    --right-margin-mm -16.9333 \
    --output-format tiff \
    --force-dpi=450 \
    --dpi=450 \
    --compensate-levels \
    --finger-dilate-mm=6.2089 \
    --max-asymmetric-dilation-ratio=1.6 \
    --finger-zone-light-increment=20,20 \
    --extra-erosion-mm=4.5156 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-mm=1.016,4.064 \
    --bg-fill-method=nearest \
    --bg-fill-blur-mm=0.903111 \
    --log-level=INFO \
    --remove-fingers \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-mm=2.709333 \
    --crop-fill-fade=0.0 \

