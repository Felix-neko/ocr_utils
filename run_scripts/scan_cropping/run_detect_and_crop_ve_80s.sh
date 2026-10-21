#!/usr/bin/env bash
#
# Прогон detect_and_crop на партии «Вопросы экономики» (80-е):
# детекция разворота → кроп с припусками БЕЗ поворота (--crop-mode=pixel-exact:
# пиксели не пересэмплируются, выпрямление — снаружи, в ScanTailor) → компенсация
# уровней, сохранение в TIFF. Debug-оверлеи (JPEG) — рядом, в отдельной папке.
#
# Зона вне книги заполняется продлением края по нормали к сторонам crop-зоны
# (--crop-fill-method=replicate): линия корешка продолжается прямо, и ScanTailor
# находит по ней разрез разворота. --bg-fill-* в режиме pixel-exact не участвуют
# (нужны только для --crop-mode=rotate).
#
# Размытие заливки (--crop-fill-blur-mm): у самого края книги его нет, к дальнему
# краю кадра нарастает до указанной σ — прячет полосатость продления. ЧТОБЫ ВЫКЛЮЧИТЬ
# — поставить 0: --crop-fill-blur-mm=0. Держать в уме: размытие смазывает и саму
# продолженную линию корешка, так что если ScanTailor начнёт терять разрез —
# выключать надо в первую очередь его.
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
#INPUT_DIR="/mnt/hotstore/scan_processing/плохие сканы  ВЭ/06"
#INPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/in"
#INPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/in/1989/06 проверить зональный пересвет"
#OUTPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/test_896_tiff_9/out"
#DEBUG_DIR="/mnt/hotstore/scan_processing/ve_80s/test_896_tiff_9/debug"

INPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/in"
OUTPUT_DIR="/mnt/hotstore/scan_processing/ve_80s/iter_5/cropped"
DEBUG_DIR="/mnt/hotstore/scan_processing/ve_80s/iter_5/debug"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.scan_cropping.v1 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin-mm -5.08 \
    --bottom-margin-mm -15.24 \
    --left-margin-mm -16.9333 \
    --right-margin-mm -16.9333 \
    --output-format tiff \
    --force-dpi=300 \
    --dpi=300 \
    --compensate-levels \
    --finger-dilate-mm=6.7733 \
    --max-asymmetric-dilation-ratio=1.6 \
    --finger-zone-light-increment=20,20 \
    --extra-erosion-mm=9.3133 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-mm=1.016,4.064 \
    --bg-fill-method=nearest \
    --bg-fill-blur-mm=1.354667 \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-mm=2.709333 \
    --crop-fill-fade=0.0 \
    --log-level=INFO \
    --remove-fingers

