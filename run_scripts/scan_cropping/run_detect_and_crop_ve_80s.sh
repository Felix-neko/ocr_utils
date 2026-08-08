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
# Размытие заливки (--crop-fill-blur-px): у самого края книги его нет, к дальнему
# краю кадра нарастает до указанной σ — прячет полосатость продления. ЧТОБЫ ВЫКЛЮЧИТЬ
# — поставить 0: --crop-fill-blur-px=0. Держать в уме: размытие смазывает и саму
# продолженную линию корешка, так что если ScanTailor начнёт терять разрез —
# выключать надо в первую очередь его.
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

INPUT_DIR="/mnt/system/raw/ve_80s/in"
OUTPUT_DIR="/mnt/system/raw/ve_80s/iter_5/cropped"
DEBUG_DIR="/mnt/system/raw/ve_80s/iter_5/debug"

echo "detect_and_crop:"
echo "  input  = $INPUT_DIR"
echo "  output = $OUTPUT_DIR"
echo "  debug  = $DEBUG_DIR"

uv run python -m ocr_utils.scan_cropping \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir "$DEBUG_DIR" \
    --recursive \
    --top-margin -60 \
    --bottom-margin -180 \
    --left-margin -200 \
    --right-margin -200 \
    --output-format tiff \
    --force-dpi=300 \
    --compensate-levels \
    --finger-dilate-px=80 \
    --max-asymmetric-dilation-ratio=1.6 \
    --finger-zone-light-increment=20,20 \
    --extra-erosion-px=110 \
    --protect-text-layout \
    --text-protect-mode=copy-back-layout-zones \
    --layout-pad-px=12,48 \
    --bg-fill-method=nearest \
    --bg-fill-blur-px=16 \
    --crop-mode=pixel-exact \
    --crop-fill-method=replicate \
    --crop-fill-blur-px=32 \
    --crop-fill-fade=0.0 \
    --log-level=INFO \
    --remove-fingers

