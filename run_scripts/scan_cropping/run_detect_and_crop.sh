#!/usr/bin/env bash
#
# Запуск detect_and_crop: детекция разворота (YOLO-World + SAM) → правильный поворот
# вокруг центра тяжести → вырез crop-зоны.
#   - выпрямленные кропы          → OUTPUT_DIR (имя файла сохраняется)
#   - отладочные оверлеи          → DEBUG_DIR (граница / min-bbox / crop-зона)
#
# Все аргументы опциональны (значения по умолчанию — как в типовом прогоне):
#   ./run_detect_and_crop.sh [INPUT_DIR] [OUTPUT_DIR] [DEBUG_DIR] [X_MARGINS] [Y_MARGINS]
#
# Пример «по умолчанию» (то же, что зашито):
#   ./run_detect_and_crop.sh
#
set -euo pipefail

# Работаем из корня проекта (папка, где лежит этот скрипт)
# Скрипт лежит в run_scripts/<подсистема>/, а пути внутри отсчитываются от корня
# репозитория — поднимаемся на два уровня.
cd "$(dirname "$0")/../.."

INPUT_DIR="${1:-ocr_utils/finger_removal/inpainted_lama}"
OUTPUT_DIR="${2:-ocr_utils/finger_removal/inpainted_lama_cropped}"
DEBUG_DIR="${3:-ocr_utils/finger_removal/inpainted_lama_cropped_debug}"
X_MARGINS="${4:--300}"
Y_MARGINS="${5:--120}"

echo "detect_and_crop:"
echo "  input     = $INPUT_DIR"
echo "  output    = $OUTPUT_DIR"
echo "  debug     = $DEBUG_DIR"
echo "  x-margins = $X_MARGINS | y-margins = $Y_MARGINS"

uv run python -m ocr_utils.scan_cropping \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir  "$DEBUG_DIR" \
    --x-margins  "$X_MARGINS" \
    --y-margins  "$Y_MARGINS"
