#!/usr/bin/env bash
#
# Запуск detect_and_crop: детекция разворота (YOLO-World + SAM) → правильный поворот
# вокруг центра тяжести → вырез crop-зоны.
#   - выпрямленные кропы          → OUTPUT_DIR (имя файла сохраняется)
#   - отладочные оверлеи          → DEBUG_DIR (граница / min-bbox / crop-зона)
#
# Все аргументы опциональны (значения по умолчанию — как в типовом прогоне):
#   ./run_detect_and_crop.sh [INPUT_DIR] [OUTPUT_DIR] [DEBUG_DIR] [DPI] [X_MARGINS_MM] [Y_MARGINS_MM]
#
# Припуски — в мм бумаги (слева и справа X, сверху и снизу Y; отрицательные сужают зону), в пиксели кадра их
# переводит --dpi. Прежние умолчания -300 / -120 px при 300 dpi — это -25.4 / -10.16 мм. До 2026-10-19 скрипт
# передавал несуществующие --x-margins / --y-margins и не работал.
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
DPI="${4:-300}"
X_MARGINS_MM="${5:--25.4}"
Y_MARGINS_MM="${6:--10.16}"

echo "detect_and_crop:"
echo "  input     = $INPUT_DIR"
echo "  output    = $OUTPUT_DIR"
echo "  debug     = $DEBUG_DIR"
echo "  dpi       = $DPI | x-margins = $X_MARGINS_MM мм | y-margins = $Y_MARGINS_MM мм"

uv run python -m ocr_utils.scan_cropping.v1 \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --debug-dir  "$DEBUG_DIR" \
    --dpi "$DPI" \
    --left-margin-mm "$X_MARGINS_MM" \
    --right-margin-mm "$X_MARGINS_MM" \
    --top-margin-mm "$Y_MARGINS_MM" \
    --bottom-margin-mm "$Y_MARGINS_MM"
