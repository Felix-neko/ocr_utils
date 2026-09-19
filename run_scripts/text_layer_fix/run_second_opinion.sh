#!/usr/bin/env bash
#
# Второе мнение surya (GPU, один процесс) по зонам выборки, где tesseract ненадёжен
# (чтение не принято или уверенность ниже 0.7): вырезки пересчитываются из PDF, ответы с
# фильтром выдумок пишутся обратно в JSON кэша; words.csv/zones.csv пересобираются.
#
# Читает: $TEXT_LAYER_PDF_DIR, $TEXT_LAYER_RUN_DIR/{sample.csv,cache}. Пишет: cache/, zones.csv.
# Идёт ~0.3 с на зону на GPU плюс загрузка модели: минуты на несколько сотен зон.
# Видеопамять одна на всех: не запускать одновременно с другими GPU-задачами.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

uv run python -m ocr_utils.text_layer_fix second-opinion \
    --pdf-dir "$TEXT_LAYER_PDF_DIR" \
    --out-dir "$TEXT_LAYER_RUN_DIR" \
    --batch 16 \
    "$@"
