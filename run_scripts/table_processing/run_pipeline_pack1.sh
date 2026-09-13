#!/usr/bin/env bash
#
# Шаг 2. По каждой вырезанной таблице: сетка ячеек, поиск боковых, распознавание, замена
# бокового текста прямым. Результат — пары «было-стало» в $OUT_DIR/pairs и листы в sheets.
#
# Движок распознавания по умолчанию tesseract: на 37 размеченных ячейках у него CER 0.036
# при 0.05 с на ячейку против 0.085 и 2.24 с у PaddleOCR и медианного нуля при среднем 3.97
# у surya (та дописывает к короткой надписи выдуманный текст). Замеры — в README пакета.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

uv run python -m research.legacy.table_processing run \
    --out-dir "$OUT_DIR" \
    --engine tesseract \
    --dpi 300 \
    --jobs "$JOBS" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
wait "$PID"

uv run python -m research.legacy.table_processing pairs --out-dir "$OUT_DIR" --per-sheet 4
