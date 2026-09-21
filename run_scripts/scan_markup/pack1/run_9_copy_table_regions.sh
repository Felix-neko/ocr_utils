#!/usr/bin/env bash
#
# Шаг 9: автоматические находки таблиц и схем — в уточнённую базу, не дожидаясь разметчика.
#
# Уточнённая база (шаг 3) — снимок разметки CVAT, и таблицы попадут в неё только после
# того, как разметчик отсмотрит дозалитые шагом 2 рамки. Потребителям ниже по конвейеру
# они нужны раньше, поэтому переносятся как есть, с source=auto, — только виды table и
# line_art_schema плюс крупный штрих stroke_table и stroke_drawing (детектор
# line_art_detection), ручной растр в целевой базе не трогается. Следующий from-cvat заменит
# их уточнёнными.

set -euo pipefail
source "$(dirname "$0")/common.sh"
echo "Итог:     $DB_REVIEWED"

if [ -f "$DB_REVIEWED" ]; then
    cp -f "$DB_REVIEWED" "$DB_REVIEWED.bak"
fi

uv run python -m ocr_utils.scan_markup copy-regions \
    --db "$DB" \
    --out-db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --kinds table,line_art_schema,stroke_table,stroke_drawing
