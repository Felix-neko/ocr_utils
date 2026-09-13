#!/usr/bin/env bash
#
# Шаг 1. Найти в выгрузке DOCX таблицы, которые FineReader превратил в мешанину, привязать
# их к страницам распознанных PDF и вырезать из сканов в 600 dpi.
#
# Порог счёта мешанины 0.20 замерен на 1966/01: шесть известных испорченных таблиц набирают
# 0.25-0.64, чистые — 0.01-0.19. Прогон по 103 выпускам занимает около трёх минут и даёт
# порядка 230 вырезок.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

uv run python -m research.legacy.table_processing mine \
    --docx-dir "$DOCX_DIR" \
    --pdf-dir "$RECOGNIZED_PDF_DIR" \
    --sharpened-dir "$SHARPENED_DIR" \
    --db "$DB" \
    --pack "$PACK" \
    --out-dir "$OUT_DIR" \
    --min-rank 0.20 \
    --jobs "$JOBS" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
wait "$PID"

# Контактные листы находок — их смотрят глазами, прежде чем идти дальше.
uv run python -m research.legacy.table_processing sheet --out-dir "$OUT_DIR" --top 240
