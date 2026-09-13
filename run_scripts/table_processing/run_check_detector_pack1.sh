#!/usr/bin/env bash
#
# Сверка детектора таблиц по всему паку: старая версия, новая и таблицы самого FineReader.
#
# Один проход по 12 135 полосам (около 6 минут на 12 воркерах) считает сразу три множества:
# таблицы, которые FineReader создал в DOCX, находки детектора без проверки и с проверкой.
# Пишет debug-наборы картинок и отчёт со сравнением.
#
# Картинок получается около 2400 штук, порядка 190 МБ; папки с ними в .gitignore.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARTIFACTS="reports/проверка детектора таблиц"

uv run python -m research.legacy.table_processing check-detector \
    --docx-dir "$DOCX_DIR" \
    --pdf-dir "$RECOGNIZED_PDF_DIR" \
    --sharpened-dir "$SHARPENED_DIR" \
    --db "$DB" \
    --pack "$PACK" \
    --out-dir "$ARTIFACTS" \
    --jobs "$JOBS" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
wait "$PID"

echo "отчёт:  $ARTIFACTS/README.md"
echo "листы:  $ARTIFACTS/листы"
