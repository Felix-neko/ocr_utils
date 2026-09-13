#!/usr/bin/env bash
#
# Приёмка четвёртой версии детектора: полная сверка по паку, три версии и FineReader.
#
# Пишет в отдельную папку. Если есть кэш surya (run_layout_pack.sh), четвёртая версия
# получает разметку полос: передайте путь первым аргументом или оставьте пустым для CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARTIFACTS="reports/сверка детектора v4"
LAYOUT="${1:-}"
EXTRA=()
if [[ -n "$LAYOUT" ]]; then EXTRA=(--layout-dir "$LAYOUT"); fi

uv run python -m research.legacy.table_processing check-detector \
    --docx-dir "$DOCX_DIR" \
    --pdf-dir "$RECOGNIZED_PDF_DIR" \
    --sharpened-dir "$SHARPENED_DIR" \
    --db "$DB" \
    --pack "$PACK" \
    --out-dir "$ARTIFACTS" \
    --jobs "$JOBS" "${EXTRA[@]}" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
wait "$PID"

echo "отчёт: $ARTIFACTS/README.md"
