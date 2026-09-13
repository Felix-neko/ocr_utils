#!/usr/bin/env bash
#
# Приёмка третьей версии детектора: полная сверка по паку, все три множества сразу.
#
# Пишет в ОТДЕЛЬНУЮ папку, а не в «проверка детектора таблиц»: там лежит раскладка по
# диагнозам, сделанная руками, и перезаписывать её нельзя.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARTIFACTS="reports/сверка детектора v3"

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

echo "отчёт: $ARTIFACTS/README.md"
