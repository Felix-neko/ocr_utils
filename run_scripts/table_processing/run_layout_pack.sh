#!/usr/bin/env bash
#
# Surya layout по всему паку: pickle на полосу в $OUT_DIR/layout_surya/{год}/{выпуск}/{основа}.pkl.
#
# GPU-задача: идёт в одном процессе, воркеры только читают JPEG. Около секунды на полосу,
# по паку — три с половиной часа; уже посчитанные полосы пропускаются, так что прерванный
# прогон можно просто перезапустить.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

uv run python -m research.legacy.table_processing layout-pack \
    --sharpened-dir "$SHARPENED_DIR" \
    --out-dir "$OUT_DIR/layout_surya" \
    --mode "${1:-all}" \
    --jobs "$JOBS" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
wait "$PID"
