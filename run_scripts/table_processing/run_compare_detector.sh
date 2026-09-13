#!/usr/bin/env bash
#
# Ежедневный цикл детектора: прогон версий по размеченным полосам (190 полос, две раскладки).
#
# Идёт секунды, поэтому гоняется после каждой правки порога. Полная сверка по паку
# (run_check_detector_v4.sh) — это уже приёмка релиза, она стоит четверть часа.
#
# Первый аргумент --rebuild-labels пересобирает разметку из папок (нужно, когда папок стало
# больше); остальные аргументы уходят команде как есть, например
# --versions v2,v3,v4 или --layout-dir "$OUT_DIR/layout_surya".
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "${1:-}" == "--rebuild-labels" ]]; then
    uv run python -m research.legacy.table_processing labels-from-folders
    shift
fi

uv run python -m research.legacy.table_processing compare-detector \
    --sharpened-dir "$SHARPENED_DIR" \
    --jobs "$JOBS" "$@" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 2; done
wait "$PID"

echo "оверлеи по диагнозам: reports/детектор v4/оверлеи_v4/"
