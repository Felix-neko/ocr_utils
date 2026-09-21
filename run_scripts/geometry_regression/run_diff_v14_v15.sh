#!/usr/bin/env bash
#
# Разница вердиктов v14 (ядро) → v15 (стенд): CSV смен и пары «было | стало» по видам смен
# (ok_to_bad, bad_to_ok, mixed_to_bad, …), рамка виновника — от движка с худшим вердиктом.
#
# Читает: …/pack1_v14/metrics.csv и …/pack1_v15/{metrics.csv,cache/}. Пишет: $OUT/changed_v14_v15.csv
# и $OUT/changed/<смена>/*.jpg; OUT по умолчанию — папка просмотра v15 в ~/Projects/mts_markup.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-$HOME/Projects/mts_markup/pack1_geometry_v15_review}"

uv run python -m research.geometry_regression diff \
    --old-dir "$GEOMETRY_REGRESSION_ROOT/pack1_v14" \
    --new-dir "${GEOMETRY_RUN_DIR_V15:-$GEOMETRY_REGRESSION_ROOT/pack1_v15}" \
    --old-engine core --new-engine v15 \
    --labels "$GEOMETRY_LABELS" \
    --csv-out "$OUT/changed_v14_v15.csv" \
    --pairs-dir "$OUT/changed" \
    --jobs 12 \
    "$@"
