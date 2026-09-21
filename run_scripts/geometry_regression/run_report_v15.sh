#!/usr/bin/env bash
#
# Сводка и картинки «было | стало» по прогону v15 — без пересчёта, с порогами движка v15:
# ./run_report_v15.sh --thr field_shear_p90_deg=0.9 --min-score 0 --pairs-dir ~/Projects/mts_markup/pack1_geometry_v15_review/pairs
#
# Читает: $GEOMETRY_RUN_DIR (…/pack1_v15)/{metrics.csv,cache/}. Пишет: metrics_flagged.csv, report.md,
# pairs/<вердикт>/<год>/*.jpg (по умолчанию — все страницы со score ≥ 1, без панели поля).

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

GEOMETRY_RUN_DIR="${GEOMETRY_RUN_DIR_V15:-$GEOMETRY_REGRESSION_ROOT/pack1_v15}"
JOBS=12

uv run python -m research.geometry_regression report \
    --engine v15 \
    --out-dir "$GEOMETRY_RUN_DIR" \
    --labels "$GEOMETRY_LABELS" \
    --md-report "$GEOMETRY_RUN_DIR/report.md" \
    --pairs-dir "$GEOMETRY_RUN_DIR/pairs" \
    --no-arrows \
    --jobs "$JOBS" \
    "$@"
