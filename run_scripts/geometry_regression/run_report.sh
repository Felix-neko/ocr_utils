#!/usr/bin/env bash
#
# Сводка и картинки «было | стало» по готовому metrics.csv — без пересчёта, можно гонять с
# любыми порогами: ./run_report.sh --thr line_slope_abs_delta_max=0.7 --min-score 1.3
#
# Читает: $GEOMETRY_RUN_DIR/{metrics.csv,cache/}. Пишет: metrics_flagged.csv,
# report.md и pairs/<вердикт>/<год>/*.jpg (по умолчанию все страницы со score ≥ 1, третья панель — поле).

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
# Абсолютный путь до sourcing: common.sh пака (через common.sh направления) сам делает cd в корень
# репо, после чего относительный dirname указывал бы мимо.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

JOBS=8  # рендер пар: упирается в диск меньше, чем в CPU, но картинок сотни, не тысячи

uv run python -m research.geometry_regression report \
    --out-dir "$GEOMETRY_RUN_DIR" \
    --labels "$GEOMETRY_LABELS" \
    --md-report "$GEOMETRY_RUN_DIR/report.md" \
    --pairs-dir "$GEOMETRY_RUN_DIR/pairs" \
    --arrows \
    --jobs "$JOBS" \
    "$@"
