#!/usr/bin/env bash
#
# Смены вердикта «bad / не bad» v17 → v18 при одних порогах: оверлеи нового прогона по папкам перехода и вида страницы
# (line art по разбору v6 без коррекции / остальные), diff.csv и diff.md с эталоном.
#
# Читает: кэши $QUALITY_RUN_DIR (v17) и $QUALITY_RUN_DIR_V18, оба PDF, разбор v6, эталон $GEOMETRY_LABELS.
# Пишет: $DIFF_DIR/{<было>_to_<стало>/{lineart,other}/*.jpg, diff.csv, diff.md}.
# Время: секунды на сравнение, отрисовка — секунда на смену (рендер двух PDF).
# Пороги перекрыть: EXTRA="--thr lineart_shape_mm=2.0".

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

EXTRA="${EXTRA:-}"
# По умолчанию — боевой прогон v18 из common.sh пака ($GEOMETRY_V18_DIR). pack1_v18 … pack1_v18f — прогоны
# отладки стенда (кэш той же версии, но старого кода и разбора v6): не использовать.
QUALITY_RUN_DIR_V18="${QUALITY_RUN_DIR_V18:-$GEOMETRY_V18_DIR}"
DIFF_DIR="${DIFF_DIR:-$MARKUP_ROOT/pack1_geometry_v18_lineart_diff_f}"

# shellcheck disable=SC2086
uv run python -m research.geometry_quality diff --old-run "$QUALITY_RUN_DIR" --new-run "$QUALITY_RUN_DIR_V18" \
    --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --out-dir "$DIFF_DIR" --layout-root "$QUALITY_LAYOUT_ROOT" \
    --labels "$GEOMETRY_LABELS" $EXTRA
