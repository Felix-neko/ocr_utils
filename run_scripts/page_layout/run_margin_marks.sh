#!/usr/bin/env bash
#
# Стенд «пометки на полях, ложно принятые за рисунок» (research/margin_marks, reports/margin_marks.md): признаки
# кандидатов-рисунков и «неясно» по разбору v6 обоих PDF FineReader и пересчёт решения line art всех кандидатов без
# GPU (по готовому выводу DeepSeek) с правилом «пометка».
#
# Читает: $MARKUP_ROOT/pack1_page_analysis_v6_fr/{nogeo,geo} (pages, work), PDF $GEO_PDF_DIR, $NOGEO_PDF_DIR.
# Пишет: $MARKUP_ROOT/margin_marks/{features.csv,replay.csv}.
# Время: признаки ~1.5 мин, пересчёт ~4 мин (16 воркеров, CPU).

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

JOBS="${JOBS:-16}"
LAYOUT="$MARKUP_ROOT/pack1_page_analysis_v6_fr"
OUT="$MARKUP_ROOT/margin_marks"

uv run python -m research.margin_marks features --layout-root "$LAYOUT" --out-dir "$OUT" --jobs "$JOBS"
uv run python -m research.margin_marks replay --layout-root "$LAYOUT" --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" \
    --out-dir "$OUT" --jobs "$JOBS"
