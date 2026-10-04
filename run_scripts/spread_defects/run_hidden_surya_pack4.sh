#!/usr/bin/env bash
#
# Буквы под переплётом, пак-4 ПХ: surya (полное разрешение, полосы у сгиба) по кандидатам геометрического предотбора
# (25-й процентиль отступа концов строк от сгиба < 1.0 высоты буквы, 566 кадров) + 150 случайных кадров для базы;
# словарь pymorphy3 по крайним словам с учётом переносов (research/spread_defects/truncated_words.py).
#
# Читает: $SPREAD_DEFECTS_ROOT/hidden_candidates.csv, кадры $PH_PACK4_ROOT.
# Пишет: $SPREAD_DEFECTS_ROOT/{ocr_fold/…json (кэш surya), truncated.csv}.
# Время: ~8–12 с на кадр (замер на ЭСХ 1979/12, 2026-10-04), ~2 ч. GPU — один процесс, не в пуле.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

# Подготовка вырезок — CPU-пул; surya — один на GPU.
JOBS="${JOBS:-6}"

uv run python -m research.spread_defects truncated --root "$PH_PACK4_ROOT" --out-root "$SPREAD_DEFECTS_ROOT" \
    --keys "$SPREAD_DEFECTS_ROOT/hidden_candidates.csv" --jobs "$JOBS"
