#!/usr/bin/env bash
#
# Ложные линейки-сироты детектора таблиц: признаки всех сирот пака-1 v4, правила отбраковки на разметке,
# пересчёт текстовых блоков затронутых полос «было | стало» (стенд research/loose_rules,
# reports/loose_rules_false.md). Боевой код не меняется: сироты берутся из итоговых JSON разбора v4.
#
# Что делает: features — признаки каждой сироты (доля трассы на буквах, самый длинный пробег «не букв»,
#   краска вплотную с обеих сторон, ядро штриха поперёк); evaluate — причины отбраковки, сводка на
#   разметке research/loose_rules/sets/labels.csv и по паку; compare — текстовые блоки с прежними
#   сиротами и без отброшенных, склейки на каждую полосу с отброшенными.
# Читает: $PACK_DIR/pages/*.json (разбор v4), $SHARPENED_DIR (заострённые сканы).
# Пишет: $OUT/features.csv, $OUT/eval/verdicts.csv, $OUT/compare/{compare.csv,pages/*.jpg} — на SSD, вне git.
# Сколько идёт (2026-09-29, 16 воркеров): features ~4 мин (6 023 полосы с сиротами, 9 334 сироты),
#   evaluate — секунды, compare ~15 мин (949 полос, по два разбора текстовых блоков на полосу).
# Числа (2026-09-29): на разметке 289 сирот (71 настоящая) — полнота по ложным 0.91, потеряна 1 спорная
#   (тонкая линия у кромки листа); по паку отброшено 1 864 из 9 334: буквы 1 093, формулы 535,
#   кромка 153, короткий пробег 83.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_loose_rules.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

PACK_DIR="${PACK_DIR:-/mnt/system/raw/mts/pack1_page_analysis_v4}"
OUT="${OUT:-/mnt/system/raw/mts/research_loose_rules}"
JOBS="${JOBS:-16}"

uv run python -m research.loose_rules features \
    --pack-dir "$PACK_DIR" --sharpened-dir "$SHARPENED_DIR" --out-dir "$OUT" --jobs "$JOBS"
uv run python -m research.loose_rules evaluate --features-csv "$OUT/features.csv" --out-dir "$OUT/eval"
uv run python -m research.loose_rules compare \
    --pack-dir "$PACK_DIR" --sharpened-dir "$SHARPENED_DIR" --verdicts "$OUT/eval/verdicts.csv" \
    --out-dir "$OUT/compare" --jobs "$JOBS"
