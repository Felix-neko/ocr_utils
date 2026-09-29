#!/usr/bin/env bash
#
# Разбор страниц пака-1 v5: полный пересчёт со стадии кандидатов — детектор таблиц v4 (TABLES_VERSION 4:
# ложные линейки-сироты отсеиваются по признакам трассы, tables.loose_filter; проверка находки считает только
# ячейки от 2 мм) и текущий детектор текстовых блоков (запреты сращивания строк разного набора —
# text_blocks/typeset.py, reports/text_blocks_heading_merge.md; защита сторон CRAFT + pero,
# reports/edge_guard.md; перескоки строк, reports/text_blocks_row_jumps.md).
#
# * РАСТР — из $DB_REVIEWED, как в v3/v4 (детектор растра не запускается).
# * ТАБЛИЦЫ, LINE ART, ФОРМУЛЫ, ПОВЁРНУТЫЙ ТЕКСТ — заново по всем полосам (кэш surya — только чтение).
#   --reuse-from не годится: он принёс бы таблицы и линейки старого детектора (рабочая папка v3 — от
#   2026-09-28 18:42, оба изменения детектора таблиц позже).
# * DEEPSEEK — заново по кандидатам line art (в v3 их 1262): номера кандидатов после смены таблиц могут
#   сдвинуться, и ответы прошлого прогона легли бы на чужие вырезки.
# * ТЕКСТОВЫЕ БЛОКИ — по второй оси (--axis body), межколонники short, второй проход защиты сторон
#   (--edge-craft, GPU) — по умолчанию.
#
# МЕСТО: /mnt/system заполнен (2026-09-29, свободно 17 ГБ) — выход временно в $MARKUP_ROOT (домашний SSD).
#
# ВРЕМЯ (оценка): кандидаты ~12 135 полос / 16 воркеров ≈ 30–40 мин (v3: 554 полосы за 1–2 мин), DeepSeek —
# минуты с пуском движка, текстовые блоки ~3–4 с/полосу ≈ 50–60 мин, защита сторон — ещё минуты по полосам
# с выступами.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/page_layout/run_pack1_analysis_v5.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-$MARKUP_ROOT/pack1_page_analysis_v5}"
JOBS="${JOBS:-16}"

uv run python -m ocr_utils.page_layout analyze-pack \
    --sharpened-dir "$SHARPENED_DIR" --cache "$LAYOUT_CACHE_DIR" --out-dir "$OUT" --jobs "$JOBS" --no-orientation \
    --raster-db "$DB_REVIEWED" --pack-name "$PACK_NAME" --axis body --gutter-mode short "$@"
