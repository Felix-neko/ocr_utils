#!/usr/bin/env bash
#
# Разбор страниц пака-1 v4: текстовые блоки пересчитаны с межколонниками в режиме short
# (columns.GutterMode), всё остальное — объекты (растр из базы, таблицы, line art, формулы, повёрнутый
# текст), надписи, кандидаты с решениями DeepSeek и ориентация — ровно как в v3.
#
# * ПОЧЕМУ ОТДЕЛЬНЫЙ ХОД. Рабочая папка v3 (work/pages, crops, deepseek) удалена 2026-09-28 ~14:34, и
#   analyze-pack с --reuse-from v3 не пройдёт; пересчёт с нуля заново гонял бы DeepSeek и менял объекты.
#   Команда reblock-pack берёт всё, кроме текстовых блоков, из итоговых JSON v3 (pages/*.json).
# * ЧТО ИЗМЕНИЛОСЬ ПРОТИВ v3 (стенд research/gutter_crossing, reports/text_blocks_gutter_crossing.md):
#   наклонный межколонник больше не выпадает из запретов сцепки (запрет по отрезкам ломаной); короткий
#   двухколонный фрагмент за заголовком получает свой межколонник (продолжение длинного, кромка колонки
#   выровнена); межколонник не продлевается сквозь заголовок крупного набора; строка не режется по
#   межколоннику, если просвет там не шире её пробелов. На выборке P (71 полоса) и N (71): из 45
#   настоящих сращиваний исправлено 25, на N порчи нет.
# * ТЕКСТОВЫЕ БЛОКИ — по второй оси строки (--axis body), как в v3.
#
# ВРЕМЯ: ~3–4 с/полосу × 12 135 / 16 воркеров ≈ 50–60 мин (замер стенда 2026-09-28: ~200 полос/мин).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/page_layout/run_pack1_analysis_v4.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-/mnt/system/raw/mts/pack1_page_analysis_v4}"
SOURCE="${SOURCE:-/mnt/system/raw/mts/pack1_page_analysis_v3}"
JOBS="${JOBS:-16}"

uv run python -m ocr_utils.page_layout reblock-pack \
    --sharpened-dir "$SHARPENED_DIR" --from "$SOURCE" --out-dir "$OUT" --jobs "$JOBS" \
    --axis body --gutter-mode short "$@"
