#!/usr/bin/env bash
#
# Разбор страниц пака-1 v3: как run_pack1_analysis.sh, но растр — из базы после ревью в CVAT, line art —
# только там, где растр изменился против v2, блоки текста — по второй оси строки.
#
# * РАСТР: детектор растра НЕ запускается. Цветной и серый растр и цветной текст берутся из
#   $DB_REVIEWED (rect_regions, все source='cvat'). Печатей нет ни в каком виде: ни stamp_suspect, ни
#   масок library_stamp (решение пользователя 2026-09-28).
# * LINE ART (и таблицы, формулы surya, повёрнутый текст, DeepSeek): пересчитываются только на полосах,
#   где растр в базе отличается от растра v2 хоть чем-то — числом рамок, видом, координатами (точное
#   сравнение). Замер 2026-09-28: 554 полосы из 12 135 (из них часть — полосы, где в v2 были печати
#   детектора). Остальные полосы берутся из v2 целиком: запись стадии кандидатов, вырезки и вывод
#   DeepSeek. Список пересчитанных — $OUT/work/raster_changed.txt.
# * ТЕКСТОВЫЕ БЛОКИ — по всем полосам, ряды и блоки собираются по ВТОРОЙ оси строки (--axis body,
#   text_blocks/baseline_axis.py), она же на оверлее. На оверлее ещё дополнительные линии левой и правой
#   стороны блока (sides.filled_side, выравнивание robust): невыровненные концы выброшены, выбросы в
#   середине заменены PCHIP-заплаткой (пунктир).
#
# ДЕТЕКТОР ТАБЛИЦ ИЗМЕНИЛСЯ (2026-09-28): куски линеек зашиваются через разрывы до 2 мм, а не 3 мм;
# линейки-сироты — только на чистой бумаге; порог «пустого бланка» 0.43. Поэтому --reuse-from v2 для
# НОВОГО прогона не годится: он принёс бы таблицы и линейки старого детектора. Текущий
# pack1_page_analysis_v3 пересчитан со стадии кандидатов (ai_slop/recompute_v3_candidates.py,
# прежние рабочие файлы — work/*_before_rules_v2). Для нового прогона — без --reuse-from.
#
# ВРЕМЯ (оценка): кандидаты по 554 полосам — 1–2 мин, DeepSeek по их кандидатам — минуты вместе с пуском
# движка, текстовые блоки ~3 с/полосу × 12 135 / 16 воркеров ≈ 40–50 мин.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/page_layout/run_pack1_analysis_v3.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-/mnt/hotstore/scan_processing/mts/pack1_page_analysis_v3}"
# Прошлый полный разбор (2026-09-27). Для --reuse-from — только при том же детекторе таблиц (см. выше).
PREVIOUS="${PREVIOUS:-/mnt/hotstore/scan_processing/mts/pack1_page_analysis_v2}"
JOBS=16

uv run python -m ocr_utils.page_layout analyze-pack \
    --sharpened-dir "$SHARPENED_DIR" --cache "$LAYOUT_CACHE_DIR" --out-dir "$OUT" --jobs "$JOBS" --no-orientation \
    --raster-db "$DB_REVIEWED" --pack-name "$PACK_NAME" --axis body "$@"
