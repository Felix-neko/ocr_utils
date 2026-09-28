#!/usr/bin/env bash
#
# Стенд «оси, сошедшиеся на отточии» (research/leader_merge): ряд таблицы «подпись . . . . число» распадался на
# две оси, обе доходили до одних и тех же точек отточия и не срастались (1966/03 IMG_0131_2R). Шаг
# segment.join_on_leaders сращивает такие пары; стенд проверяет, что он чинит проблемные полосы (P) и не
# трогает беспроблемные (N).
#
# Вход — тот же, что в разборе пака v3 (объекты и запреты из pack1_page_analysis_v3, вторая ось строки,
# межколонники legacy). Источник осей для отбора — полный прогон research/gutter_crossing (legacy).
#
# Шаги (STEP=all — все подряд):
#   select  — P (пары осей на общем глифе) и N по слоям (отточия, не только текст, колонки, заголовки,
#             обычные) → $OUT/problem.txt, normal.txt, scan.csv;
#   before  — P ∪ N без сращивания (--no-join) с оверлеями → $OUT/before;
#   after   — P ∪ N со сращиванием → $OUT/after;
#   compare — склейки «было | стало» и CSV → $OUT/compare_P, $OUT/compare_N.
#
# Время: отбор — минута, прогон 100 полос — пара минут на 16 воркерах.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_leader_merge.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 10; done

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scan_markup/pack1/common.sh"

OUT="${OUT:-/mnt/system/raw/mts/pack1_leader_merge}"
PACK="${PACK:-/mnt/system/raw/mts/pack1_page_analysis_v3}"
SOURCE="${SOURCE:-/mnt/system/raw/mts/pack1_gutter_crossing/legacy}"
JOBS="${JOBS:-16}"
STEP="${STEP:-all}"
# P: 20 полос (с разбросом по выпускам), N: 80 (по 16 на слой). Обязательные в P: пример пользователя и
# 1971/10 с.93 (IMG_0050_1L) — на ней когда-то запретили сцепку «подпись + число правой графы».
PROBLEM=20
NORMAL=80

stand() {
    uv run python -m research.leader_merge "$@"
}

if [[ "$STEP" == all || "$STEP" == select ]]; then
    stand select --source "$SOURCE" --pack-dir "$PACK" --out-dir "$OUT" --problem "$PROBLEM" --normal "$NORMAL" \
        --leaders-from "$OUT/classify" --must 1966/03/IMG_0131_2R --must 1971/10/IMG_0050_1L --jobs "$JOBS"
    cat "$OUT/problem.txt" "$OUT/normal.txt" > "$OUT/sets.txt"
fi
if [[ "$STEP" == all || "$STEP" == before ]]; then
    stand run --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --out-dir "$OUT/before" --keys "$OUT/sets.txt" \
        --no-join --jobs "$JOBS"
fi
if [[ "$STEP" == all || "$STEP" == after ]]; then
    stand run --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --out-dir "$OUT/after" --keys "$OUT/sets.txt" \
        --join --jobs "$JOBS"
fi
if [[ "$STEP" == all || "$STEP" == compare ]]; then
    stand compare --before "$OUT/before" --after "$OUT/after" --keys "$OUT/problem.txt" --out-dir "$OUT/compare_P"
    stand compare --before "$OUT/before" --after "$OUT/after" --keys "$OUT/normal.txt" --out-dir "$OUT/compare_N"
fi
