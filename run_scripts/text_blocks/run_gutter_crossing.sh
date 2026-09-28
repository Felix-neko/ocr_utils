#!/usr/bin/env bash
#
# Стенд «строки через межколонник» (research/gutter_crossing): пересчёт текстовых блоков пака-1 тем же
# входом, что в разборе v3 (объекты и запреты из pack1_page_analysis_v3, вторая ось строки), в трёх
# режимах межколонников (columns.GutterMode):
#   * legacy    — прежний: один запрет на межколонник по общей части ломаной, на наклоне он выпадает;
#   * segmented — запрет на каждый отрезок ломаной (класс A: наклонная полоса, 1966/02 IMG_0076_1L);
#   * short     — плюс короткие межколонники — продолжения найденных за заголовком (класс B:
#                 1969/03 IMG_0147_1L, 1967/05 IMG_0095_2R, 1975/12 IMG_0141_2R).
# Мера — metrics.gutter_crossings_of: ось, под которой пустота шире пробела, продолжающаяся столбцом
# по соседним строкам с выровненными краями.
#
# Шаги (ШАГ=all — все подряд):
#   legacy  — весь пак в прежнем режиме без оверлеев (~60 мин на 16 воркерах, ~200 полос/мин);
#   measure — пересчёт меры по сохранённым осям (после правки порогов меры, без разбора);
#   select  — P (оси через межколонник) и N того же размера по слоям → $OUT/problem.txt, normal.txt;
#   sets    — P ∪ N в трёх режимах с оверлеями;
#   compare — склейки «было | стало» и CSV: $OUT/compare_legacy_segmented, compare_legacy_short.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_gutter_crossing.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scan_markup/pack1/common.sh"

OUT="${OUT:-/mnt/system/raw/mts/pack1_gutter_crossing}"
PACK="${PACK:-/mnt/system/raw/mts/pack1_page_analysis_v3}"
JOBS="${JOBS:-16}"
STEP="${STEP:-all}"

stand() {
    uv run python -m research.gutter_crossing "$@"
}

if [[ "$STEP" == all || "$STEP" == legacy ]]; then
    stand run --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --out-dir "$OUT" --mode legacy --no-overlays \
        --jobs "$JOBS"
fi
if [[ "$STEP" == measure ]]; then
    stand measure --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --out-dir "$OUT" --mode legacy --jobs "$JOBS"
fi
if [[ "$STEP" == all || "$STEP" == select ]]; then
    stand select --out-dir "$OUT"
fi
if [[ "$STEP" == all || "$STEP" == sets ]]; then
    cat "$OUT/problem.txt" "$OUT/normal.txt" > "$OUT/sets.txt"
    for MODE in legacy segmented short; do
        stand run --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --out-dir "$OUT/sets" --keys "$OUT/sets.txt" \
            --mode "$MODE" --jobs "$JOBS"
    done
fi
if [[ "$STEP" == all || "$STEP" == compare ]]; then
    stand compare --out-dir "$OUT/sets" --before legacy --after segmented
    stand compare --out-dir "$OUT/sets" --before legacy --after short
fi
