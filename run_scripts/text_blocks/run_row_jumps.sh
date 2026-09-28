#!/usr/bin/env bash
#
# Стенд «перескок оси на соседнюю строку» (research/row_jumps): мера metrics.row_jumps_of по осям пака и
# проверка резки двухрядных сгустков RLSA (segment._split_two_rows, флаг split_rows).
#
# Мера: у оси, перескочившей на соседнюю строку (в том числе КОСО, чего ступенька steps_of не ловит),
# расстояние до соседней оси, накрывающей её почти целиком (≥ 90 % длины), меняется вдоль оси больше
# 0.6 шага (p95 − p5). Замер 2026-09-28 по осям прогона pack1_gutter_crossing/legacy: перескоки на 352
# полосах из 12 135 (412 осей); глазами по 24 вырезкам поясов 0.6–0.8 / 0.8–0.95 / ≥ 0.95 шага —
# настоящих 7/8, 6/8, 7/8 (ложные: заголовок крупного набора, выходные данные вразрядку, край тени,
# сноска петитом).
#
# Причина перескока (1966/01 IMG_0011_2R, 1967/06 IMG_0150_1L, 1972/07 IMG_0006_1L): смыкание RLSA
# сливает буквы двух рядов в один сгусток через карандашную пометку, соринку или слипшийся выносной
# элемент — кусок строки уходит на соседний ряд ещё до сцепки. Резка: буквы такого сгустка делятся
# на два уровня (середины разошлись больше 1.1 высоты буквы), каждая группа смыкается отдельно.
#
# Шаги (STEP=all — все подряд): measure (~2 мин) → select (P — все полосы с перескоками, N — столько же
# случайных без них) → plain, split, marks по P ∪ N с оверлеями (~5 мин каждый; VARIANTS — какие) →
# compare (plain→split, plain→marks, split→marks).
#
# Вторая причина (остаток после резки, 2026-09-28): кратка «й», точки «ё» и верхние индексы получали
# якорь на высоте строчной над строкой, ось куска задиралась на конце («путей.», «этой», «дней;»),
# и сцепка уводила последнюю строку абзаца на строку выше (1974/08 IMG_0089_2R). Вариант marks —
# резка + верхние знаки на базовой линии (pieces.baselines_of).
#
# Выход: $OUT/measures.csv, problem.txt, normal.txt, sets.txt, plain/, split/, compare/ (compare.csv,
# pages/, zoom/ — только полосы, где что-то изменилось).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_row_jumps.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-/mnt/system/raw/mts/pack1_row_jumps}"
PACK="${PACK:-/mnt/system/raw/mts/pack1_page_analysis_v4}"
# Оси всего пака — из прогона стенда межколонников (report.page_json, прежний ход межколонников).
AXES="${AXES:-/mnt/system/raw/mts/pack1_gutter_crossing/legacy/pages}"
JOBS="${JOBS:-16}"
STEP="${STEP:-all}"

stand() {
    uv run python -m research.row_jumps "$@"
}

mkdir -p "$OUT"
if [[ "$STEP" == all || "$STEP" == measure ]]; then
    stand measure --pages-dir "$AXES" --out "$OUT/measures.csv" --jobs "$JOBS"
fi
if [[ "$STEP" == all || "$STEP" == select ]]; then
    stand select --measures "$OUT/measures.csv" --out-dir "$OUT"
    cat "$OUT/problem.txt" "$OUT/normal.txt" > "$OUT/sets.txt"
fi
if [[ "$STEP" == all || "$STEP" == sets ]]; then
    for VARIANT in ${VARIANTS:-plain split marks}; do
        stand run --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --out-dir "$OUT" --keys "$OUT/sets.txt" \
            --variant "$VARIANT" --jobs "$JOBS"
    done
fi
if [[ "$STEP" == all || "$STEP" == compare ]]; then
    stand compare --out-dir "$OUT"
    stand compare --out-dir "$OUT" --before plain --after marks
    stand compare --out-dir "$OUT" --before split --after marks
fi
