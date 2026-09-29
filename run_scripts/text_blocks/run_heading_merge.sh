#!/usr/bin/env bash
#
# Стенд «сращивание строк разного набора» (research/heading_merge): подпись автора + заголовок, шапка журнала +
# номер / месяц / «Год издания», дата + строки оглавления, эмблема + заголовок. Запреты — подменой функций
# детектора текстовых блоков на время разбора (HeadingPatch), боевой код не меняется.
#
# Выборки (research/heading_merge/sets): C — 47 подтверждённых полос (48 случаев, cases.tsv, с типом),
# P — все 322 полосы с кандидатами меры (set_p.txt, C входит в P), N — 300 случайных полос без кандидатов
# (set_n.txt). Мера кандидата: стык внутри ряда блока — крупная сторона ≥ 18 px рабочей копии, кегль ≥ 1.6×,
# просвет ≥ 1.5 меньшего кегля (measure.Joint.candidate).
#
# Итог 2026-09-29 (HEAD 5c9350b, reports/text_blocks_heading_merge.md), вариант full против base:
#   C: стыков 66 → 0, полос со стыками 46 → 0 (подписи 18, шапки 25, оглавления 2, эмблемы 2);
#   P: полос со стыками 263 → 216; N: изменилось 9 из 300 (заметно — сдвиг кромки > 10 px — 7, глазами без
#   порчи), осей 16 901 → 16 902, блоков 927 → 927.
#
# Шаги (STEP=all — все подряд): run base и run full по P ∪ N (~3 мин каждый на 16 воркерах), compare
# (сводка по C по типам, P, N; склейки «было | стало» и листы 2×2), significant (листы заметных изменений),
# trace (виновники оставшихся стыков C под full, последовательно ~3 мин).
#
# Выход: $OUT/<вариант>/{pages,overlays,summary.csv}, $OUT/compare_base_<вариант>/{compare.csv,C,P,N,sheets_*},
# $OUT/trace_<вариант>.tsv.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_heading_merge.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-/mnt/system/raw/mts/pack1_heading_merge/bench}"
PACK="${PACK:-/mnt/system/raw/mts/pack1_page_analysis_v4}"
VARIANT="${VARIANT:-full}"
STEP="${STEP:-all}"
JOBS="${JOBS:-16}"
SETS="research/heading_merge/sets"

mkdir -p "$OUT"
cat "$SETS/set_p.txt" "$SETS/set_n.txt" | sort -u > "$OUT/set_pn.txt"

bench() { uv run python -m research.heading_merge "$@"; }

if [[ "$STEP" == all || "$STEP" == run ]]; then
    for variant in base "$VARIANT"; do
        bench run --sharpened-dir "$SHARPENED_DIR" --source "$PACK" --out-dir "$OUT" --pages "$OUT/set_pn.txt" \
            --variant "$variant" --jobs "$JOBS"
    done
fi
if [[ "$STEP" == all || "$STEP" == compare ]]; then
    bench compare --run-dir "$OUT" --after "$VARIANT"
    bench significant --run-dir "$OUT" --after "$VARIANT"
fi
if [[ "$STEP" == all || "$STEP" == trace ]]; then
    bench trace --sharpened-dir "$SHARPENED_DIR" --source "$PACK" --variant "$VARIANT" --out "$OUT/trace_$VARIANT.tsv"
fi
