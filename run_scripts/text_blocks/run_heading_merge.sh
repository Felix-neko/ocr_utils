#!/usr/bin/env bash
#
# Стенд «сращивание строк разного набора» (research/heading_merge): подпись автора + заголовок, шапка журнала +
# номер / месяц / «Год издания», дата + строки оглавления, эмблема + заголовок.
# С 2026-09-29 запреты в боевом коде (text_blocks/typeset.py и места вызова); стенд сравнивает два прогона
# текущего кода под метками NAME (прогон «до» — из снимка `git archive` старого коммита, запуском из его папки).
#
# Выборки (research/heading_merge/sets): C — 47 подтверждённых полос (48 случаев, cases.tsv, с типом),
# P — все 322 полосы с кандидатами меры (set_p.txt, C входит в P), N — 300 случайных полос без кандидатов
# (set_n.txt). Мера кандидата: стык внутри ряда блока — крупная сторона ≥ 18 px рабочей копии, кегль ≥ 1.6×,
# просвет ≥ 1.5 меньшего кегля (measure.Joint.candidate).
#
# Итог 2026-09-29 (reports/text_blocks_heading_merge.md), боевой код против HEAD 04b2ea0:
#   C: стыков 66 → 0, полос со стыками 46 → 0 (подписи 18, шапки 25, оглавления 2, эмблемы 2);
#   P: полос со стыками 263 → 216; N: изменилось 9 из 300 (заметно — сдвиг кромки > 10 px — 7, глазами без
#   порчи), осей 16 901 → 16 902, блоков 927 → 927. Первый шаг (base) — прогон снимка старого кода.
#
# Шаги (STEP=all — все подряд): run по P ∪ N (~3 мин на 16 воркерах), compare
# (сводка по C по типам, P, N; склейки «было | стало» и листы 2×2), significant (листы заметных изменений),
# trace (виновники оставшихся стыков C, последовательно ~3 мин).
#
# Выход: $OUT/<NAME>/{pages,overlays,summary.csv}, $OUT/compare_<BEFORE>_<NAME>/{compare.csv,C,P,N,sheets_*},
# $OUT/trace_<NAME>.tsv. Временно — в домашней папке: /mnt/system заполнен.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_heading_merge.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-/mnt/hotstore/scan_processing/mts_markup/pack1_heading_merge/bench}"
PACK="${PACK:-/mnt/hotstore/scan_processing/mts/pack1_page_analysis_v4}"
NAME="${NAME:-after}"
BEFORE="${BEFORE:-base}"
STEP="${STEP:-all}"
JOBS="${JOBS:-16}"
SETS="research/heading_merge/sets"

mkdir -p "$OUT"
cat "$SETS/set_p.txt" "$SETS/set_n.txt" | sort -u > "$OUT/set_pn.txt"

bench() { uv run python -m research.heading_merge "$@"; }

if [[ "$STEP" == all || "$STEP" == run ]]; then
    bench run --sharpened-dir "$SHARPENED_DIR" --source "$PACK" --out-dir "$OUT" --pages "$OUT/set_pn.txt" \
        --name "$NAME" --jobs "$JOBS"
fi
if [[ "$STEP" == all || "$STEP" == compare ]]; then
    bench compare --run-dir "$OUT" --before "$BEFORE" --after "$NAME"
    bench significant --run-dir "$OUT" --before "$BEFORE" --after "$NAME"
fi
if [[ "$STEP" == all || "$STEP" == trace ]]; then
    bench trace --sharpened-dir "$SHARPENED_DIR" --source "$PACK" --out "$OUT/trace_$NAME.tsv"
fi
