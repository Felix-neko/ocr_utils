#!/usr/bin/env bash
#
# «Хозяйство и право» (пробный пак): выпуски 1978/07–12, 1979/01–12, 1981/01–03 — журнальные развороты JPG 6273×5020,
# 1 164 кадра. Проверка на пропуски страниц, смаз, расфокус и текст у корешка — как пак-4 ПХ.
#   1. pages (верхний и нижний колонтитул) + sequence — номера страниц и разрывы ряда по выпускам;
#   2. features + pass2 + motion-rank — смаз: agree_peak ≥ 6 и mid_at_peak_rel ≥ 1.15; листы худших;
#   3. pass3 + pass3-rank — корешок: сжатие по шагу штрихов < 0.62, порча черт по 4 худшим строкам (worst_run) < 0.60;
#   4. defocus_detection по годам — балл DOM; пороги dom-si здесь абсолютные, смотреть хвост распределения партии.
#
# Читает: $HP_1991_PACK_ROOT/{1978,1979,1981}/<выпуск>/IMG_*.jpg (Я.Диск, только чтение).
# Пишет: $HP_1978_1981_OUT/{pages/,pages_footer/,motion/,midtone/,glyphs.csv,pass3_scores.csv,lines/,defocus/<год>/}.

set -euo pipefail
set -m
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"
cd "$(dirname "$0")/../.."

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-12}"
ROOT="$HP_1991_PACK_ROOT"
OUT="$HP_1978_1981_OUT"
SD=(uv run python -m research.spread_defects)
ONLY=(--only 1978/07 --only 1978/08 --only 1978/09 --only 1978/10 --only 1978/11 --only 1978/12 --only 1979 --only 1981)

"${SD[@]}" pages --root "$ROOT" "${ONLY[@]}" --out-dir "$OUT/pages" --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages"
"${SD[@]}" pages --root "$ROOT" "${ONLY[@]}" --out-dir "$OUT/pages_footer" --place footer --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages_footer"

"${SD[@]}" features --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass2 --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-rank --run-dir "$OUT/motion"
"${SD[@]}" motion-sheets --run-dir "$OUT/motion" --count 48 --out-dir "$OUT/motion/sheets"

"${SD[@]}" pass3 --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass3-rank --out-root "$OUT"

# Расфокус по годам (1978 — целиком, лишние 01–06 отбрасываются при разборе).
for YEAR in 1978 1979 1981; do
    mkdir -p "$OUT/defocus/$YEAR"
    uv run python -m ocr_utils.defocus_detection "$ROOT/$YEAR" --recursive \
        --tag-preset dom-si --worst-percent 30 --zonal-percent 30 --zonal-axis all \
        --txt-report "$OUT/defocus/$YEAR/defocus.txt" --md-report "$OUT/defocus/$YEAR/defocus.md" \
        --csv "$OUT/defocus/$YEAR/defocus.csv" --retake-list "$OUT/defocus/$YEAR/retake.txt" --reserve-cores 4
done
