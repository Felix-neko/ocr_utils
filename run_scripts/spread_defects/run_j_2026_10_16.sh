#!/usr/bin/env bash
#
# Партия 2026-10-16: журнальные развороты 1979 года, выпуски 01–12 (JPG 5078×4075, 982 кадра; номер страницы —
# в верхнем колонтитуле). Проверка на пропуски страниц, смаз, расфокус и текст у корешка — как пак-4 ПХ.
#   1. pages + sequence — номера колонтитула (tesseract) и разрывы ряда номеров по выпускам;
#   2. features + pass2 + motion-rank — смаз: правило пака-4 agree_peak ≥ 6 и mid_at_peak_rel ≥ 1.15; листы худших;
#   3. pass3 + pass3-rank — корешок: сжатие по шагу штрихов < 0.62, порча черт по 4 худшим строкам < 0.45 / 0.60;
#   4. defocus_detection — расфокус DOM с пресетом dom-si, худшие 30 % и зональный отчёт.
#
# Читает: $J_2026_10_16_ROOT/1979/<выпуск>/IMG_*.jpg (Я.Диск, только чтение).
# Пишет: $J_2026_10_16_OUT/{pages/,motion/,midtone/,glyphs.csv,pass3.csv,pass3_scores.csv,lines/,defocus/}.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"
cd "$(dirname "$0")/../.."

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-12}"
ROOT="$J_2026_10_16_ROOT"
OUT="$J_2026_10_16_OUT"
SD=(uv run python -m research.spread_defects)

"${SD[@]}" pages --root "$ROOT" --only 1979 --out-dir "$OUT/pages" --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages"

"${SD[@]}" features --root "$ROOT" --only 1979 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass2 --root "$ROOT" --only 1979 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-rank --run-dir "$OUT/motion"
"${SD[@]}" motion-sheets --run-dir "$OUT/motion" --count 48 --out-dir "$OUT/motion/sheets"

"${SD[@]}" pass3 --root "$ROOT" --only 1979 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass3-rank --out-root "$OUT"

# Расфокус: пресет dom-si калиброван на разворотах; JPG того же порядка разрешения.
mkdir -p "$OUT/defocus"
uv run python -m ocr_utils.defocus_detection "$ROOT/1979" --recursive \
    --tag-preset dom-si --worst-percent 30 --zonal-percent 30 --zonal-axis all \
    --txt-report "$OUT/defocus/defocus.txt" --md-report "$OUT/defocus/defocus.md" --csv "$OUT/defocus/defocus.csv" \
    --retake-list "$OUT/defocus/retake.txt" --link-dir "$OUT/defocus/worst" --reserve-cores 4
