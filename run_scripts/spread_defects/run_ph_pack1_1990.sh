#!/usr/bin/env bash
#
# «Плановое хозяйство», пак-1 (300 dpi), 1990/09–12 — журнальные развороты JPG 3922×3299, 268 кадров. Проверка на
# пропуски страниц, смаз, расфокус и текст у корешка — как пак-4 ПХ (reports/spread_defects_pack4.md).
#   0. scripts/rescale_frames.py — копия ×1.5 (300 → 450 dpi, Lanczos): все пороги стенда заданы в px пака-4
#      (корпус буквы 23–26 px, двоение 4–5 px, полосы у сгиба), копия приводит их к тем же мм бумаги;
#   1. pages (верхний и нижний колонтитул) + sequence — номера страниц и разрывы ряда по выпускам;
#   2. features + pass2 + motion-rank — смаз: agree_peak ≥ 6 и mid_at_peak_rel ≥ 1.15; листы худших;
#   3. pass3 + pass3-rank — корешок: сжатие по шагу штрихов < 0.62, порча черт по 4 худшим строкам < 0.45 / 0.60;
#   4. defocus_detection по оригиналам — балл DOM с пресетом dom-si; пороги абсолютные, смотреть хвост партии.
#
# Читает: $PH_PACK1_1990_ROOT/1990/<выпуск>/NNNN.jpg (Я.Диск, только чтение).
# Пишет: $PH_PACK1_1990_OUT/{frames_450/,pages/,pages_footer/,motion/,midtone/,glyphs.csv,pass3_scores.csv,lines/,defocus/}.

set -euo pipefail
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"
cd "$(dirname "$0")/../.."

# Пересчёт читает NTFS-3G — 12; дальше всё с SSD — 16.
RESCALE_JOBS="${RESCALE_JOBS:-12}"
JOBS="${JOBS:-16}"
OUT="$PH_PACK1_1990_OUT"
ROOT="$OUT/frames_450"
SD=(uv run python -m research.spread_defects)

uv run python scripts/rescale_frames.py --src "$PH_PACK1_1990_ROOT/1990" --dst "$ROOT/1990" \
    --src-dpi 300 --dst-dpi 450 --jobs "$RESCALE_JOBS"

"${SD[@]}" pages --root "$ROOT" --only 1990 --out-dir "$OUT/pages" --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages"
"${SD[@]}" pages --root "$ROOT" --only 1990 --out-dir "$OUT/pages_footer" --place footer --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages_footer"

"${SD[@]}" features --root "$ROOT" --only 1990 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass2 --root "$ROOT" --only 1990 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-rank --run-dir "$OUT/motion"
"${SD[@]}" motion-sheets --run-dir "$OUT/motion" --count 48 --out-dir "$OUT/motion/sheets"

"${SD[@]}" pass3 --root "$ROOT" --only 1990 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass3-rank --out-root "$OUT"

# Расфокус — по оригиналам 300 dpi (пересчитанная копия мягче и сдвинула бы шкалу DOM).
mkdir -p "$OUT/defocus"
uv run python -m ocr_utils.defocus_detection "$PH_PACK1_1990_ROOT/1990" --recursive \
    --tag-preset dom-si --worst-percent 30 --zonal-percent 30 --zonal-axis all \
    --txt-report "$OUT/defocus/defocus.txt" --md-report "$OUT/defocus/defocus.md" --csv "$OUT/defocus/defocus.csv" \
    --retake-list "$OUT/defocus/retake.txt" --reserve-cores 4
