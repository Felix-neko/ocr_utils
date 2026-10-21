#!/usr/bin/env bash
#
# «Хозяйство и право» (пробный пак): выпуски 1981/01–12 и 1982/01–12 — журнальные развороты JPG ~6300–6700×5000, 434 dpi,
# ~1 270 кадров (1981/01–03 — папки «01–03 оплачено»). Проверка на пропуски, смаз, расфокус,
# пересвет и текст у корешка — как пак-4 ПХ.
#   1. pages (верхний и нижний колонтитул) + sequence — номера страниц и разрывы ряда по выпускам;
#   2. features + pass2 + motion-rank — смаз: agree_peak ≥ 6 и mid_at_peak_rel ≥ 1.15; листы худших;
#      motion-zones + motion-zone-sheets — карта смаза по зонам всего кадра (ореол к медиане выпуска ≥ 1.6), лист
#      на кадр с зонами: отсматривать ЭТИ листы, а не круп у одной точки пика;
#   3. pass3 + pass3-rank — корешок: сжатие по шагу штрихов < 0.62, порча черт по 4 худшим строкам (worst_run) < 0.60;
#   4. pass4 — пересвет по плиткам: худшее окно 3×3 плиток по толщине штриха к медиане выпуска (< 0.72 — сильный);
#   5. defocus_detection по годам — балл DOM; пороги dom-si здесь абсолютные, смотреть хвост распределения партии.
#
# Читает: $HP_1991_PACK_ROOT/{1981,1982}/<выпуск>/IMG_*.jpg (Я.Диск, только чтение).
# Пишет: $HP_1981_1982_OUT/{pages/,pages_footer/,motion/{scores,zones,zone_frames}.csv,motion/zone_sheets/,midtone/,glyphs.csv,pass3_scores.csv,lines/,exposure.csv,exposure/,defocus/<год>/}.

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
OUT="$HP_1981_1982_OUT"
SD=(uv run python -m research.spread_defects)
ONLY=(--only 1981 --only 1982)

"${SD[@]}" pages --root "$ROOT" "${ONLY[@]}" --out-dir "$OUT/pages" --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages"
"${SD[@]}" pages --root "$ROOT" "${ONLY[@]}" --out-dir "$OUT/pages_footer" --place footer --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages_footer"

"${SD[@]}" features --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass2 --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-rank --run-dir "$OUT/motion"
"${SD[@]}" motion-sheets --run-dir "$OUT/motion" --count 48 --out-dir "$OUT/motion/sheets"
# Карта смаза по всему кадру (не одна точка пика): зоны ореола и двоения без растра и цветных плашек; лист на кадр.
"${SD[@]}" motion-zones --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-zone-sheets --out-root "$OUT" --jobs "$JOBS"

"${SD[@]}" pass3 --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass3-rank --out-root "$OUT"

"${SD[@]}" pass4 --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"

# Расфокус по годам.
for YEAR in 1981 1982; do
    mkdir -p "$OUT/defocus/$YEAR"
    uv run python -m ocr_utils.defocus_detection "$ROOT/$YEAR" --recursive \
        --tag-preset dom-si --worst-percent 30 --zonal-percent 30 --zonal-axis all \
        --txt-report "$OUT/defocus/$YEAR/defocus.txt" --md-report "$OUT/defocus/$YEAR/defocus.md" \
        --csv "$OUT/defocus/$YEAR/defocus.csv" --retake-list "$OUT/defocus/$YEAR/retake.txt" --reserve-cores 4
done
