#!/usr/bin/env bash
#
# «Хозяйство и право» (пробный пак), партия Лены 2026-10-30: выпуски 1985/07–12, 1986/01–12, 1987/01–06 — журнальные
# развороты JPG, ~1 300 кадров, и 13 досканов 1977, 1980, 1991. Проверка — как ХиП 1982–1985.
#   0. scripts/rescale_frames.py --src-dpi-from-file — копия кадров из DOWN на SSD, кадры 599 dpi (с 1986/10 IMG_1013)
#      ужимаются до 434: пороги стенда заданы в px съёмки 434–450 dpi, а 599 — тот же кадр, растянутый при выгрузке;
#   1. pages (верхний и нижний колонтитул) + sequence — номера страниц и разрывы ряда по выпускам;
#   2. features + pass2 + motion-rank — смаз: agree_peak ≥ 6 и mid_at_peak_rel ≥ 1.15; листы худших;
#      motion-zones + motion-zone-sheets — карта смаза по зонам всего кадра (ореол к медиане выпуска ≥ 1.6), лист
#      на кадр с зонами: отсматривать ЭТИ листы, а не круп у одной точки пика;
#   3. pass3 + pass3-rank — корешок: сжатие по шагу штрихов < 0.62, порча черт по 4 худшим строкам (worst_run) < 0.60;
#   4. pass4 — пересвет по плиткам: худшее окно 3×3 плиток по толщине штриха к медиане выпуска (< 0.72 — сильный);
#   5. defocus_detection по годам (по копии 434 dpi) — балл DOM; пороги dom-si абсолютные, смотреть хвост партии;
#   6. досканы: их выпуски с Я.Диска деревом симлинков, доскан вместо бракованного кадра — карта смаза и sequence.
#
# Читает: $HP_LENA_2026_10_30_SRC/<год>/<выпуск>/IMG_*.jpg и …/перефотанное/ (DOWN, только чтение); выпуски досканов —
#   Я.Диск $HP_1991_PACK_ROOT (только чтение).
# Пишет: $HP_1985_1987_OUT/{src/,pages/,pages_footer/,motion/,midtone/,glyphs.csv,pass3_scores.csv,lines/,exposure.csv,
#   exposure/,defocus/<год>/}, $HP_RESHOOT_2026_10_30_OUT/{root/,pages/,motion/,…}.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"
cd "$(dirname "$0")/../.."

# Копия читает NTFS-3G — 12; дальше всё с SSD — 16.
RESCALE_JOBS="${RESCALE_JOBS:-12}"
JOBS="${JOBS:-16}"
SRC="$HP_LENA_2026_10_30_SRC"
ROOT="$HP_1985_1987_ROOT"
OUT="$HP_1985_1987_OUT"
SD=(uv run python -m research.spread_defects)
ONLY=(--only 1985 --only 1986 --only 1987)

# 1985/01–06 проверены в партии 2026-10-27, 1987/07–12 пустые.
for ISSUE in 1985/{07..12} 1986/{01..12} 1987/{01..06}; do
    uv run python scripts/rescale_frames.py --src "$SRC/$ISSUE" --dst "$ROOT/$ISSUE" \
        --src-dpi-from-file --dst-dpi 434 --jobs "$RESCALE_JOBS"
done

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
for YEAR in 1985 1986 1987; do
    mkdir -p "$OUT/defocus/$YEAR"
    uv run python -m ocr_utils.defocus_detection "$ROOT/$YEAR" --recursive \
        --tag-preset dom-si --worst-percent 30 --zonal-percent 30 --zonal-axis all \
        --txt-report "$OUT/defocus/$YEAR/defocus.txt" --md-report "$OUT/defocus/$YEAR/defocus.md" \
        --csv "$OUT/defocus/$YEAR/defocus.csv" --retake-list "$OUT/defocus/$YEAR/retake.txt" --reserve-cores 4
done

# Досканы: выпуск с Я.Диска симлинками, доскан под именем бракованного кадра. Папка выпуска на Диске — «MM» с меткой
# («04 оплачено»), в «перефотанное» — просто «MM». Доскан сопоставляется по номеру кадра, а не по имени: в партии
# есть `1980/11/IMG_00847.jpg` (лишний ноль) вместо IMG_0847.
RS_OUT="$HP_RESHOOT_2026_10_30_OUT"
RS_ROOT="$RS_OUT/root"
for SHOT_DIR in "$SRC"/перефотанное/{1977,1980,1991}/*/; do
    ISSUE="$(basename "$(dirname "$SHOT_DIR")")/$(basename "$SHOT_DIR")"
    ISSUE_DIRS=("$HP_1991_PACK_ROOT/${ISSUE%/*}/${ISSUE#*/}"*/)
    [[ ${#ISSUE_DIRS[@]} -eq 1 ]] || { echo "выпуск $ISSUE на Диске: ${#ISSUE_DIRS[@]} папок" >&2; exit 1; }
    declare -A SHOTS=()
    for SHOT in "$SHOT_DIR"IMG_*.jpg; do
        NUM="$(basename "$SHOT" .jpg)"
        SHOTS[$((10#${NUM#IMG_}))]="$SHOT"
    done
    mkdir -p "$RS_ROOT/$ISSUE"
    for FRAME in "${ISSUE_DIRS[0]}"IMG_*.jpg; do
        NAME="$(basename "$FRAME")"
        NUM="${NAME%.jpg}"
        NUM="${NUM#IMG_}"
        # Переименованные пользователем кадры («IMG_0197.смаз_не_критичный.jpg») — как есть.
        if [[ "$NUM" =~ ^[0-9]+$ && -n "${SHOTS[$((10#$NUM))]:-}" ]]; then
            ln -sfn "${SHOTS[$((10#$NUM))]}" "$RS_ROOT/$ISSUE/$NAME"
            unset "SHOTS[$((10#$NUM))]"
        else
            ln -sfn "$FRAME" "$RS_ROOT/$ISSUE/$NAME"
        fi
    done
    [[ ${#SHOTS[@]} -eq 0 ]] || { echo "доскан без кадра в $ISSUE: ${SHOTS[*]}" >&2; exit 1; }
    unset SHOTS
done
"${SD[@]}" pages --root "$RS_ROOT" --out-dir "$RS_OUT/pages" --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$RS_OUT/pages"
"${SD[@]}" features --root "$RS_ROOT" --out-root "$RS_OUT" --jobs "$JOBS"
"${SD[@]}" pass2 --root "$RS_ROOT" --out-root "$RS_OUT" --jobs "$JOBS"
"${SD[@]}" motion-zones --out-root "$RS_OUT" --jobs "$JOBS"
"${SD[@]}" motion-zone-sheets --out-root "$RS_OUT" --jobs "$JOBS"
