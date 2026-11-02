#!/usr/bin/env bash
#
# «Хозяйство и право» (пробный пак): перепроверка выпусков 1985/07–12, 1986/01–12, 1987/01–06 партии Лены 2026-10-30
# по Я.Диску, после внесения (кадры 599 dpi с 1986/10 IMG_1013 уже ужаты до 434, исходники — в `.removed/`).
# Шаги — как run_hp_1985_1987.sh, без пересчёта dpi и досканов:
#   0. корень — дерево симлинков `<год>/<MM оплачено>/IMG_*.jpg` на кадры Диска, без `.removed/` (иначе стенд
#      примет `.removed` за отдельный выпуск);
#   1. pages (верхний и нижний колонтитул) + sequence — номера страниц и разрывы ряда по выпускам;
#   2. features + pass2 + motion-rank — смаз; motion-zones + motion-zone-sheets — карта смаза по зонам всего кадра;
#   3. pass3 + pass3-rank — корешок: сжатие < 0.62, порча черт < 0.60;
#   4. pass4 — пересвет по плиткам;
#   5. defocus_detection по годам — балл DOM.
#
# Читает: $HP_1991_PACK_ROOT/{1985,1986,1987}/<MM оплачено>/IMG_*.jpg (Я.Диск, только чтение).
# Пишет: $HP_1985_1987_DISK_OUT/{root/,pages/,pages_footer/,motion/,midtone/,glyphs.csv,pass3_scores.csv,lines/,
#   exposure.csv,exposure/,defocus/<год>/}.

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
OUT="$HP_1985_1987_DISK_OUT"
ROOT="$OUT/root"
SD=(uv run python -m research.spread_defects)
ONLY=(--only 1985 --only 1986 --only 1987)

# 1985/01–06 проверены в партии 2026-10-27.
for ISSUE in 1985/{07..12} 1986/{01..12} 1987/{01..06}; do
    ISSUE_DIRS=("$HP_1991_PACK_ROOT/$ISSUE"*/)
    [[ ${#ISSUE_DIRS[@]} -eq 1 ]] || { echo "выпуск $ISSUE на Диске: ${#ISSUE_DIRS[@]} папок" >&2; exit 1; }
    DST="$ROOT/${ISSUE%/*}/$(basename "${ISSUE_DIRS[0]}")"
    mkdir -p "$DST"
    for FRAME in "${ISSUE_DIRS[0]}"IMG_*.jpg; do
        ln -sfn "$FRAME" "$DST/$(basename "$FRAME")"
    done
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
