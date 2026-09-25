#!/usr/bin/env bash
#
# Валидационный прогон curved_layout: табличный и нетекстовый контент.
#
# Часть валидационного множества «табличный и нетекстовый контент»: line art, таблицы и
# повёрнутый текст. Детектор под них НЕ дорабатывался — прогон нужен, чтобы посмотреть глазами,
# что он найдёт на сложных страницах.
#
# Страницы отобраны по разметке пака (`pack1_reviewed.sqlite`, вид области `line_art_schema` и
# `table`) и по сводке повёрнутых таблиц (`pack1_rotated_tables/summary.csv`):
#   * line art (12): крупнейшие схемы и графики — 1975/05 с.99 (19.6 Мпкс), 1968/01 с.54,
#     1971/09 с.80, 1974/07 с.78, 1973/10 с.51, 1976/07 с.19, 1973/01 с.84, 1974/11 с.45;
#     и схемы рядом с таблицами — 1969/11 с.69, 1970/06 с.62, 1966/03 с.34, 1970/01 с.19;
#   * таблицы с повёрнутым текстом в ячейках (3): 1973/08 с.19 (45 повёрнутых ячеек из 50),
#     1973/08 с.24 (44 из 51), 1967/07 с.73 (36 из 94);
#   * повёрнутый текст вне ячеек, таблица набрана боком целиком (3): 1968/01 с.75,
#     1971/11 с.59, 1969/06 с.44;
#   * повёрнутый текст на line art (3): 1969/02 с.43, 1974/11 с.50, 1972/09 с.13.
# Каждая страница считается в обоих вариантах рендера. ~1 с на страницу.
#
# Читает: $GEO_PDF_DIR и $NOGEO_PDF_DIR. Пишет: $OUT/{pages,overlays,blocks.csv,report.md}.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../geometry_regression/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-/mnt/system/raw/mts/curved_layout_validation/tables_and_nontext}"
PAGES="${PAGES:-full_1975_05:99,full_1968_01:54,full_1971_09:80,full_1974_07:78,full_1973_10:51,full_1976_07:19,full_1973_01:84,full_1974_11:45,full_1969_11:69,full_1970_06:62,full_1966_03:34,full_1970_01:19,full_1973_08:19,full_1973_08:24,full_1967_07:73,full_1968_01:75,full_1971_11:59,full_1969_06:44,full_1969_02:43,full_1974_11:50,full_1972_09:13}"

uv run python -m ocr_utils.curved_layout analyze \
    --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" \
    --pages "$PAGES" \
    --variant both \
    --engine ink \
    --smooth-block 2.5 \
    --coarse-factor 3 \
    --smooth-line 0.6 \
    --out-dir "$OUT" \
    "$@"
