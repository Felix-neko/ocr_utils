#!/usr/bin/env bash
#
# Отладочный прогон подпакета curved_layout по отобранным страницам пака-1: оси строк, огибающие
# блоков, выключка, оверлеи для просмотра глазами.
#
# Страницы отобраны по метрикам прогона pack1_v16 (детектор геометрии):
#   * самые кривые строки (sagitta p90 в долях высоты): 1971/10 с.87 — 0.53, 1973/07 с.88 — 0.48,
#     1967/10 с.63 — 0.46, 1971/10 с.93 — 0.45, 1973/11 с.79 — 0.44, 1968/07 с.93 — 0.43;
#   * вёрстка в 2–4 колонки: 1973/06 с.65 (пример пользователя), 1973/07 с.77, 1973/08 с.85,
#     1971/10 с.95, 1970/02 с.90, 1975/05 с.97.
# Каждая страница считается в обоих вариантах рендера (с коррекцией геометрии и без).
#
# Читает: $GEO_PDF_DIR и $NOGEO_PDF_DIR. Пишет: $OUT/{pages,overlays,blocks.csv,report.md}.
# ~0.5 с на страницу, весь набор — меньше минуты.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../geometry_regression/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-/mnt/system/raw/mts/curved_layout_debug}"
PAGES="${PAGES:-full_1973_06:65,full_1971_10:87,full_1973_07:88,full_1967_10:63,full_1971_10:93,full_1973_11:79,full_1968_07:93,full_1973_07:77,full_1973_08:85,full_1971_10:95,full_1970_02:90,full_1975_05:97}"

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
