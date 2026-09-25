#!/usr/bin/env bash
#
# Стороны границы текстового блока и выравнивание по ним (page_layout.text_blocks.sides):
# все методы разметки (construct, rays, frame) и проверки выравнивания (trend, tangent, robust)
# на одних и тех же полосах, оверлеи по методу и склейки «метод рядом с методом».
#
# Полосы: 13 текстовых из валидационного набора (run_text_pages.sh) и 4 табличные и со схемами
# (1971/09 с.80 — блок-схема, 1968/01 с.75 — шапка таблицы, 1973/08 с.19, 1969/11 с.69 — сноска,
# разрезанная на два блока). Контрольные для выравнивания: 1976/09 с.92 (колонки по формату, дефис
# переноса вынесен за край набора на 1.1 мм), 1975/05 с.97, 1973/08 с.85. Вариант nogeo.
# Разбор без подсказок, ~2 с на полосу.
#
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR. Пишет: $OUT/{sides_*,align_*,compare,pages}/ и $OUT/report.md.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../geometry_regression/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-/mnt/system/raw/mts/curved_layout_sides}"
PAGES="${PAGES:-full_1976_09:92,full_1973_06:65,full_1971_10:87,full_1973_07:88,full_1967_10:63,full_1971_10:93,full_1973_11:79,full_1968_07:93,full_1973_07:77,full_1973_08:85,full_1971_10:95,full_1970_02:90,full_1975_05:97,full_1971_09:80,full_1968_01:75,full_1973_08:19,full_1969_11:69}"

uv run python -m ocr_utils.page_layout.text_blocks sides \
    --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --pages "$PAGES" --variant nogeo \
    --sides-method construct --out-dir "$OUT" "$@"
