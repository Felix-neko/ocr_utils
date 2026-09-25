#!/usr/bin/env bash
#
# Выравнивание текстовых блоков на СИЛЬНО НАКЛОНЁННЫХ колонках (page_layout.text_blocks):
# разбор и стенд сторон в обоих вариантах рендера, чтобы сравнить вердикты одной колонки в geo и nogeo.
#
# Полосы отобраны по прогону детектора порчи геометрии pack1_v16
# (/mnt/system/raw/mts/pack1_geometry_regression/pack1_v16/metrics.csv):
#   * наибольший наклон края в варианте с коррекцией (edge_shear_max_a_deg): 1966/04 с.85 — 12.0°
#     (проверить, не выброс), 1975/02 с.95 — 4.4°, 1967/12 с.91 — 4.3°, 1975/09 с.97 — 4.2°,
#     1968/04 с.90 — 4.2°, 1976/01 с.56 — 4.1°, 1976/01 с.66 — 4.1°, 1972/09 с.83 — 3.3°;
#   * наибольший сдвиг края коррекцией (edge_shear_delta_mm): 1973/01 с.31 — 7.3 мм,
#     1968/01 с.76 — 6.6 мм, 1967/07 с.14 — 6.1 мм.
# ~2 с на полосу и вариант на каждую команду.
#
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR. Пишет: $OUT/analyze/{pages,overlays,blocks.csv,report.md} и
# $OUT/sides/{sides_*,align_*,compare,pages,report.md}.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../geometry_regression/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-/mnt/system/raw/mts/curved_layout_tilted}"
PAGES="${PAGES:-full_1966_04:85,full_1975_02:95,full_1967_12:91,full_1975_09:97,full_1968_04:90,full_1976_01:56,full_1976_01:66,full_1972_09:83,full_1973_01:31,full_1968_01:76,full_1967_07:14}"

uv run python -m ocr_utils.page_layout.text_blocks analyze --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" \
    --pages "$PAGES" --variant both --engine ink --out-dir "$OUT/analyze" "$@"
uv run python -m ocr_utils.page_layout.text_blocks sides --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" \
    --pages "$PAGES" --variant both --sides-method construct --out-dir "$OUT/sides"
