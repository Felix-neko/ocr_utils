#!/usr/bin/env bash
#
# Валидационный прогон детектора текстовых блоков (page_layout.text_blocks): текстовый контент и оглавления.
#
# Часть валидационного множества «текстовый контент и оглавления»: сплошной набор в 1–4 колонки,
# заголовки, сноски, подписи авторов, одна таблица с отточиями.
#
# Страницы отобраны по метрикам прогона pack1_v16 (детектор геометрии):
#   * самые кривые строки (sagitta p90 в долях высоты): 1971/10 с.87 — 0.53, 1973/07 с.88 — 0.48,
#     1967/10 с.63 — 0.46, 1971/10 с.93 — 0.45, 1973/11 с.79 — 0.44, 1968/07 с.93 — 0.43;
#   * вёрстка в 2–4 колонки: 1973/06 с.65, 1973/07 с.77, 1973/08 с.85, 1971/10 с.95,
#     1970/02 с.90, 1975/05 с.97;
#   * последняя строка блока (достройка хвоста до линии отсечки): 1976/09 с.92 PDF (с.90 журнала,
#     скан IMG_0149_1L) — предпоследние строки колонок выгнуты аркой, последние короткие.
# Каждая страница считается в обоих вариантах рендера (с коррекцией геометрии и без).
# ~1 с на страницу, весь набор — меньше минуты.
#
# Второй проход — с ВСПОМОГАТЕЛЬНОЙ ИНФОРМАЦИЕЙ page_layout ($OUT_hinted): зоны растра, таблиц и
# line art запрещены, искать текст внутри них нельзя. На этом наборе таблиц нет, а line art —
# на двух полосах (1973/06 с.65 и 1973/11 с.79: заголовки-вензели), и они из разбора уходят
# вместе со своей зоной. Остальные десять полос обязаны совпасть с основным проходом.
#
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR, $LAYOUT_CACHE_DIR (промах кэша = разбор по пикселям).
# Пишет: $OUT/{pages,overlays,blocks.csv,report.md} и то же в $OUT_hinted.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../geometry_regression/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-/mnt/system/raw/mts/curved_layout_validation/text_and_toc}"
OUT_HINTED="${OUT_HINTED:-${OUT}_hinted}"
LAYOUT_CACHE="${LAYOUT_CACHE_DIR:-/home/felix/Projects/mts_markup/pack1_page_layout}"
PAGES="${PAGES:-full_1973_06:65,full_1971_10:87,full_1973_07:88,full_1967_10:63,full_1971_10:93,full_1973_11:79,full_1968_07:93,full_1973_07:77,full_1973_08:85,full_1971_10:95,full_1970_02:90,full_1975_05:97,full_1976_09:92}"

common=(--geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --pages "$PAGES"
        --variant both --engine ink --smooth-block 2.5 --coarse-factor 3 --smooth-line 0.6)

echo "== без подсказок =="
uv run python -m ocr_utils.page_layout.text_blocks analyze "${common[@]}" --out-dir "$OUT" "$@"

echo "== с масками растра, таблиц и line art =="
uv run python -m ocr_utils.page_layout.text_blocks analyze "${common[@]}" --out-dir "$OUT_HINTED" \
    --layout-cache "$LAYOUT_CACHE" --forbid-figures "$@"
