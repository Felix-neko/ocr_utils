#!/usr/bin/env bash
#
# Валидационный прогон детектора текстовых блоков (page_layout.text_blocks) со ВСПОМОГАТЕЛЬНОЙ ИНФОРМАЦИЕЙ page_layout.
#
# Страницы с боковым текстом, таблицами и блок-схемами: на них видно, что дают подсказки —
# маска запретного поля (растр), рамки таблиц и схем как запрет сцепки, области бокового текста
# отдельным разбором с поворотом.
#
# Каждая страница считается дважды: без подсказок ($OUT/plain) и с ними ($OUT/hinted) — оверлеи
# «было — стало» лежат рядом под одинаковыми именами.
#
# Читает: $NOGEO_PDF_DIR, $LAYOUT_CACHE_DIR (кэш surya, промах = разбор по одним пикселям).
# Пишет: $OUT/{plain,hinted,cells}/{pages,overlays,blocks.csv,report.md}. ~4 с на страницу.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../geometry_regression/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-/mnt/system/raw/mts/curved_layout_hints}"
LAYOUT_CACHE="${LAYOUT_CACHE_DIR:-/mnt/system/raw/mts/pack1_page_layout}"
# Кэш text_layer_fix: ячейки таблиц и поворот текста в них (123 выпуска пака-1).
TEXT_LAYER_CACHE="${TEXT_LAYER_CACHE:-/mnt/system/raw/mts/pack1_text_layer_fix/pack1_v2/cache}"
# Боковой текст на схемах и в таблицах, блок-схемы, крупные чертежи.
PAGES="${PAGES:-full_1974_11:50,full_1969_02:43,full_1972_09:13,full_1973_08:19,full_1967_07:73,full_1968_01:75,full_1971_11:59,full_1975_05:99,full_1970_06:62}"

common=(--geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --pages "$PAGES"
        --variant nogeo --engine ink --smooth-block 2.5 --coarse-factor 3 --smooth-line 0.6)

echo "== без подсказок =="
uv run python -m ocr_utils.page_layout.text_blocks analyze "${common[@]}" --out-dir "$OUT/plain" "$@"
echo "== с подсказками =="
uv run python -m ocr_utils.page_layout.text_blocks analyze "${common[@]}" --out-dir "$OUT/hinted" --layout-cache "$LAYOUT_CACHE" "$@"
echo "== с подсказками и ячейками таблиц =="
uv run python -m ocr_utils.page_layout.text_blocks analyze "${common[@]}" --out-dir "$OUT/cells" \
    --layout-cache "$LAYOUT_CACHE" --text-layer-cache "$TEXT_LAYER_CACHE" "$@"
