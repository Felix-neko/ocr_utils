#!/usr/bin/env bash
#
# Пары «было | стало» для находок детектора кривых строк: исходная полоса против страницы,
# в которую её превратил FineReader (с включённой коррекцией геометрии).
#
# ЗАЧЕМ. Проверить глазами, хорошо ли FineReader распрямляет строки именно на тех полосах,
# где они кривые. Реперные горизонтали через 1/12 высоты на обеих панелях.
#
# ВХОД — симлинки combo/ детектора и PDF после FineReader. Порядок страниц PDF = порядок
# файлов в папке выпуска; число страниц сверяется с числом полос. FineReader обработал не
# все выпуски — полосы без PDF перечисляются в отчёте.
# ВЫХОД — по картинке на находку, имя как у симлинка; на SSD, не в репозиторий.
set -euo pipefail
source "$(dirname "$0")/common.sh"
set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

OUT_DIR="/mnt/SYSTEM/raw/mts/pack1_finereader_compare"

ARGS=(
    --links-dir curved_lines_pack1_links/combo
    --pack-root "$PACK_DIR"
    --source-root "$SHARPENED_DIR"     # левая панель — то, что реально ушло в PDF
    --pdf-dir "$PDF_ROOT/full_pdfs_binary_no_bg_brightening"
    --out-dir "$OUT_DIR"
    --md-report finereader_compare_pack1.md
    --dpi 150
    --jobs 8
)

echo "Пары было|стало → $OUT_DIR"
uv run python -m ocr_utils.scan_markup.curved_lines.finereader_compare "${ARGS[@]}" "$@"
