#!/usr/bin/env bash
#
# Выгрузка полос оглавления по выпускам в списки для внешнего OCR:
# $TOC_LISTS_DIR/<год>/<выпуск>/toc_pages.txt — имена заострённых JPEG, после «#» вид и сила.
#
# Один и тот же файл идёт в run_scripts/external_ocr_models/run_issue_deepseek.sh как
# --skip-pages (основной прогон без оглавления) и как --pages запроса извлечения оглавления.
#
# Аргументы уходят команде toc-pages (например --only-year 1975 --kinds contents).

set -euo pipefail
source "$(dirname "$0")/common.sh"

uv run python -m ocr_utils.scan_markup toc-pages \
    --db "$DB" \
    --pack-name "$PACK_NAME" \
    --out-dir "$TOC_LISTS_DIR" \
    --suffix .jpg \
    "$@"
echo "Списки: $TOC_LISTS_DIR"
