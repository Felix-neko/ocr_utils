#!/usr/bin/env bash
#
# Шаг 3: уточнённая руками разметка из CVAT обратно в базу — в отдельный файл той же схемы ($DB_REVIEWED).
#
# Автоматическая детекция ($DB) остаётся нетронутой, её всегда можно сравнить с правкой. В целевой базе пак
# «досканы-1» пересоздаётся целиком. Оригиналы не читаются: координаты пересчитываются делителем из базы. Секунды.
# После забора — сверить числа (count-shapes) с отчётом команды.

set -euo pipefail
source "$(dirname "$0")/common.sh"
echo "Итог:     $DB_REVIEWED"

FROM_CVAT_ARGS=(
    --db "$DB"
    --out-db "$DB_REVIEWED"
    --pack-name "$PACK_NAME"
)
uv run python -m ocr_utils.scan_markup from-cvat "${FROM_CVAT_ARGS[@]}" "$@"
