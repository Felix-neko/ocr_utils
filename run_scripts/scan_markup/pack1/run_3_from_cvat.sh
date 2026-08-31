#!/usr/bin/env bash
#
# Шаг 3: уточнённая руками разметка из CVAT обратно в базу — в отдельный файл той же схемы.
#
# Отдельная база, а не правка исходной: автоматическая детекция остаётся нетронутой, и её
# всегда можно сравнить с тем, что получилось после ручной правки.
#
# Оригиналы не читаются вовсе: координаты пересчитываются делителем из базы. Прогонять стоит
# и просто так, для страховки, — особенно перед шагом 2 с --recreate-stale.

set -euo pipefail
source "$(dirname "$0")/common.sh"
echo "Итог:     $DB_REVIEWED"

FROM_CVAT_ARGS=(
    --db "$DB"
    --out-db "$DB_REVIEWED"
    --pack-name "$PACK_NAME"
    # --only-year 1966
)
uv run python -m ocr_utils.scan_markup from-cvat "${FROM_CVAT_ARGS[@]}"
