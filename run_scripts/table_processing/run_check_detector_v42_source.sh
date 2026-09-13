#!/usr/bin/env bash
#
# Прогон детектора таблиц v.4.2 по ИСХОДНЫМ сканам «Готовое/пак-1» — TIFF до поворота полос,
# размытия фона и заострения — с кэшем surya, посчитанным по тем же TIFF
# (run_layout_pack_source.sh). Так детектор увидит полосы такими, какими их даст общий
# конвейер scan_markup. Пишет в отдельную папку отчёта.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SOURCE_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/Готовое/пак-1"
ARTIFACTS="reports/прогон детектора таблиц v.4.2"

uv run python -m research.legacy.table_processing check-detector \
    --docx-dir "$DOCX_DIR" \
    --pdf-dir "$RECOGNIZED_PDF_DIR" \
    --sharpened-dir "$SOURCE_DIR" \
    --ext tif \
    --db "$DB" \
    --pack "$PACK" \
    --out-dir "$ARTIFACTS" \
    --layout-dir "$OUT_DIR/layout_surya_готовое" \
    --jobs "$JOBS" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
wait "$PID"

echo "отчёт: $ARTIFACTS/README.md"
