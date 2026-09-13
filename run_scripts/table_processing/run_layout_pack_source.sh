#!/usr/bin/env bash
#
# Surya layout по ИСХОДНЫМ сканам «Готовое/пак-1» (TIFF 600 dpi на медленном NTFS-3G):
# pickle на полосу в $OUT_DIR/layout_surya_готовое/{год}/{выпуск}/{основа}.pkl — та же
# структура, что у кэша по заострённым копиям (run_layout_pack.sh), только другой каталог.
#
# Диск здесь узкое место: 38 МБ на полосу, поэтому потоков чтения три, а не четыре.
# Результаты НЕ пишутся в /mnt/dump3 — его синхронит Яндекс.Диск.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SOURCE_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/Готовое/пак-1"

uv run python -m research.legacy.table_processing layout-pack \
    --sharpened-dir "$SOURCE_DIR" \
    --out-dir "$OUT_DIR/layout_surya_готовое" \
    --ext tif \
    --source-dpi 600 \
    --readers 3 \
    --mode "${1:-all}" \
    --jobs "$JOBS" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
wait "$PID"
