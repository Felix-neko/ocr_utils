#!/usr/bin/env bash
# Детектор ухода текста под корешок на размеченной папке «Плановое хозяйство» 1926/08.
# Папка размечена вручную: припиской «тугой_переплёт» в имени файла. Служит валидацией.
set -euo pipefail

PACK="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/неразобранное/2026-08-22-ПХ/1926/08 оплачено страница 246 отсутствует в подшивке + нужно пересканить часть страниц с сильным уходом под корешок"
OUT="${1:-reports/gutter_loss/ph_1926_08}"

mkdir -p "$OUT"
uv run python -m ocr_utils.gutter_loss_detection "$PACK" \
    --csv "$OUT/отчёт.csv" \
    --md-report "$OUT/отчёт.md" \
    --link-dir "$OUT/худшие" \
    --sheet "$OUT/врезки.png" \
    --count 60
