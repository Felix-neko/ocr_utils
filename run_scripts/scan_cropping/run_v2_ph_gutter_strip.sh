#!/usr/bin/env bash
#
# scan_cropping v2 по набору разворотов ПХ 1975 с проклейкой у корешка ($PH_GUTTER_STRIP_SET: 11 strip, 10 neighbor,
# 2 maybe) — ключи боевого прогона пака-4 (run_v2_ph_pack4_1975.sh) и картинки шагов сегментации бумаги (--debug-steps).
#
# Читает: $PH_GUTTER_STRIP_SET/{strip,neighbor,maybe}/<выпуск>/ (симлинки на кадры пака-4 на Я.Диске, только чтение).
# Пишет: $PH_GUTTER_STRIP_WORK/{output,cache,debug}/ на SSD hotstore; оверлеи — по папкам типа в debug/:
#   main/, shadow/, surya/ и шаги 01_book, 02_sam3, 03_snap, 04_glue, 05_sides, 06_pages.
# Время: ~11 с на кадр (упор в GPU), ~5 мин на 23 кадра; кэш свежий — считается всё.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

WORK="${GUTTER_WORK:-$PH_GUTTER_STRIP_WORK}"
JOBS="${JOBS:-12}"  # как у пака-4: CPU-шаги идут параллельно GPU, кадров всего 23

mkdir -p "$WORK"
echo "=== scan_cropping v2: $PH_GUTTER_STRIP_SET → $WORK/output"
uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$PH_GUTTER_STRIP_SET" \
    --output "$WORK/output" \
    --cache "$WORK/cache" \
    --debug "$WORK/debug" \
    --debug-steps \
    "${V2_PACK6_ARGS[@]}" \
    --finger-zone-light-increment 0,0 \
    --finger-shadow \
    --jobs "$JOBS" \
    --log-level INFO
