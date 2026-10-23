#!/usr/bin/env bash
#
# scan_cropping v2 по «Хозяйство и право» 1991 целиком (12 выпусков, 966 фото-разворотов JPG на чёрном коврике,
# пальцы) — по выпуску за вызов, у каждого своя плотность бумаги. Ключи — как run_v2_hp_1991_02.sh (боевые пака-6
# + осветление тени пальца по карте) и картинки шагов сегментации (--debug-steps: 01_book … 07_lines).
# Числа:
#   PX_PER_MM — плотность бумаги по выпускам: камеру переставляли, размер кадра 5229×4041 … 5813×4419 при одном
#       теге 434 dpi. Замер 2026-10-21 (ai_slop/hp1991/{pitch,page_height}.py) относительно выпуска 02 = 17.1 px/мм
#       (рамка книги, run_v2_hp_1991_02.sh) двумя способами — шаг строк основного текста (12 кадров на выпуск) и
#       высота светлого блока страниц на коврике (25 кадров); в таблице — среднее двух, расхождение способов 2–6 %.
#   --force-dpi 434 — dpi в метаданных итога (EXIF кадров).
#
# Читает: $HP_1991_ROOT/<выпуск>/ (Я.Диск, только чтение).
# Пишет: $HP_1991_WORK/{output,cache,debug}/<выпуск>/ на SSD hotstore.
# Время: ~10–11 с на кадр (упор в GPU), ~2.5–3 ч на год; видеопамять — пик ~8.5 ГБ (с SAM 2 вето фона).
# Перезапуск безопасен: готовые итоги пропускаются (--skip-if-exists), кэш шагов берётся с диска.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

JOBS="${JOBS:-12}"

# Плотность бумаги по выпускам, px/мм (см. шапку).
declare -A PX_PER_MM=(
    [01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2 [05]=17.0 [06]=17.3
    [07]=16.9 [08]=16.6 [09]=16.6 [10]=16.6 [11]=16.3 [12]=16.3
)

for DIR in "$HP_1991_ROOT"/*/; do
    NAME="$(basename "$DIR")"
    ISSUE="${NAME:0:2}"
    PX="${PX_PER_MM[$ISSUE]}"
    echo "=== scan_cropping v2: 1991/$NAME ($PX px/мм) → $HP_1991_WORK/output/$NAME"
    # Ключи после V2_PACK6_ARGS перекрывают одноимённые оттуда (click берёт последнее значение).
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$DIR" \
        --output "$HP_1991_WORK/output/$NAME" \
        --cache "$HP_1991_WORK/cache/$NAME" \
        --debug "$HP_1991_WORK/debug/$NAME" \
        --debug-steps \
        "${V2_PACK6_ARGS[@]}" \
        --px-per-mm "$PX" \
        --force-dpi 434 \
        --finger-zone-light-increment 0,0 \
        --finger-shadow \
        --jobs "$JOBS" \
        --log-level INFO
done
