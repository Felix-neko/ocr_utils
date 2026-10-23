#!/usr/bin/env bash
#
# Стенд research/page_rectify по итогам прогона v2 «spline_frames» (run_v2_spline_bends.sh, 2026-10-22): подгоночная
# область каждой страницы (верх, низ, внешняя сторона — сплайны, у корешка — линия корешка) → прямоугольник
# поверхностью Кунса, поля — продолжением карты. Области в кэше не хранятся — пересчитываются по кэшу v2 тем же кодом,
# что оверлей 07_lines; кадр, где рамка вырезки по кэшу не совпала с итогом (кэш перезаписан), копируется как есть.
# Наборы — те же три: ph_gutter_strip (23 кадра), hp_1991 (12 выпусков, 966 кадров), ph_1975 (1027 кадров).
#
# SAMPLE=N — по N кадров на набор равномерно по выпускам (ХиП — по N/12 на выпуск, не меньше 2), выход
# $SPLINE_RECTIFIED_SAMPLE_WORK; без SAMPLE — все кадры в $SPLINE_RECTIFIED_WORK.
# Внутри — одна output/ и одна debug/ с подпапками наборов (ph_gutter_strip, ph_1975, hp_1991/<выпуск>), сводки —
# summary/<набор>.csv (ХиП — summary/hp_1991/<выпуск>.csv). Листы «было | стало» — в debug.
# Читает: Я.Диск (оригиналы, только чтение), кэш и итоги v2 на SSD. Пишет: SSD hotstore.
# JOBS=12, не 16: кадр читает оригинал с Я.Диска (NTFS-3G) и держит ~1 ГБ карт и копий — упор в диск и память.
# Время: кадр ~5 с (пересчёт областей ~4 с, карта и запись ~1 с); 2016 кадров на 12 воркерах — ~15–20 мин.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-12}"
SAMPLE="${SAMPLE:-0}"
if [[ "$SAMPLE" -gt 0 ]]; then
    OUT_ROOT="${OUT_ROOT:-$SPLINE_RECTIFIED_SAMPLE_WORK}"
    HP_SAMPLE=$(( SAMPLE / 12 > 2 ? SAMPLE / 12 : 2 ))
else
    OUT_ROOT="${OUT_ROOT:-$SPLINE_RECTIFIED_WORK}"
    HP_SAMPLE=0
fi

# Один набор: исходные кадры, кэш v2, итоги v2, подпуть набора в output/ и debug/ (он же подпись), сколько кадров (0 — все).
rectify() {
    local input="$1" cache="$2" v2="$3" sub="$4" sample="$5"
    mkdir -p "$OUT_ROOT/summary/$(dirname "$sub")"
    echo "=== page_rectify: $v2 → $OUT_ROOT/{output,debug}/$sub"
    uv run python -m research.page_rectify run \
        --input-dir "$input" \
        --cache "$cache" \
        --v2-output "$v2" \
        --output "$OUT_ROOT/output/$sub" \
        --debug "$OUT_ROOT/debug/$sub" \
        --summary "$OUT_ROOT/summary/$sub.csv" \
        --label "$sub" \
        --sample "$sample" \
        --skip-if-exists \
        --jobs "$JOBS"
}

rectify "$PH_GUTTER_STRIP_SET" "$PH_GUTTER_STRIP_WORK/cache" "$SPLINE_FRAMES_WORK/ph_gutter_strip/output" \
    "ph_gutter_strip" "$SAMPLE"

for DIR in "$HP_1991_ROOT"/*/; do
    NAME="$(basename "$DIR")"
    rectify "$DIR" "$HP_1991_WORK/cache/$NAME" "$SPLINE_FRAMES_WORK/hp_1991/$NAME/output" "hp_1991/$NAME" "$HP_SAMPLE"
done

rectify "$PACK4_PH_ROOT/1975" "$PACK4_PH_WORK/cache/1975" "$SPLINE_FRAMES_WORK/ph_1975/output" "ph_1975" "$SAMPLE"
