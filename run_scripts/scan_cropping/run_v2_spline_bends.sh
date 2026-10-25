#!/usr/bin/env bash
#
# scan_cropping v2 с подгоночными областями страниц на шаге 07_lines (--debug-steps): прежний B-сплайн (page_boundary)
# для сравнения и новая область (page_frames: внешняя сторона, верх и низ — сплайн с 1–2 узлами на изгибах, у корешка —
# линия корешка; у полосы-вклейки — до вспомогательных линий, через полосу — отрезок; полоса — gutter_strip).
# Три набора, у каждого итог и оверлеи — в своей папке $OUT_ROOT/<набор>/{output,debug} (по умолчанию
# $SPLINE_FRAMES_WORK; прежний прогон без полосы — $SPLINE_BENDS_WORK); кэш шагов — прежних
# прогонов (пальцы, SAM 3, SAM 2, surya от рамок не зависят, готовое берётся с диска):
#   ph_gutter_strip — тестовый набор ПХ 1975 с проклейкой у корешка и без (23 кадра, $PH_GUTTER_STRIP_SET),
#       кэш $PH_GUTTER_STRIP_WORK/cache;
#   hp_1991 — «Хозяйство и право» 1991, 12 выпусков (966 кадров), кэш $HP_1991_WORK/cache/<выпуск>; плотность бумаги
#       по выпускам — как в run_v2_hp_1991.sh (замер 2026-10-21);
#   ph_1975 — пак-4 ПХ 1975 целиком (1027 кадров), кэш $PACK4_PH_WORK/cache/1975.
# Ключи — боевые пака-6 (v2_pack6_args.sh) + осветление тени пальца по карте, как у прежних прогонов этих наборов.
# Читает: Я.Диск (только чтение) и кэши на SSD. Пишет: $OUT_ROOT на SSD hotstore.
# Время: кадр с готовым кэшем ~4–6 с (LaMa пальцев заново: кадр без пальцев в кэше не хранится), без кэша ~11 с;
# всего ~3–4 ч. Перезапуск безопасен (--skip-if-exists).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

JOBS="${JOBS:-12}"
OUT_ROOT="${OUT_ROOT:-$SPLINE_FRAMES_WORK}"

# Один прогон v2: вход, кэш, папка итога и оверлеев, плотность бумаги (пусто — из V2_PACK6_ARGS), dpi итога.
run_v2() {
    local input="$1" cache="$2" out="$3" px="$4" dpi="$5"
    local extra=()
    [[ -n "$px" ]] && extra+=(--px-per-mm "$px")
    [[ -n "$dpi" ]] && extra+=(--force-dpi "$dpi")
    mkdir -p "$out"
    echo "=== scan_cropping v2: $input → $out"
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$input" \
        --output "$out/output" \
        --cache "$cache" \
        --debug "$out/debug" \
        --debug-steps \
        "${V2_PACK6_ARGS[@]}" \
        "${extra[@]}" \
        --finger-zone-light-increment 0,0 \
        --finger-shadow \
        --jobs "$JOBS" \
        --log-level INFO
}

run_v2 "$PH_GUTTER_STRIP_SET" "$PH_GUTTER_STRIP_WORK/cache" "$OUT_ROOT/ph_gutter_strip" "" ""

# ХиП 1991: плотность бумаги по выпускам, px/мм (run_v2_hp_1991.sh).
declare -A HP_PX_PER_MM=(
    [01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2 [05]=17.0 [06]=17.3
    [07]=16.9 [08]=16.6 [09]=16.6 [10]=16.6 [11]=16.3 [12]=16.3
)
for DIR in "$HP_1991_ROOT"/*/; do
    NAME="$(basename "$DIR")"
    run_v2 "$DIR" "$HP_1991_WORK/cache/$NAME" "$OUT_ROOT/hp_1991/$NAME" "${HP_PX_PER_MM[${NAME:0:2}]}" 434
done

run_v2 "$PACK4_PH_ROOT/1975" "$PACK4_PH_WORK/cache/1975" "$OUT_ROOT/ph_1975" "" ""
