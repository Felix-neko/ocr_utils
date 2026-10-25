#!/usr/bin/env bash
#
# scan_cropping v2 с рваными внешними краями страниц: двойная грубая сегментация (--dual-levels: SAM 3 без растяжения
# уровней — база, у внешних кромок книги — проход по кадру с растянутыми уровнями; промахи прохода — зона на базе),
# детектор рваной стороны по сырой маске SAM 3 (--ragged-threshold, извилистость к сплайну) и тонкое уточнение рваных
# сторон (--refine-ragged: лента вдоль стороны в полном разрешении, SAM 3 и SAM 2.1 — сокращённый набор способов, при
# неудаче полный; способ годен при покрытии ≥ 90 %, рваности ≤ 0,3 мм, ступени ≤ 1 мм, без текста за краем).
# Три набора, итог и оверлеи — $RAGGED_EDGES_WORK/<набор>/{output,debug}, кэш — своя копия $RAGGED_EDGES_WORK/<набор>/cache
# (research.ragged_edges copy-cache: пальцы, книга, фон SAM 2 — из кэшей прежних прогонов, остальное заново):
#   set — тестовый набор ($RAGGED_EDGES_SET: 210 кадров stack_edges/refine_set — 190 рваных и 20 ровных сторон ПХ 1975
#       и ХиП-1991 по ragged.csv — плюс примеры пользователя research/ragged_edges/sets/user_examples.txt);
#   hp_1991 — «Хозяйство и право» 1991, 966 кадров, плотность по выпускам (run_v2_hp_1991.sh);
#   ph_1975 — пак-4 ПХ 1975, 1027 кадров.
# Порог 1,02: на сырой маске после двойной сегментации рваные стороны пользователя 01/IMG_0031 L и 0035 L — 1,020 и
# 1,037, ровные — до 1,005 (пробный прогон ragged_edges/smoke, 2026-10-22); до калибровки по разметке поясов.
# Ключи — боевые пака-6 + осветление тени пальца по карте, как у spline_frames.
# Читает: Я.Диск (только чтение), кэши на SSD. Пишет: $RAGGED_EDGES_WORK на SSD hotstore.
# Время: +1 проход SAM 3 на кадр (~0,7 с) и ~8 с на рваную сторону (пробный прогон: 9 сторон, 74 с); LaMa пальцев
# заново (~8 с на кадр) — всего ~5–7 ч на три набора. Перезапуск безопасен (--skip-if-exists, кэш дозабирается).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

JOBS="${JOBS:-12}"
RAGGED_THRESHOLD="${RAGGED_THRESHOLD:-1.02}"

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
        --dual-levels \
        --refine-ragged \
        --ragged-threshold "$RAGGED_THRESHOLD" \
        --skip-if-exists \
        --jobs "$JOBS" \
        --log-level INFO
}

# Копия кэша без шагов, которые прогон считает заново (идемпотентно: готовые файлы не перезаписываются).
copy_cache() {
    uv run python -m research.ragged_edges copy-cache --src "$1" --dst "$2" "${@:3}"
}

declare -A HP_PX_PER_MM=(
    [01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2 [05]=17.0 [06]=17.3
    [07]=16.9 [08]=16.6 [09]=16.6 [10]=16.6 [11]=16.3 [12]=16.3
)

# Тестовый набор: ПХ 1975 (кэш пака-4) и ХиП-1991 по выпускам.
copy_cache "$PACK4_PH_WORK/cache/1975" "$RAGGED_EDGES_WORK/set/cache/ph1975" --input-dir "$RAGGED_EDGES_SET/ph1975"
run_v2 "$RAGGED_EDGES_SET/ph1975" "$RAGGED_EDGES_WORK/set/cache/ph1975" "$RAGGED_EDGES_WORK/set/ph1975" "" ""
for DIR in "$RAGGED_EDGES_SET"/hp1991/*/; do
    NAME="$(basename "$DIR")"
    copy_cache "$HP_1991_WORK/cache/$NAME" "$RAGGED_EDGES_WORK/set/cache/hp1991/$NAME" --input-dir "$DIR"
    run_v2 "$DIR" "$RAGGED_EDGES_WORK/set/cache/hp1991/$NAME" "$RAGGED_EDGES_WORK/set/hp1991/$NAME" \
        "${HP_PX_PER_MM[${NAME:0:2}]}" 434
done

for DIR in "$HP_1991_ROOT"/*/; do
    NAME="$(basename "$DIR")"
    copy_cache "$HP_1991_WORK/cache/$NAME" "$RAGGED_EDGES_WORK/hp_1991/cache/$NAME"
    run_v2 "$DIR" "$RAGGED_EDGES_WORK/hp_1991/cache/$NAME" "$RAGGED_EDGES_WORK/hp_1991/$NAME" \
        "${HP_PX_PER_MM[${NAME:0:2}]}" 434
done

copy_cache "$PACK4_PH_WORK/cache/1975" "$RAGGED_EDGES_WORK/ph_1975/cache"
run_v2 "$PACK4_PH_ROOT/1975" "$RAGGED_EDGES_WORK/ph_1975/cache" "$RAGGED_EDGES_WORK/ph_1975" "" ""
