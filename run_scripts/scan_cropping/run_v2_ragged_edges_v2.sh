#!/usr/bin/env bash
#
# Перепрогон scan_cropping v2 по тем же трём наборам, что run_v2_ragged_edges.sh, после правок 2026-10-23:
#   - уточнение рваной стороны: строка «под пальцем» — только где ядро пальца у грубого края (±3 мм, FINGER_EDGE_MM);
#     ладонь снаружи книги строк не выбрасывает (ХиП 07/IMG_0133 R: 134 → 38 мм стороны); пропуск края длиннее 15 мм и
#     у конца стороны — грубым краем со швом 5 мм вместо прямой (FILL_LINEAR_MM, FILL_BLEND_MM);
#   - полоса-вклейка: щель SAM 3 до связной части страницы (клочки бумаги на полосе не сужают, ПХ 03/IMG_0004 — 3.6 →
#     12.2 мм), порог 8.2 мм вместо 7.8 (разметка ПХ 1975: 32 полосы из 37, ложных нет);
#   - закрас пальцев в 3 потоках (--finger-threads), сети под общей блокировкой GPU.
# Кэш: копия кэша прогона ragged_edges (research.ragged_edges copy-cache --keep sam3.npz: пальцы, книга, фон SAM 2,
# оба прохода SAM 3 — как есть; двойная сегментация, детектор, уточнение, страницы, surya, финал — заново). Кадр без
# пальцев clean.png после финала удаляется, поэтому закрас пальцев считается заново (LaMa детерминирована: fingers.npz
# совпадает с прежним — проверка finger_speed, 2026-10-23).
# Итог и оверлеи — $RAGGED_EDGES_V2_WORK/<набор>/{output,debug}; база для сравнения — $RAGGED_EDGES_WORK (не трогается).
# Читает: Я.Диск (только чтение), кэши на SSD. Пишет: $RAGGED_EDGES_V2_WORK на SSD hotstore.
# Время: прогон 1 — 11 ч 46 мин на 2206 кадров (~19 с на кадр, упор в закрас пальцев в главном потоке); здесь SAM 3 из
# кэша и пальцы в потоках — оценка по проверке finger_speed.
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

JOBS="${JOBS:-12}"
FINGER_THREADS="${FINGER_THREADS:-3}"
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
        --finger-threads "$FINGER_THREADS" \
        --log-level INFO
}

# Копия кэша прогона ragged_edges с SAM 3 (идемпотентно: готовые файлы не перезаписываются).
copy_cache() {
    uv run python -m research.ragged_edges copy-cache --src "$1" --dst "$2" --keep sam3.npz "${@:3}"
}

declare -A HP_PX_PER_MM=(
    [01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2 [05]=17.0 [06]=17.3
    [07]=16.9 [08]=16.6 [09]=16.6 [10]=16.6 [11]=16.3 [12]=16.3
)

# Тестовый набор: ПХ 1975 и ХиП-1991 по выпускам.
copy_cache "$RAGGED_EDGES_WORK/set/cache/ph1975" "$RAGGED_EDGES_V2_WORK/set/cache/ph1975"
run_v2 "$RAGGED_EDGES_SET/ph1975" "$RAGGED_EDGES_V2_WORK/set/cache/ph1975" "$RAGGED_EDGES_V2_WORK/set/ph1975" "" ""
for DIR in "$RAGGED_EDGES_SET"/hp1991/*/; do
    NAME="$(basename "$DIR")"
    copy_cache "$RAGGED_EDGES_WORK/set/cache/hp1991/$NAME" "$RAGGED_EDGES_V2_WORK/set/cache/hp1991/$NAME"
    run_v2 "$DIR" "$RAGGED_EDGES_V2_WORK/set/cache/hp1991/$NAME" "$RAGGED_EDGES_V2_WORK/set/hp1991/$NAME" \
        "${HP_PX_PER_MM[${NAME:0:2}]}" 434
done

for DIR in "$HP_1991_ROOT"/*/; do
    NAME="$(basename "$DIR")"
    copy_cache "$RAGGED_EDGES_WORK/hp_1991/cache/$NAME" "$RAGGED_EDGES_V2_WORK/hp_1991/cache/$NAME"
    run_v2 "$DIR" "$RAGGED_EDGES_V2_WORK/hp_1991/cache/$NAME" "$RAGGED_EDGES_V2_WORK/hp_1991/$NAME" \
        "${HP_PX_PER_MM[${NAME:0:2}]}" 434
done

copy_cache "$RAGGED_EDGES_WORK/ph_1975/cache" "$RAGGED_EDGES_V2_WORK/ph_1975/cache"
run_v2 "$PACK4_PH_ROOT/1975" "$RAGGED_EDGES_V2_WORK/ph_1975/cache" "$RAGGED_EDGES_V2_WORK/ph_1975" "" ""
