#!/usr/bin/env bash
#
# Стенд research/gutter_cut: мера «текст впритык к линии разреза у корешка» по холсту Кунса кэшей прогонов v2 и полосы
# у корешка с предлагаемой уточнённой линией разреза; листы по вердиктам.
# Наборы: ХиП 1991 01–04 (прогон $GUTTER_DECOMPRESSION_TABLES_ROOT, плотность по выпускам — как в
# run_v2_gutter_decompression.sh), пак-6 (кэш и итоги $PACK6_450_WORK, 13.8 px/мм; N_PACK6 кадров равномерно),
# пак-7 МТС (прогон run_v2_pack7_gutter.sh, 12.2 px/мм), если он есть.
# Читает: Я.Диск (кадры, только чтение), кэши и итоги v2 на SSD. Пишет: $GUTTER_CUT_ROOT/measure_<метка>/.
# Время: ~25 с CPU на кадр, пул $JOBS.
# Запуск: run_gutter_cut_measure.sh [метка]; ONLY_PACK7=1 — только пак-7 (перемер после его прогона v2).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

TAG="${1:-$(date +%Y-%m-%d)}"
JOBS="${JOBS:-6}"  # 6 — параллельно идёт GPU-прогон v2 по паку-7 (его выпрямление держит 9 воркеров)
N_PACK6="${N_PACK6:-400}"
HP="$GUTTER_DECOMPRESSION_TABLES_ROOT/hp_1991"
P7="$GUTTER_CUT_ROOT/pack7"
PACK7_MTS_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-7 от Бори/МТС"
declare -A HP_PX_PER_MM=([01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2)

SETS=()
ONLY_PACK7="${ONLY_PACK7:-0}"
[[ "$ONLY_PACK7" == 1 ]] || for NAME in "01 оплачено" "02 оплачено" "03 оплачено" "04 оплачено"; do
    SETS+=(--set "hp_${NAME:0:2}" "$HP_1991_ROOT/$NAME" "$HP/cache/$NAME" "$HP/output/$NAME" "${HP_PX_PER_MM[${NAME:0:2}]}" 0)
done
[[ "$ONLY_PACK7" == 1 ]] || SETS+=(--set pack6 "$PACK6_450_ROOT" "$PACK6_450_WORK/cache" "$PACK6_450_WORK/output" 13.8 "$N_PACK6")
for ISSUE in 1991/05 1991/06 1990/07; do
    SUB="${ISSUE//\//__}"
    if [[ -d "$P7/output/$SUB" ]]; then
        SETS+=(--set "pack7_$SUB" "$PACK7_MTS_ROOT/$ISSUE" "$P7/cache/$SUB" "$P7/output/$SUB" 12.2 0)
    fi
done

OUT="$GUTTER_CUT_ROOT/measure_$TAG"
uv run python -m research.gutter_cut measure "${SETS[@]}" --root "$OUT" --jobs "$JOBS"
uv run python -m research.gutter_cut sheets --root "$OUT"
