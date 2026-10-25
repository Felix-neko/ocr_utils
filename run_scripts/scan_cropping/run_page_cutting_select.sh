#!/usr/bin/env bash
#
# Стенд research/page_cutting, этап 1: меры всех кадров трёх прогонов v2 (рваность и изгиб сторон страниц, зазор текста у
# корешка, цветность, число блоков surya — по кэшу, без GPU), кандидаты в тестовую коллекцию по группам и листы отсмотра
# оверлеев v2 по ~500 px на половину кадра.
# Прогоны и плотность (px/мм — из их run-скриптов):
#   pack6 — $PACK6_V6_WORK (МТС и РИСК, 13.8; v2_pack6_args.sh);
#   ph4   — $PACK4_PH_WORK, подпапка 1975 (13.8; run_v2_ph_pack4_1975.sh);
#   hp91  — $HP_1991_WORK, подпапка «02 оплачено» («Хозяйство и право» 1991/02, 17.1; run_v2_hp_1991.sh — прогон с
#           вето фона; прежний $HP_1991_02_WORK — до вето, край с печатью там срезан).
# Читает: кэш и debug прогонов (SSD). Пишет: $PAGE_CUTTING_ROOT/select/{measures.csv,candidates/*.txt,sheets/*.jpg}.
# Время: ~0.5 с на кадр на воркер, ~3600 кадров — ~2–3 мин на 16 воркерах.
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-16}"
OUT="$PAGE_CUTTING_ROOT/select"
RUNS=(
    --run "pack6=$PACK6_V6_WORK,,13.8"
    --run "ph4=$PACK4_PH_WORK,1975,13.8"
    --run "hp91=$HP_1991_WORK,02 оплачено,17.1"
)

uv run python -m research.page_cutting select "${RUNS[@]}" --out-dir "$OUT" --per-run 8 --jobs "$JOBS"
SETS=()
for f in "$OUT"/candidates/*.txt; do SETS+=(--set "$f"); done
uv run python -m research.page_cutting sheets "${RUNS[@]}" --out-dir "$OUT/sheets" "${SETS[@]}"
