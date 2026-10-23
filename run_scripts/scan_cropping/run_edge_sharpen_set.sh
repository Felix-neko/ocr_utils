#!/usr/bin/env bash
#
# Стенд research/edge_sharpen: фильтры кадра сегментации (нерезкая маска 0.5 и 2 мм, CLAHE по L, detailEnhance,
# bilateral + нерезкая маска, усиление морфологического градиента, гамма 1.8) перед SAM 3 на кадрах, где край верхней
# страницы сливается с торцами нижних листов; бумага и страницы — как cpu_pages v2 (--fit-sides, --no-snap-sides) в
# двух режимах: фильтр только во вход SAM 3 (sam) и также в доводку края (all). Затем меры края и листы вырезок.
# Наборы: research/edge_sharpen/sets/{ragged,control}.txt (23 кадра). Плотность — 13.8 px/мм у всех трёх паков
# (run_v2_ph_pack4_1975.sh, v2_pack6_args.sh, run_v2_pack5_1990_q1.sh); масштаб сегментации — 6.9 px/мм.
# Читает: кэш прогонов v2 (seg.png, seg_noink.png, book.npz, fingers.npz, paper.npz, sam3.npz) на SSD.
# Пишет: $EDGE_SHARPEN_ROOT/{<фильтр>/<прогон>/<ключ>/,measures.csv,sheets/}.
# Время: SAM 3 ×2 прохода ~2 с на кадр × 8 фильтров × 23 кадра ≈ 7 мин (GPU, последовательно); CPU — ~1 мин.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-16}"
OUT="$EDGE_SHARPEN_ROOT"
SETS=(--set research/edge_sharpen/sets/ragged.txt --set research/edge_sharpen/sets/control.txt)
RUNS=(
    --run "ph4=$PACK4_PH_WORK,1975,13.8"
    --run "pack6=$PACK6_V6_WORK,,13.8"
    --run "p5a=$PACK5_1990_Q1_WORK,1990/01,13.8"
    --run "p5b=$PACK5_1990_Q1_WORK,1990/02,13.8"
    --run "p5c=$PACK5_1990_Q1_WORK,1990/03,13.8"
)

mkdir -p "$OUT"
uv run python -m research.edge_sharpen run "${RUNS[@]}" "${SETS[@]}" --out-dir "$OUT" --jobs "$JOBS"
uv run python -m research.edge_sharpen measure "${RUNS[@]}" "${SETS[@]}" --out-dir "$OUT" --jobs "$JOBS"
for mode in sam all; do
    uv run python -m research.edge_sharpen sheets "${RUNS[@]}" "${SETS[@]}" --out-dir "$OUT" --mode "$mode"
done
