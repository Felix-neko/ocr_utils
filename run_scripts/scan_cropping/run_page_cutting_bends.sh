#!/usr/bin/env bash
#
# Стенд research/page_cutting, изгибы верха и низа страниц вдали от корешка: прежний сглаживающий сплайн
# (scan_cropping/v2/page_boundary, 8 отрезков, узлы гуще только у корешка) против нового (ocr_utils/scan_cropping/v2/page_boundary_adaptive.py):
# тот же сплайн плюс не больше 2 узлов (не ближе 15 мм друг к другу) на пологих изломах края (медиана остатка в окне 10 мм от 0.3 до 1.5 мм), штраф
# у узла — 0.2 прежнего. Без узлов кривая — ровно прежняя. Боковые стороны — прежние. Вход — как у шага 07_lines v2
# (маска и трапеция v2, без сгиба), только CPU.
# Запреты для узлов (числа — подбор 2026-10-22 по кадрам, на которые указал пользователь):
#   дальний угол — 30 мм (углы страниц не скруглены; ПХ 1975/02 IMG_0038), зона корешка + 5 мм (излом низа бывает в
#   30–50 мм от корешка — ХиП 1991/01 IMG_1052); яма маски (контур дальше 3 мм от прежнего сплайна) — 25 мм вокруг, яма
#   длиннее 5 мм — вся сторона (IMG_0038); узел — только на сплошном участке точек от 15 мм (островок между зонами
#   пальца — ХиП 1991/02 IMG_1144); дальше 20 мм от узла кривая без точек и в зоне корешка сдвигается не больше 0.2 мм
#   (МТС 1991/05 IMG_0250), по точкам ложится не хуже (p90 остатка +0.05 мм).
#   Точки верха и низа в зоне закраса пальца годны, кроме ядра пальца с припуском 3 мм — но только для поиска излома и
#   подгонки с узлами (горб низа с вершиной под пальцем, ХиП 1991/01 IMG_1052, 1991/02 IMG_1143).
# Два прохода:
#   sets — наборы стенда (tight_gutter, curved, ragged, flat) по трём прогонам, как run_page_cutting_fit.sh;
#   hp_1991 — все выпуски «Хозяйство и право» 1991, какие есть в кэше run_v2_hp_1991.sh (px/мм — оттуда же), оверлеи
#       только кадров с расхождением от 0.15 мм, по папкам поясов (0.15-0.3, 0.3-0.5, ge0.5).
# Читает: кэш прогонов v2 (SSD). Пишет: $PAGE_CUTTING_ROOT/bends/{sets,hp_1991}/{bends.csv,overlays/<пояс>/}.
# Время: ~0.2 с на кадр на воркер; 85 кадров выпуска — 20 с на 16 воркерах, год (~970 кадров) — ~3–4 мин.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-16}"
OUT="$PAGE_CUTTING_ROOT/bends"
SETS_DIR=research/page_cutting/sets

uv run python -m research.page_cutting bends --out-dir "$OUT/sets" --jobs "$JOBS" \
    --run "pack6=$PACK6_V6_WORK,,13.8" --run "ph4=$PACK4_PH_WORK,1975,13.8" --run "hp91=$HP_1991_WORK,02 оплачено,17.1" \
    --set "$SETS_DIR/tight_gutter.txt" --set "$SETS_DIR/curved.txt" --set "$SETS_DIR/ragged.txt" --set "$SETS_DIR/flat.txt"

# Плотность бумаги по выпускам, px/мм (run_v2_hp_1991.sh).
declare -A PX_PER_MM=(
    [01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2 [05]=17.0 [06]=17.3
    [07]=16.9 [08]=16.6 [09]=16.6 [10]=16.6 [11]=16.3 [12]=16.3
)
RUNS=()
for DIR in "$HP_1991_WORK"/cache/*/; do
    NAME="$(basename "$DIR")"
    ISSUE="${NAME:0:2}"
    RUNS+=(--run "hp91_$ISSUE=$HP_1991_WORK,$NAME,${PX_PER_MM[$ISSUE]}")
done
uv run python -m research.page_cutting bends --out-dir "$OUT/hp_1991" --jobs "$JOBS" --overlay-min-mm 0.15 "${RUNS[@]}"
