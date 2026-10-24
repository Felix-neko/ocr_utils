#!/usr/bin/env bash
#
# Стенд research/single_page («в кадре одна страница») по кэшам прогонов v2: признаки всех кадров пяти наборов
# (features), листы миниатюр по поясам на отсмотр (sheets), сверка правила с разметкой sets/labels.txt (evaluate).
# Числа (research/single_page/rule.py, по 4 739 кадрам, 2026-10-24):
#   min_paper_h_mm 130 — бумага ниже не страница: ярлыки на крышках 53–72 мм, самая низкая страница (ХиП 1991) 195 мм.
#   max_width_ratio 0.5 — узкая/широкая трапеция: одностраничные кадры 0.07–0.39, развороты не ниже 0.714.
#   max_aspect 0.78 — ширина/высота рамки бумаги: одностраничные 0.57–0.72, развороты не ниже 0.807.
# Читает (только чтение): кэши v2 ПХ 1975, ПХ 1974/01, ХиП 1991, пак-5 1990/01–03, пак-6 v6 (SSD).
# Пишет: $SINGLE_PAGE_ROOT/{features.csv,thumbs/,sheets/,evaluate.txt}.
# Время: features ~2.5 мин на 12 воркерах (упор в чтение npz с SSD), листы — секунды. Только CPU.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-12}"
OUT="${SINGLE_PAGE_ROOT:-/mnt/hotstore/scan_processing/single_page}"
LABELS=research/single_page/sets/labels.txt

uv run python -m research.single_page features --out "$OUT" --jobs "$JOBS" \
    --cache "ph1975=$PACK4_PH_WORK/cache" \
    --cache "ph1974_01=$PH_PAGES_WORK/1974/01_оплачено/cache" \
    --cache "hp1991=$HP_1991_WORK/cache" \
    --cache "pack5=$PACK5_1990_Q1_WORK/cache" \
    --cache "pack6=$PACK6_V6_WORK/cache"
uv run python -m research.single_page evaluate --out "$OUT" --labels "$LABELS"

sheets() { uv run python -m research.single_page sheets --out "$OUT" "$@"; }
sheets --name a_print --where "source=='print'"
sheets --name b_le1page --where "n_pages<=1" --sort key
sheets --name c_inst_narrow --where "source=='instances' and width_ratio<0.6"
sheets --name d_inst_aspect --where "source=='instances' and width_ratio>=0.6 and paper_aspect<1.05" --sort paper_aspect
sheets --name e_edges --issue-edges 2 --sort key
sheets --name v_single --where "verdict=='single'" --sort key
sheets --name v_sliver --where "verdict=='sliver'" --sort key
sheets --name v_no_page --where "verdict=='no_page'" --sort key
sheets --name w_spread_border --where "verdict=='spread' and (width_ratio<0.85 or paper_aspect<1.0)" --sort width_ratio
