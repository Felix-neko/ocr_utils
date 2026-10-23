#!/usr/bin/env bash
#
# Стенд research/page_cutting, этапы 2–3: криволинейная рамка страниц по маске v2 (варианты quad — трапеция v2, poly —
# полиномы по сторонам, spline — сглаживающие B-сплайны), выпрямление в прямоугольник (гомография по углам + Coons-патч
# по кривым), прямизна строк выпрямленной страницы по осям page_layout.text_blocks; оверлеи рамок и листы
# «quad | poly | spline» по странице.
# Наборы: research/page_cutting/sets/{tight_gutter,curved,ragged,flat}.txt (~40 кадров).
# Прогоны и px/мм — как в run_page_cutting_select.sh.
# Читает: кэш и output прогонов v2 (SSD). Пишет: $PAGE_CUTTING_ROOT/fit/<RUN_NAME>/{fit.csv,overlays,compare,pages}.
# Время: ~30–40 с на кадр на воркер (3 варианта × 2 страницы × разбор строк), ~3 мин на 8 воркерах.
# JOBS=8: remap полноразмерной страницы и разбор строк — до ~3 ГБ ОЗУ на воркер.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-8}"
OUT="$PAGE_CUTTING_ROOT/fit/${RUN_NAME:-v1}"
SETS_DIR=research/page_cutting/sets
RUNS=(
    --run "pack6=$PACK6_V6_WORK,,13.8"
    --run "ph4=$PACK4_PH_WORK,1975,13.8"
    --run "hp91=$HP_1991_WORK,02 оплачено,17.1"
)

uv run python -m research.page_cutting fit "${RUNS[@]}" --out-dir "$OUT" --jobs "$JOBS" ${SAVE_PAGES:+--save-pages} \
    --set "$SETS_DIR/tight_gutter.txt" --set "$SETS_DIR/curved.txt" --set "$SETS_DIR/ragged.txt" --set "$SETS_DIR/flat.txt"
