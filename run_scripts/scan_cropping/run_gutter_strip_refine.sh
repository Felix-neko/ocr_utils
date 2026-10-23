#!/usr/bin/env bash
#
# Стенд research/gutter_strip, уточняющая сегментация ленты у корешка (команда refine, GPU в главном процессе):
# по кадрам множеств лента ±25 мм вдоль разделителя v2 в полном разрешении (исходный кадр с Я.Диска — только чтение),
# способы — SAM 2.1 по точкам страниц (×4, ×4 lvL), SAM 3 «book page» (×4), пара прямых ступеней, сплошная кромка
# грубого SAM 3 с поиском наклонной линии (solid); при неудаче — полный набор. Края: гладкие куски без объездов (доля
# строк в кусках ≥ 0.6), выбор по согласию сетей, вытеснение сильной линией ближе к своей странице; полоса — ≥ 5 мм.
# Числа (research/gutter_strip/refine.py, refine_region.py; стартовые, по этим множествам 2026-10-23):
#   STRIP_WIDTH_MM 5 (полосы ПХ от 5.1, без полосы до 4.4), SUPPORT_STRONG_L 10, JUMP_MM 1.5, PIECE_MIN_MM 20,
#   DETOUR_MM 2, INK_OUT_MM2 10 (вытеснение — 5), INK_FROM_EDGE_MM 2.5; область: ROW_MIN_MM 3, RUN_MIN_MM 30,
#   END_SNAP_MM 8.
# Множества: sets/refine_ph1975.txt (107 кадров: 41 полоса, 33 спорных и неразмеченных подозрительных, 33 без полосы),
#   sets/refine_hp1991.txt (27 кадров ХиП с сильным загибом у корешка, полос нет).
# Читает: $RAGGED_EDGES_V2_WORK/{ph_1975,hp_1991}/cache (SSD), исходники $PACK4_PH_ROOT/1975, $HP_1991_ROOT.
# Пишет: $GUTTER_STRIP_ROOT/refine/{ph1975,hp1991}/{refine.csv,json,npz,band,strip,none,fail}.
# Затем без GPU: rechoose (края — гладкие куски без объездов) и region-fit (полоса-область, страницы SAM 3 по уточнённым
# краям, линия корешка, рамки) → $GUTTER_STRIP_ROOT/refine/{ph1975,hp1991}_{rc,region}; ~2 мин.
# Время: проба 2026-10-23 — ~4 с на кадр на сокращённом наборе, ~18 с с полным; на всё ~15 мин.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

OUT="$GUTTER_STRIP_ROOT/refine"

uv run python -m research.gutter_strip refine --cache "$RAGGED_EDGES_V2_WORK/ph_1975/cache" \
    --src-root "$PACK4_PH_ROOT/1975" --set research/gutter_strip/sets/refine_ph1975.txt --out "$OUT/ph1975"
uv run python -m research.gutter_strip refine --cache "$RAGGED_EDGES_V2_WORK/hp_1991/cache" \
    --src-root "$HP_1991_ROOT" --set research/gutter_strip/sets/refine_hp1991.txt --out "$OUT/hp1991"
for SET in ph1975:"$RAGGED_EDGES_V2_WORK/ph_1975/cache":"$PACK4_PH_ROOT/1975" hp1991:"$RAGGED_EDGES_V2_WORK/hp_1991/cache":"$HP_1991_ROOT"; do
    NAME=${SET%%:*}; REST=${SET#*:}; CACHE=${REST%%:*}; SRC=${REST#*:}
    uv run python -m research.gutter_strip rechoose --cache "$CACHE" --src-root "$SRC" \
        --set "research/gutter_strip/sets/refine_$NAME.txt" --refine-dir "$OUT/$NAME" --out "$OUT/${NAME}_rc"
    uv run python -m research.gutter_strip region-fit --cache "$CACHE" --set "research/gutter_strip/sets/refine_$NAME.txt" \
        --refine-dir "$OUT/${NAME}_rc" --out "$OUT/${NAME}_region"
done
