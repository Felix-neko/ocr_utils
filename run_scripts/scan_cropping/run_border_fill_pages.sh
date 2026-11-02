#!/usr/bin/env bash
#
# research/border_fill pages: страницы второго преобразования v2 по итогам прогона run_v2_gutter_decompression.sh
# без правки складки (ХиП 1991, выпуски 01–07 — дальше прежний прогон не дошёл). На каждой странице: площадь повтора
# концов строк у корешка (складка карты, rectify/pages.unfolded) и варианты опоры закраса — A боевой, B без краски по
# Сауволе (окно 3 мм, k 0.2), C без тёмного на 25 уровней относительно бумаги участка, D строже пороги (6/12 по L);
# меры: кляксы Сауволы в полосе 3 мм закраса и шлейф ΔE > 15. Повод: ХиП 1991/01 IMG_0982 L (2026-10-25).
# Читает: $GUTTER_DECOMPRESSION_ROOT/hp_1991/{output,cache}, кадры и сайдкары $HP_1991_ROOT (только чтение).
# Пишет: $BORDER_FILL_PAGES_ROOT/hp_1991/{pages.csv,pages/,sheets_pages_*.jpg}.
# Время: кадр ~25 с CPU (три прохода поля + 5 закрасов на страницу); 559 кадров при JOBS=6 — ~40 мин.
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-6}"  # рядом идёт прогон v2 на 12 воркерах; кадр держит ~3 ГБ ОЗУ
RUN="$GUTTER_DECOMPRESSION_ROOT/hp_1991"
OUT="$BORDER_FILL_PAGES_ROOT/hp_1991"
# Плотность ХиП 1991 по выпускам, px/мм (как в run_v2_gutter_decompression.sh).
PX="01=17.9,02=17.1,03=17.3,04=17.2,05=17.0,06=17.3,07=16.9,08=16.6,09=16.6,10=16.6,11=16.3,12=16.3"

uv run python -m research.border_fill pages --run "$RUN" --photos "$HP_1991_ROOT" --px-per-mm "$PX" \
    --out-dir "$OUT" --jobs "$JOBS"
for BY in fold blot trail; do
    uv run python -m research.border_fill pages-sheets --pages-dir "$OUT" --by "$BY" --top 30
done
