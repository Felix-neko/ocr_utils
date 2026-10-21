#!/usr/bin/env bash
#
# Стенд research/spread_lines: прямые разворота (разделитель страниц четырьмя способами и боевым auto, края книги,
# ограничители внешних краёв) по готовому кэшу прогонов v2 — без GPU. Наборы: проклейка у корешка ПХ 1975 (23 кадра), пак-4 ПХ 1975
# (1024 разворота), пак-6 v6 МТС/РИСК (2458 разворотов, пальцы у края, книга за кадром у ~30 %).
# Пороги — умолчания spread_lines.py: край у рамки — ближе 2 мм, «в кадре» — таких строк < 5 %, рука у края — 5 мм,
# рука — кожа не дальше 15 мм от закраса пальца; auto — gap_mid, при наклоне дальше 1° от quad_mid — quad_mid.
#
# Читает: кэш v2 ($PH_GUTTER_STRIP_WORK/cache, $PACK4_PH_WORK/cache/1975, $SCAN_CROPPING_V2_ROOT/pack6_v6/cache).
# Пишет: $SPREAD_LINES_ROOT/<набор>/{lines.csv,<способ>/,compare/,select/}.
# Время: ~3 мин на 1000 кадров при 16 воркерах (упор в CPU: линия сгиба, картинки).
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-16}"
declare -A SETS=(
    [gutter_strip]="$PH_GUTTER_STRIP_WORK/cache"
    [ph_pack4_1975]="$PACK4_PH_WORK/cache/1975"
    [pack6_v6]="$SCAN_CROPPING_V2_ROOT/pack6_v6/cache"
)
for name in gutter_strip ph_pack4_1975 pack6_v6; do
    out="$SPREAD_LINES_ROOT/$name"
    echo "=== $name: ${SETS[$name]} → $out"
    rm -rf "$out/select"
    uv run python -m research.spread_lines run --cache "${SETS[$name]}" --out "$out" --jobs "$JOBS"
    uv run python -m research.spread_lines select --out "$out"
    uv run python -m research.spread_lines evaluate --out "$out"
done
