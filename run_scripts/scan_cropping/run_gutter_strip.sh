#!/usr/bin/env bash
#
# Стенд research/gutter_strip по кэшу прогона v2 пака-4 ПХ 1975 (1024 разворота): признаки всех способов обнаружения
# полосы-вклейки у корешка (features), сравнение по разметке (evaluate), сегментация полосы, линия корешка, рамки
# страниц «было | стало» и оверлеи по размеченным кадрам (run).
# Числа (в коде стенда, research/gutter_strip/segment.py):
#   STRIP_GAP_MM 8.2 (ocr_utils/scan_cropping/v2/gutter_strip.py) — полоса есть, если медианная щель между связными
#       частями экземпляров SAM 3 у корешка не уже: по разметке sets/ph1975.txt (253 кадра, 37 полос) — 32 полосы из
#       37, ложных нет (evaluate, 2026-10-23; прежние 7.8 и щель до первого пикселя — 2026-10-22).
#   EDGE_SEARCH_MM 4 — окно поиска края полосы вокруг кромки щели SAM 3.
# Читает: $PACK4_PH_WORK/cache/1975 (SSD). Пишет: $GUTTER_STRIP_ROOT/ph1975/{features.csv,run/{run.csv,strip,none}}.
# Время: features ~7 мин на 10 воркерах, run по разметке ~1 мин. Только CPU.
# JOBS=10: прогон v2 параллельно занимает 12 воркеров.
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-10}"
CACHE="$PACK4_PH_WORK/cache/1975"
OUT="$GUTTER_STRIP_ROOT/ph1975"
LABELS=research/gutter_strip/sets/ph1975.txt

uv run python -m research.gutter_strip features --cache "$CACHE" --out "$OUT/features.csv" --jobs "$JOBS"
uv run python -m research.gutter_strip evaluate --features-csv "$OUT/features.csv" --labels "$LABELS"
uv run python -m research.gutter_strip run --cache "$CACHE" --out "$OUT/run" --labels "$LABELS" --jobs "$JOBS"
