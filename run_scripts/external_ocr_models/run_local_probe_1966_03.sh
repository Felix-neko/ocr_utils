#!/usr/bin/env bash
#
# Локальные специализированные OCR-движки по тем же 13 полосам пробника, что и облачные
# модели. Каждый движок — своё окружение в research/external_ocr_models/local/<движок>
# (uv sync --project … при первом запуске воркера делается сам).
#
# ВЫХОД — EXTERNAL_OCR_PROBE_ROOT/local-<движок>/1966/03/…, как у облачных.
# ВРЕМЯ — 20-35 с на полосу на RTX 5060 Ti, движки идут строго по очереди: видеопамять одна.
# Запускать в фоне и ждать ПО СОХРАНЁННОМУ PID (см. CLAUDE.md):
#
#   setsid ./run_scripts/external_ocr_models/run_local_probe_1966_03.sh > /tmp/local.log 2>&1 < /dev/null & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 30; done
set -euo pipefail
source "$(dirname "$0")/common.sh"

set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

PAGES="$(dirname "$0")/probe_pages_1966_03.txt"
ENGINES=(local-dots-ocr local-deepseek-ocr2 local-marker local-paddleocr-vl)

for engine in "${ENGINES[@]}"; do
    echo "=== $engine"
    uv run python -m research.external_ocr_models run \
        --in-dir "$SHARPENED_DIR" \
        --out-dir "$EXTERNAL_OCR_PROBE_ROOT/$engine" \
        --model "$engine" \
        --pages "$PAGES" \
        "$@" || echo "!!! $engine не завёлся, см. лог"
done
