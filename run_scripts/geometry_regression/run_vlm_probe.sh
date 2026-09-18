#!/usr/bin/env bash
#
# Пробник VLM: может ли DeepSeek V4.1 Flash сам увидеть, что коррекция геометрии сделала хуже.
# Выборка — эталон плюс по 30 случайных страниц из поясов score классического детектора;
# четыре варианта картинки × 2 повтора ≈ 1050 запросов ≈ $0.3 (0.02–0.04 ¢/запрос по замеру).
#
# Читает: $GEOMETRY_RUN_DIR/{metrics.csv,cache/}, PDF из $GEO_PDF_DIR/$NOGEO_PDF_DIR.
# Пишет: $GEOMETRY_RUN_DIR/vlm/{sample.csv,<вариант>/*.json}, vlm_report.md.
# Ключ — $OPENROUTER_API_KEY. Повторный запуск не платит за готовое (--skip-done).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

JOBS=4  # сеть: лимиты провайдера, не CPU

uv run python -m research.geometry_regression vlm-probe \
    --out-dir "$GEOMETRY_RUN_DIR" \
    --labels "$GEOMETRY_LABELS" \
    --per-belt 30 --repeats 2 --jobs "$JOBS" --budget-usd 2 \
    "$@"

uv run python -m research.geometry_regression vlm-report \
    --out-dir "$GEOMETRY_RUN_DIR" \
    --labels "$GEOMETRY_LABELS" \
    --md-report "$GEOMETRY_RUN_DIR/vlm_report.md"
