#!/usr/bin/env bash
#
# Весь пак заострённых полос через DeepSeek V4.1 Flash по выпускам (см. run_issue_sharpened.sh).
# Идемпотентен: --skip-done пропускает готовые полосы, прерванный прогон продолжается тем же
# вызовом. Оглавления, найденные моделью вне базы, НЕ перераспознаются, а копятся в
# EXTERNAL_OCR_SERVICES_OUT/missed_toc.txt — после прогона проставить теги в CVAT
# («Оглавление» / «Годовой указатель» / вето «Не оглавление»), забрать from-cvat и прогнать
# нужные выпуски run_issue_sharpened.sh.
#
# ЧИТАЕТ  SHARPENED_DIR (144 ГиБ, SSD), DB_REVIEWED (только чтение).
# ПИШЕТ   EXTERNAL_OCR_SERVICES_OUT, EXTERNAL_OCR_SERVICES_DEBUG (≈ 7 ГиБ тайлов и ответов).
# ОРИЕНТИР: 12 135 полос, ~$12 и ~8 часов при 4 потоках (7 с на запрос в стенде).
# Запуск в фон из сессии агента:
#   setsid run_scripts/external_ocr_services/run_pack_sharpened.sh > /tmp/ocr_pack.log 2>&1 < /dev/null & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 60; done
set -euo pipefail
source "$(dirname "$0")/common.sh"

uv run python -m ocr_utils.external_ocr_services run \
    --in-dir "$SHARPENED_DIR" \
    --out-dir "$EXTERNAL_OCR_SERVICES_OUT" \
    --debug-dir "$EXTERNAL_OCR_SERVICES_DEBUG" \
    --db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --source "$EXTERNAL_OCR_SOURCE" \
    --jobs "$JOBS" \
    --skip-done \
    --on-missed-toc skip \
    "$@"
uv run python -m ocr_utils.external_ocr_services balance
