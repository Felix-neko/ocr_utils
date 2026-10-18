#!/usr/bin/env bash
#
# Весь пак-2 (597 полос, 6 выпусков) через DeepSeek V4.1 Flash по выпускам (как ../run_pack_sharpened.sh у
# пака-1). Идемпотентен: --skip-done пропускает готовые полосы, кэш запросов — готовые ответы; прерванный
# прогон продолжается тем же вызовом. Оглавления, найденные моделью вне базы, перераспознаются сами
# (--on-missed-toc redo) и попадают в лог — после прогона проставить по ним теги в CVAT.
#
# Batch API OpenRouter не годится: пачки с картинками в base64 он отвергает целиком («only public http(s)
# image URLs are supported in batch», проба 2026-10-18).
#
# Провайдер — только DeepSeek (откат запрещён в models.py): 2026-10-18 первый прогон молча ушёл к OpenInference,
# и тот пересказывал ~16 % полос. В конце run сверяет ответы с tesseract (fidelity.csv в pages, подозрительные —
# в логе); полосы ниже порога смотреть глазами. Первый прогон перенесён в pack2_external_ocr_services_openinference_2026-10-18.
#
# ЧИТАЕТ  EXTERNAL_OCR_IN_DIR (очищенные TIFF без заострения, 14 ГБ, SSD), DB_REVIEWED (только чтение:
#         теги оглавлений, 12 полос).
# ПИШЕТ   EXTERNAL_OCR_SERVICES_PAGES (полосы, toc.json, sidecar, summary.csv, run.log),
#         EXTERNAL_OCR_SERVICES_OUT (только md выпусков <год>/<год>_<номер>.md),
#         EXTERNAL_OCR_SERVICES_DEBUG (~0.6 МБ на полосу), EXTERNAL_OCR_SERVICES_CACHE (кэш запросов).
# ОРИЕНТИР: пак-1 — ~$1 на 1000 полос и ~8 ч на 12 135 полос при 4 потоках; пак-2 — $0.60 и 9 мин при 12 (2026-10-18), сверка с tesseract — ещё ~2,5 мин.
# Запуск в фон из сессии агента:
#   setsid run_scripts/external_ocr_services/pack2/run_pack.sh > лог 2>&1 < /dev/null & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 60; done
set -euo pipefail
source "$(dirname "$0")/common.sh"

uv run python -m ocr_utils.external_ocr_services run \
    --in-dir "$EXTERNAL_OCR_IN_DIR" \
    --pages-dir "$EXTERNAL_OCR_SERVICES_PAGES" \
    --issues-dir "$EXTERNAL_OCR_SERVICES_OUT" \
    --debug-dir "$EXTERNAL_OCR_SERVICES_DEBUG" \
    --cache-dir "$EXTERNAL_OCR_SERVICES_CACHE" \
    --db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --source "$EXTERNAL_OCR_SOURCE" \
    --max-src-tile-size "$MAX_SRC_TILE_SIZE" \
    --jobs "$JOBS" \
    --skip-done \
    "$@"
uv run python -m ocr_utils.external_ocr_services balance
