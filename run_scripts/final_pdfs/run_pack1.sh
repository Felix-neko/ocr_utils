#!/usr/bin/env bash
#
# Финальные PDF пака-1: постранично из двух прогонов FineReader (с коррекцией геометрии и без),
# с правкой текстового слоя (повёрнутый текст, пропущенный прямой) и возвратом иллюстраций.
#
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR (123 пары PDF), $BLURRED_DIR (очищенные полосы — источник
#         иллюстраций, печати уже закрашены), $DB_REVIEWED (только чтение), $GEOMETRY_RUN_DIR/cache.
# Пишет:  $FINAL_PDF_DIR/{год}_{выпуск}.pdf; $FINAL_WORK_DIR/{pages/<pdf>/pNNNN.json, analysis.csv,
#         pages.csv, summary.csv, preview/}.
# Стадии: A — анализ страниц в пуле (tesseract по ячейкам и зонам, ~2 с на страницу на воркер:
#         ~1 ч на пак при 12 воркерах); B — surya по ненадёжным зонам в родителе (GPU, десятки минут);
#         C — сборка выпусков (~20 с на выпуск, 8 воркеров — упор в диск). Всё идемпотентно.
# Числа: иллюстрации 300 dpi, JPEG 75 (решение пользователя 2026-09-06), Gaussian σ = 0.05 мм
#        против растровой сетки (проба на 4 врезках, ocr_utils/final_pdfs/pictures.py); поля
#        промежуточной PDF 12.192 / 6.096 мм (12 / 6 мм, округлённые вверх до MCU JPEG — проверено
#        по образам no-geo PDF на 1181 странице); пороги детектора геометрии — по умолчанию
#        (ocr_utils/geometry_regression/scoring.py); приём чтений — reports/text_layer_fix.md.
#
# Проба на одном выпуске: ./run_pack1.sh --only-year 1966 --only-issue 03
# Весь пак в фон:  setsid ./run_pack1.sh > лог 2>&1 < /dev/null & PID=$!; ждать по PID.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

# Анализ — чистый CPU (tesseract): 16 воркеров минус резерв ядер под родителя (surya, запись).
# Сборка — запись PDF и разжатие 20-мегапиксельных TIFF: упор в диск, воркеров меньше.
JOBS="${JOBS:-16}"
RESERVE_CPU_CORES=4
ASSEMBLE_JOBS="${ASSEMBLE_JOBS:-8}"

uv run python -m ocr_utils.final_pdfs run \
    --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" \
    --out-dir "$FINAL_PDF_DIR" \
    --pictures-dir "$BLURRED_DIR" \
    --db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --work-dir "$FINAL_WORK_DIR" \
    --geometry-run-dir "$GEOMETRY_RUN_DIR" \
    --picture-dpi 300 \
    --jpeg-quality 75 \
    --descreen-sigma-mm 0.05 \
    --margin-x-mm 12.192 \
    --margin-y-mm 6.096 \
    --jobs "$JOBS" \
    --reserve-cpu-cores "$RESERVE_CPU_CORES" \
    --assemble-jobs "$ASSEMBLE_JOBS" \
    --preview-pages 2 \
    --skip-done \
    "$@"
