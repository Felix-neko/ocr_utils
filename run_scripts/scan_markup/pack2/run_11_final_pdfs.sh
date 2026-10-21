#!/usr/bin/env bash
#
# Финальные PDF пака-2 (1977, выпуски 01-06): постранично из двух прогонов FineReader (с коррекцией геометрии и без),
# с правкой текстового слоя (повёрнутый текст, пропущенный прямой) и возвратом иллюстраций. Источник страницы:
# растр в базе → без коррекции; вердикт детектора геометрии v18 bad → без коррекции; иначе — с коррекцией.
#
# Сначала: run_9_page_layout_fr.sh (разбор обоих PDF) и run_10_geometry_v18.sh (вердикты); без них сборщик меряет
# страницы сам, но разбора page_layout не строит.
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR (6 пар PDF), $BLURRED_DIR (очищенные полосы — источник иллюстраций, печати
#         уже закрашены), $DB_REVIEWED (только чтение; номера страниц полос записаны сборкой промежуточных PDF),
#         детектор геометрии v18: $GEOMETRY_V18_DIR/cache, $GEOMETRY_V16_DIR/cache, разбор $GEOMETRY_LAYOUT_ROOT.
# Пишет:  $FINAL_PDF_DIR/{год}/{год}_{выпуск}.pdf; $FINAL_WORK_DIR/{pages/<pdf>/pNNNN.json, analysis.csv,
#         pages.csv, summary.csv, preview/}.
# Время (по паку-1, ~2 с на страницу анализа на воркер, ~20 с на выпуск сборки): анализ ~600 страниц — минуты,
#         surya по ненадёжным зонам — минуты (GPU в родителе), сборка — пара минут. Всё идемпотентно (--skip-done).
# Числа — как у пака-1 (run_scripts/scan_markup/pack1/run_final_pdfs.sh): иллюстрации 300 dpi, JPEG 75 (решение
#        пользователя 2026-09-06), Gaussian σ = 0.05 мм против растровой сетки; поля промежуточной PDF
#        12.192 / 6.096 мм (12 / 6 мм, округлённые вверх до MCU JPEG — промежуточные PDF пака-2 собраны с теми же
#        полями); пороги детектора геометрии — по умолчанию (ocr_utils/geometry_regression/quality/scoring.py).
#
# Проба на одном выпуске: ./run_11_final_pdfs.sh --only-year 1977 --only-issue 01
# Весь пак в фон:  setsid ./run_11_final_pdfs.sh > лог 2>&1 < /dev/null & PID=$!; ждать по PID.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
# common.sh пака: пути PDF, BLURRED_DIR, DB_REVIEWED, детектор геометрии; он же переходит в корень репо.
# ПОЛЯ ПРОМЕЖУТОЧНОЙ PDF. Сборщик берёт их из базы (Issue.full_intermediate_pdf_margin_*_mm пака-2
# записаны intermediate_pdfs: 12.192 / 6.096 мм); --margin-x-mm / --margin-y-mm ниже — только запас для
# выпусков без полей в базе. Поле каждой страницы меряется по её образу (номинал или на MCU 8 меньше).

source "$(dirname "$0")/common.sh"

# Анализ — чистый CPU (tesseract): 16 воркеров минус резерв ядер под родителя (surya, запись).
# Сборка — запись PDF и разжатие TIFF: упор в диск, воркеров меньше; выпусков всего 6.
JOBS="${JOBS:-16}"
RESERVE_CPU_CORES=4
ASSEMBLE_JOBS="${ASSEMBLE_JOBS:-6}"

uv run python -m ocr_utils.final_pdfs run \
    --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" \
    --out-dir "$FINAL_PDF_DIR" \
    --pictures-dir "$BLURRED_DIR" \
    --db "$DB_REVIEWED" \
    --pack-name "$PACK_NAME" \
    --work-dir "$FINAL_WORK_DIR" \
    --geometry-run-dir "$GEOMETRY_V18_DIR" \
    --geometry-v16-dir "$GEOMETRY_V16_DIR" \
    --geometry-layout-root "$GEOMETRY_LAYOUT_ROOT" \
    --layout-cache "$LAYOUT_CACHE_DIR" \
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
