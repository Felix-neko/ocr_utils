#!/usr/bin/env bash
#
# Сводки по готовым выходам (ничего не запрашивает): пробник и полный выпуск 1966/03.
# ВЫХОД — reports/external_ocr_models_probe_1966_03.md, reports/external_ocr_models_issue_1966_03.md
# и scores.csv в корне каждого набора выходов.
set -euo pipefail
source "$(dirname "$0")/common.sh"

ISSUE="1966/03"
COMMON=(
    --in-dir "$SHARPENED_DIR"
    --issue "$ISSUE"
    --reference-pdf "$FINEREADER_PDF_DIR/full_${ISSUE//\//_}.pdf"   # текстовый слой FineReader: прокси-эталон букв
    --rotated-info-dir "$ROTATED_INFO_DIR"                            # фразы боковых шапок таблиц
)

uv run python -m research.external_ocr_models report --out-root "$EXTERNAL_OCR_PROBE_ROOT" "${COMMON[@]}" \
    --title "Пробник ${ISSUE}: 13 полос × все модели" --report "reports/external_ocr_models_probe_${ISSUE//\//_}.md" > /dev/null
uv run python -m research.external_ocr_models report --out-root "$EXTERNAL_OCR_ROOT" "${COMMON[@]}" \
    --title "Выпуск ${ISSUE}: 97 полос" --report "reports/external_ocr_models_issue_${ISSUE//\//_}.md" > /dev/null
echo "готово: reports/external_ocr_models_{probe,issue}_${ISSUE//\//_}.md"
