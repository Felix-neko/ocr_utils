#!/usr/bin/env bash
#
# Финальные PDF пака-1: распознанные страницы плюс возвращённые на место иллюстрации.
#
# ЗАПУСКАТЬ ПОСЛЕ FineReader. К этому моменту в $FULL_RECOGNIZED_DIR должны лежать
# распознанные полные PDF (бинаризация И распрямление строк), а в
# $PICS_ONLY_RECOGNIZED_DIR — распознанные PAGES_WITH_PICS_ONLY (бинаризация БЕЗ
# распрямления). Имена файлов те же, что у промежуточных: по ним они и ищутся.
#
# Полоса без иллюстраций берётся из полной PDF, полоса с иллюстрациями — из
# PAGES_WITH_PICS_ONLY, и поверх её текстового слоя кладётся кусок ОРИГИНАЛА: цветной или
# серый по виду размеченной области, JPEG качества 75. Текстовый слой при этом остаётся
# под картинкой и продолжает искаться.
#
# Первым делом сверяется количество страниц с тем, что записал сборщик промежуточных PDF.
# Если FineReader страницу выбросил или добавил, номера в базе к его выводу больше не
# относятся, и такой выпуск не собирается вовсе — молчаливая сборка со сдвигом на страницу
# была бы куда хуже.

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

echo "Распознанные полные:  $FULL_RECOGNIZED_DIR"
echo "Распознанные с растром: $PICS_ONLY_RECOGNIZED_DIR"
echo "Оригиналы:            $BLURRED_DIR"
echo "Выход:                $FINAL_PDF_DIR"

FINAL_ARGS=(
    --db "$DB_REVIEWED"
    --pack-name "$PACK_NAME"
    --originals-dir "$BLURRED_DIR"
    --full-pdf-dir "$FULL_RECOGNIZED_DIR"
    --pics-only-pdf-dir "$PICS_ONLY_RECOGNIZED_DIR"
    --final-dir "$FINAL_PDF_DIR"

    # Все выпуски в одну папку. --by-year разложит по годам, если так удобнее раздавать.
    --no-by-year

    --jpeg-quality 75
    --skip-if-exists
    --jobs 8
    --log-level INFO
)
uv run python -m ocr_utils.pdf_utils.final_pdfs "${FINAL_ARGS[@]}" "$@"
