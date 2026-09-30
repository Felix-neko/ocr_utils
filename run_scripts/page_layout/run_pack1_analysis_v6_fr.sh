#!/usr/bin/env bash
#
# Разбор страниц пака-1 v6 по ОБОИМ полным PDF FineReader — с коррекцией геометрии (fr_geo) и без неё
# (fr_nogeo) — под детектор порчи геометрии (research/geometry_quality): его меры строк и краёв блоков
# сравнивают одну и ту же страницу в двух вариантах.
#
# * РАСТР — не ищется (--no-raster): PDF бинарные, растр в них всё равно бинаризован (решение пользователя
#   2026-09-29).
# * ТАБЛИЦЫ, LINE ART с DeepSeek, ФОРМУЛЫ, ПОВЁРНУТЫЙ ТЕКСТ, ТЕКСТОВЫЕ БЛОКИ (вторая ось, межколонники short,
#   защита сторон CRAFT + pero) — как в v5.
# * ОРИЕНТАЦИЯ — не определяется: страницы PDF собраны из уже прямых полос.
# * ВЫХОД — JSON полосы + сайдкар .npz (text_blocks.store: оси строк с перескоками, стороны блоков сырые и с
#   заплатками поверх аномалий, трассы линеек таблиц). Оверлеи не пишутся (--overlays none): v5 с ними —
#   16 ГБ, картинки для отсмотра рисует стенд геометрии.
# * КЭШ surya — $LAYOUT_CACHE_DIR/{fr_geo,fr_nogeo}: перед разбором варианта добивается prefill-surya, сам разбор
#   открывает его только на чтение.
#
# МЕСТО: /mnt/system заполнен — выход в $MARKUP_ROOT (домашний SSD).
#
# ВРЕМЯ (оценка по v5, 12 135 полос): около 2 ч на вариант, варианты по очереди (DeepSeek и CRAFT — GPU).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/page_layout/run_pack1_analysis_v6_fr.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-$MARKUP_ROOT/pack1_page_analysis_v6_fr}"
JOBS="${JOBS:-16}"
VARIANTS="${VARIANTS:-fr_nogeo fr_geo}"
# PREFILL=0 — не добивать кэш surya (прогон по выборке --pages, кэш которой набит заранее).
PREFILL="${PREFILL:-1}"

for variant in $VARIANTS; do
    if [ "$variant" = fr_geo ]; then pdf_dir="$GEO_PDF_DIR"; else pdf_dir="$NOGEO_PDF_DIR"; fi
    echo "=== $variant: $pdf_dir → $OUT/${variant#fr_}"
    # Кэш surya варианта — добить до полного (готовые записи пропускаются): у fr_geo на 2026-09-29 было
    # только 1050 страниц выборки, разбор открывает кэш только на чтение.
    if [ "$PREFILL" = 1 ]; then
        uv run python -m ocr_utils.page_layout prefill-surya --cache "$LAYOUT_CACHE_DIR" --variant "$variant" \
            --pdf-dir "$pdf_dir" --jobs "$JOBS"
    fi
    uv run python -m ocr_utils.page_layout analyze-pack \
        --pdf-dir "$pdf_dir" --variant "$variant" --cache "$LAYOUT_CACHE_DIR" --out-dir "$OUT/${variant#fr_}" \
        --jobs "$JOBS" --no-orientation --no-raster --overlays none --axis body --gutter-mode short "$@"
done
