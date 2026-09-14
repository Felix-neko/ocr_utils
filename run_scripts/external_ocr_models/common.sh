#!/usr/bin/env bash
#
# Общие параметры прогонов внешних OCR-моделей по паку-1. Сам ничего не запускает —
# подключается через `source` из соседних run_*.sh.
#
# Ключ OpenRouter берётся из $OPENROUTER_API_KEY; в скриптах и логах он не появляется.

# Пути пака (SHARPENED_DIR и прочие) — из общего файла разметки.
source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"

# Выходы — на SSD, рядом с остальными результатами по паку-1. Подпапка на модель, внутри
# та же раскладка {год}/{выпуск}/полоса.{json,md,meta.json}, что у входа. Пробник (13 полос
# на всех моделях) живёт в своём корне, чтобы полные прогоны его не перезаписывали.
EXTERNAL_OCR_ROOT="/mnt/SYSTEM/raw/mts/pack1_external_ocr"
EXTERNAL_OCR_PROBE_ROOT="/mnt/SYSTEM/raw/mts/pack1_external_ocr_probe"

# Текстовый слой FineReader по выпуску — прокси-эталон для сравнения букв. Страница i в PDF
# соответствует i-й полосе выпуска по сортировке имён.
FINEREADER_PDF_DIR="$PDF_ROOT/full_pdfs_binary_brightened_bg"

# Распознанные боковые ячейки таблиц (ocr_utils.rotated_text.tables): по ним проверяем,
# прочитала ли модель повёрнутый текст.
ROTATED_INFO_DIR="/mnt/SYSTEM/raw/mts/pack1_rotated_tables/info"

# Параллельных запросов на модель. Это сеть, а не CPU: 4 — чтобы не упираться в лимиты
# провайдеров (429 у Fireworks ловился уже на одиночных запросах).
EXTERNAL_OCR_JOBS=4
