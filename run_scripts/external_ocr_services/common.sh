#!/usr/bin/env bash
#
# Общие параметры боевого внешнего OCR по паку-1 (ocr_utils.external_ocr_services). Сам ничего
# не запускает — подключается через `source` из соседних run_*.sh.
#
# Ключ OpenRouter берётся из $OPENROUTER_API_KEY; в скриптах и логах он не появляется.

# Пути пака (SHARPENED_DIR, DB_REVIEWED и прочие) — из общего файла разметки.
source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"

# Выходы — на SSD, рядом с остальными результатами по паку-1 и отдельно от стенда
# research/external_ocr_models (pack1_external_ocr): у боевого прогона своя схема ответа и своя
# нумерация промптов, смешивать их с пробниками нельзя. Внутри — та же раскладка
# {год}/{выпуск}/полоса.{md,json,meta.json}, что у входа, плюс toc.json/toc.md на выпуск.
EXTERNAL_OCR_SERVICES_ROOT="/mnt/SYSTEM/raw/mts/pack1_external_ocr_services"
EXTERNAL_OCR_SERVICES_OUT="$EXTERNAL_OCR_SERVICES_ROOT/out"
# Сырые ответы модели, промпты и отправленные тайлы — по полосе; полезно при разборе сбоев
# и при проверке сетки тайлов. Весит примерно как вход в JPEG 2200 px (~0.6 МБ на полосу).
EXTERNAL_OCR_SERVICES_DEBUG="$EXTERNAL_OCR_SERVICES_ROOT/debug"

# Описание издания для промпта; «{year}» подставляется годом выпуска.
EXTERNAL_OCR_SOURCE="журнал «Материально-техническое снабжение», Москва, {year}"

# Параллельных запросов. Это сеть, а не CPU: 4 — чтобы не упираться в лимиты провайдера
# (429 у Fireworks ловился уже на одиночных запросах в стенде).
JOBS=4
