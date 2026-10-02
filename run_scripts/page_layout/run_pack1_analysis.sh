#!/usr/bin/env bash
#
# Разбор страниц пака-1 (заострённые копии) в новом порядке: surya (кэш) → ориентация (CPU-детекторы,
# уверенный поворот — с повтором surya) → растр → таблицы (подсказки line art, линейки-сироты) →
# line art (кандидаты классики + подсказки surya/таблиц, формулы surya; DeepSeek-OCR-2 через vLLM в
# два прохода) → текстовые блоки (запреты: растр, печати, таблицы, line art, формулы).
#
# Выход: $OUT/pages/*.json, $OUT/overlays/{только_текст,не_только_текст/<класс|несколько_классов>,
# ориентация_спорная}/, $OUT/index.csv; рабочие файлы — $OUT/work (прерванный прогон продолжается).
#
# ОРИЕНТАЦИЯ не определяется (--no-orientation): заострённые копии пака-1 экспортированы уже
# повёрнутыми как надо. Первый прогон с ориентацией (2026-09-27) не повернул ни одной полосы из
# 12 135, а 126 «спорных» вердиктов — прямые полосы без обычных строк (таблицы, схемы, обложки)
# и боковые таблицы; стадия стоила 23 мин. Для паков без готовых поворотов флаг не ставить.
#
# ВРЕМЯ (замер 2026-09-27, --jobs 16): пробные 56 полос — 2 мин (DeepSeek ~1.5 мин вместе с пуском
# движка). На весь пак ожидается ~1.5 ч: ориентация ~0.15 с/полосу в пуле, кандидаты ~0.2,
# DeepSeek ~0.3 с/вырезку, текстовые блоки ~3 с/полосу.
# DeepSeek идёт подпроцессом под сторожем памяти (scripts/memory_watchdog.py, потолок 90 ГБ).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/page_layout/run_pack1_analysis.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

# Папку выхода можно переопределить: OUT=/mnt/hotstore/scan_processing/mts/pack1_page_analysis_v2 bash …
OUT="${OUT:-/mnt/hotstore/scan_processing/mts/pack1_page_analysis}"
JOBS=16

uv run python -m ocr_utils.page_layout analyze-pack \
    --sharpened-dir "$SHARPENED_DIR" --cache "$LAYOUT_CACHE_DIR" --out-dir "$OUT" --jobs "$JOBS" --no-orientation "$@"
