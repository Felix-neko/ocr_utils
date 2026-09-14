#!/usr/bin/env bash
#
# Полный выпуск 1966/03 (97 полос) через модели, выбранные по пробнику: три дешёвые
# (Gemini 3.1 Flash Lite, Qwen3.8 Flash, DeepSeek V4.1 Flash двумя кусками) и Gemini 3.8
# Flash как эталон подороже.
#
# ВХОД — заострённые полосы выпуска (SHARPENED_DIR/1966/03).
# ВЫХОД — EXTERNAL_OCR_ROOT/<модель>/1966/03/<полоса>.{json,md,meta.json}, summary.csv, run.log.
#
# СТОИМОСТЬ И ВРЕМЯ по пробнику: Gemini Lite 0,19 ¢/полоса, Qwen 0,08, DeepSeek 0,09,
# Gemini 3.8 0,48 — около 80 ¢ за выпуск на всех четырёх; 5-12 с на запрос при 4 потоках —
# 3-5 минут на модель. Запускать в фоне и ждать ПО СОХРАНЁННОМУ PID (см. CLAUDE.md):
#
#   setsid ./run_scripts/external_ocr_models/run_issue_1966_03.sh > /tmp/issue.log 2>&1 < /dev/null & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 30; done
set -euo pipefail
source "$(dirname "$0")/common.sh"

set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

ISSUE="1966/03"

run_model() {  # имя модели, имя папки выхода, дополнительные опции
    local model="$1" out_name="$2"; shift 2
    echo "=== $out_name"
    uv run python -m research.external_ocr_models run \
        --in-dir "$SHARPENED_DIR/$ISSUE" \
        --out-dir "$EXTERNAL_OCR_ROOT/$out_name/$ISSUE" \
        --model "$model" \
        --jobs "$EXTERNAL_OCR_JOBS" \
        --skip-done \
        "$@"
}

run_model gemini-31-flash-lite gemini-31-flash-lite
run_model qwen38-flash qwen38-flash
run_model deepseek-v41-flash deepseek-v41-flash-s2 --strips 2   # целой полосой теряет текст: потолок 1024 токена на картинку
run_model gemini-38-flash gemini-38-flash

uv run python -m research.external_ocr_models balance
