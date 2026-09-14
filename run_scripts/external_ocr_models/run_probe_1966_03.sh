#!/usr/bin/env bash
#
# Пробник внешних OCR-моделей: 13 отобранных полос выпуска 1966/03 через каждую модель
# пробника, плюс DeepSeek V4.1 Flash в вариантах «целая полоса / 2 куска / 3 куска».
#
# ВХОД — заострённые полосы выпуска (SHARPENED_DIR/1966/03) и список полос
# probe_pages_1966_03.txt рядом с этим скриптом.
#
# ВЫХОД — EXTERNAL_OCR_PROBE_ROOT/<модель>/1966/03/<полоса>.{json,md,meta.json}, summary.csv и
# run.log в папке модели.
#
# СТОИМОСТЬ И ВРЕМЯ. Замер по DeepSeek: 0,13-0,17 ¢ за полосу, 10-60 с на запрос при
# reasoning off. 13 полос × 11 прогонов ≈ 143 запроса, порядка $1-2 и 20-40 минут.
# Запускать в фоне и ждать ПО СОХРАНЁННОМУ PID (см. CLAUDE.md):
#
#   ./run_scripts/external_ocr_models/run_probe_1966_03.sh & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 30; done
set -euo pipefail
source "$(dirname "$0")/common.sh"

set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

PAGES="$(dirname "$0")/probe_pages_1966_03.txt"
MODELS=(
    deepseek-v41-flash
    gemini-31-flash-lite
    qwen38-flash
    gpt-56-luna
    mistral-small-4
    glm-53-flash
    gemini-38-flash
    qwen3-vl-235b
    claude-haiku-45
)

run_model() {  # имя модели, имя папки выхода, дополнительные опции
    local model="$1" out_name="$2"; shift 2
    echo "=== $out_name"
    uv run python -m research.external_ocr_models run \
        --in-dir "$SHARPENED_DIR" \
        --out-dir "$EXTERNAL_OCR_PROBE_ROOT/$out_name" \
        --model "$model" \
        --pages "$PAGES" \
        --jobs "$EXTERNAL_OCR_JOBS" \
        --skip-done \
        "$@"
}

for model in "${MODELS[@]}"; do
    run_model "$model" "$model"
done
# DeepSeek режет картинку до ~1300 px по стороне — смотрим, что даёт подача кусками.
run_model deepseek-v41-flash deepseek-v41-flash-s2 --strips 2
run_model deepseek-v41-flash deepseek-v41-flash-s3 --strips 3

uv run python -m research.external_ocr_models balance
