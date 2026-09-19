#!/usr/bin/env bash
#
# Сравнение геометрического вердикта с вычёркиванием мусора языковой моделью (DeepSeek V4.1
# Flash через OpenRouter, текст без картинок): 150 случайных строк дерева структуры FineReader,
# где есть хотя бы одно слово не KEEP → $TEXT_LAYER_RUN_DIR/{llm_compare.csv,llm_compare.json}.
# Нужен OPENROUTER_API_KEY. Цена — доли цента на строку; ~1 с на строку.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

uv run python -m research.text_layer_fix llm-compare \
    --out-dir "$TEXT_LAYER_RUN_DIR" \
    --model deepseek-v41-flash \
    --limit 150 \
    "$@"
