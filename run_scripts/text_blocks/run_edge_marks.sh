#!/usr/bin/env bash
#
# Стенд краевых пометок и сора (research/edge_marks): кандидаты у 190 размеченных концов строк на
# бинаризованных nogeo PDF (188 кандидатов: сор 106, знаков 82 на 99 полосах), судьи «символ или нет» —
# строго по одному на GPU с замером времени, видеопамяти и ОЗУ, сводка и оверлеи.
# Выход: $OUT/cand (вырезки), $OUT/results (<судья>.jsonl, _usage.json, summary.csv), $OUT/judges (оверлеи, листы).
#
# Окружения судей: pero — /mnt/system/raw/mts/curved_layout_engines/pero, craft/doctr/paddle —
# ~/Projects/mts_markup/line_axis_engines/<движок>, DeepSeek — ocr_utils/page_layout/line_art/deepseek/vllm_env.
# Перед GPU-судьями проверить, что видеопамять свободна (vLLM соседней сессии держит до 13 ГБ).
#
# ВРЕМЯ: rule/tesseract — минуты; deepseek ~15–20 мин; pero/craft/doctr/paddle — минуты каждый.
#
#     setsid bash run_scripts/text_blocks/run_edge_marks.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

OUT="${OUT:-/mnt/system/raw/mts/pack1_edge_marks}"
JUDGES=(rule tesseract deepseek pero craft doctr paddle surya)
if [ "$#" -gt 0 ]; then JUDGES=("$@"); fi

[ -f "$OUT/cand/candidates.json" ] || uv run python -m research.edge_marks candidates --out-dir "$OUT/cand"
for judge in "${JUDGES[@]}"; do
    uv run python -m research.edge_marks judge --judge "$judge" --cand-dir "$OUT/cand" --results-dir "$OUT/results"
done
uv run python -m research.edge_marks score --cand-dir "$OUT/cand" --results-dir "$OUT/results"
uv run python -m research.edge_marks overlays --cand-dir "$OUT/cand" --results-dir "$OUT/results" --out-dir "$OUT/judges"
