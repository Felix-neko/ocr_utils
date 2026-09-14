#!/usr/bin/env bash
#
# Мини-набор повреждённых сканов через DeepSeek V4.1 Flash в режиме повреждений:
# достроенные буквы <restored>, сомнительные <fuzzy>, нечитаемые <unknown/>.
#
# ВХОД — research/external_ocr_models/damaged/нарезанное по страницам (после run_split_damaged.sh
# и просмотра глазами), подсказки по страницам — damaged_hints.txt рядом с этим скриптом.
# ВЫХОД — research/external_ocr_models/damaged/выход/<имя прогона>/… (json/md/meta) и сводка
# research/external_ocr_models/damaged/выход/<имя прогона>.md. Секунды на страницу, ~0,2 ¢.
#
# Первый аргумент — имя прогона (по умолчанию deepseek-v41-flash-s2); остальные уходят в run.
set -euo pipefail
cd "$(dirname "$0")/../.."

DAMAGED="research/external_ocr_models/damaged"
NAME="${1:-deepseek-v41-flash-s2}"; shift || true

uv run python -m research.external_ocr_models run \
    --in-dir "$DAMAGED/нарезанное по страницам" \
    --out-dir "$DAMAGED/выход/$NAME" \
    --model deepseek-v41-flash \
    --strips 2 \
    --damage \
    --hints run_scripts/external_ocr_models/damaged_hints.txt \
    --jobs 4 \
    "$@"
uv run python -m research.external_ocr_models report --out-root "$DAMAGED/выход" --models "$NAME" \
    --title "Повреждённые сканы: $NAME" --report "$DAMAGED/выход/$NAME.md" > /dev/null
echo "готово: $DAMAGED/выход/$NAME.md"
