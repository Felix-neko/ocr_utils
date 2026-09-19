#!/usr/bin/env bash
#
# Оверлеи страниц выборки по кэшу прогона run: зоны, таблицы, картинки FineReader и слова
# слоя по вердиктам (красный DELETE, оранжевый SANITIZE, жёлтый SUSPECT, голубой KEEP_ROTATED,
# пурпурный — зона с принятым чтением) → $TEXT_LAYER_RUN_DIR/overlays/<категория>/*.png.
# Для отчёта папка копируется в reports/text_layer_fix/ (вне git).
#
# Читает: $TEXT_LAYER_PDF_DIR, $TEXT_LAYER_RUN_DIR/{sample.csv,cache}. Идёт ~0.5 с на страницу.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

JOBS="${JOBS:-8}"

uv run python -m research.text_layer_fix overlay \
    --pdf-dir "$TEXT_LAYER_PDF_DIR" \
    --out-dir "$TEXT_LAYER_RUN_DIR" \
    --jobs "$JOBS" \
    "$@"
