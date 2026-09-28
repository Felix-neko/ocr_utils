#!/usr/bin/env bash
#
# Стенд соринок (research/text_block_specks): чужие движки строк на полосах набора — 50 с выступами края блока
# из-за соринок и пометок и 20 первых чистых (eval_pages.json), вариант nogeo, по одному движку на GPU.
# Выход: $OUT/pages/*.json (оси строк движка), логи — $OUT/logs.
#
# ВРЕМЯ: ink/paddle6 — минуты, pero/surya ~5–10 мин, kraken ~20 мин, eynollah ~70 мин (60 с/полосу).
#
#     setsid bash run_scripts/text_blocks/run_specks_engines.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
cd "$(dirname "$0")/../.."
source run_scripts/geometry_regression/common.sh

OUT="${OUT:-/mnt/system/raw/mts/text_block_specks/engines}"
SET="${SET:-/mnt/system/raw/mts/text_block_specks/eval_pages.json}"
ENGINES=(ink pero paddle6 kraken surya eynollah)
if [ "$#" -gt 0 ]; then ENGINES=("$@"); fi

PAGES=$(python3 -c "
import json, sys
spec = json.load(open(sys.argv[1]))
print(','.join(x['pdf'][:-4] + ':' + str(x['page']) for x in spec['defect'] + spec['clean'][:20]))" "$SET")

mkdir -p "$OUT/logs"
for engine in "${ENGINES[@]}"; do
    uv run python -m ocr_utils.page_layout.text_blocks analyze --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" \
        --pages "$PAGES" --variant nogeo --smooth-block 2.5 --coarse-factor 3 --smooth-line 0.6 --out-dir "$OUT" \
        --engine "$engine" > "$OUT/logs/$engine.log" 2>&1 || echo "сбой $engine"
    echo "готов $engine"
done
