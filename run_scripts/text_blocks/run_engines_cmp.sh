#!/usr/bin/env bash
#
# Сравнение чужих движков осей строк с движком ink на валидационном наборе «текст и оглавления»
# (13 полос × geo/nogeo, тот же список, что в run_text_pages.sh). Отчёт — reports/line_axis_models.md.
#
# Каждый движок — отдельным вызовом analyze в один и тот же каталог: сбой одного движка не роняет
# остальные, JSON полос копятся в $OUT/pages. GPU-движки идут строго по одному (видеопамять одна).
# Окружения движков: kraken/pero/eynollah — /mnt/system/raw/mts/curved_layout_engines,
# остальные — ~/Projects/mts_markup/line_axis_engines (engines/catalog.py).
#
# Числа (2026-09-25): ink ~4 с/полосу, eynollah ~36 с, kraken ~22 с, surya ~10 с с загрузкой модели.
#
# Запуск: ./run_engines_cmp.sh [движок …]   (без аргументов — все движки из ENGINES)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../geometry_regression/common.sh"
cd "$SCRIPT_DIR/../.."

OUT="${OUT:-/mnt/system/raw/mts/line_axis_models/text_and_toc}"
PAGES="${PAGES:-full_1973_06:65,full_1971_10:87,full_1973_07:88,full_1967_10:63,full_1971_10:93,full_1973_11:79,full_1968_07:93,full_1973_07:77,full_1973_08:85,full_1971_10:95,full_1970_02:90,full_1975_05:97,full_1976_09:92}"
ENGINES=(ink pero kraken eynollah surya orli paddle paddle6 chronicling laypa craft docufcn textsnake)
if [ "$#" -gt 0 ]; then ENGINES=("$@"); fi

# Параметры постпроцесса — те же, что в валидационном прогоне ink (run_text_pages.sh).
common=(--geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR" --pages "$PAGES" --variant both
        --smooth-block 2.5 --coarse-factor 3 --smooth-line 0.6 --out-dir "$OUT")

mkdir -p "$OUT/logs"
for engine in "${ENGINES[@]}"; do
    echo "== $engine =="
    if ! uv run python -m ocr_utils.page_layout.text_blocks analyze "${common[@]}" --engine "$engine" \
            > "$OUT/logs/$engine.log" 2>&1; then
        echo "   СБОЙ: см. $OUT/logs/$engine.log"
    fi
    grep -c "блоков" "$OUT/logs/$engine.log" || true
done

echo "== меры =="
for variant in all geo nogeo; do
    uv run python -m research.line_axis_models score --run-dir "$OUT" --variant "$variant" > /dev/null
done
uv run python -m research.line_axis_models curls --run-dir "$OUT" --geo-dir "$GEO_PDF_DIR" --nogeo-dir "$NOGEO_PDF_DIR"
uv run python -m research.line_axis_models grid --run-dir "$OUT"
cat "$OUT/scores_all.md"
