#!/usr/bin/env bash
#
# Разбор страниц пака-2 по ОБОИМ PDF FineReader — с коррекцией геометрии (fr_geo) и без неё (fr_nogeo) — под детектор
# порчи геометрии v18 (run_10_geometry_v18.sh): его меры строк и краёв блоков сравнивают одну страницу в двух вариантах.
# Ключи — как у пака-1 (run_scripts/page_layout/run_pack1_analysis_v6_fr.sh, разбор pack1_page_analysis_v7_fr):
# без растра (PDF бинарные), без ориентации (полосы уже прямые), оси по базовой линии (body), межколонники short,
# защита сторон блоков CRAFT + pero, оверлеи не пишутся (картинки рисует run_12_geometry_overlays.sh).
#
# Читает: $GEO_PDF_DIR, $NOGEO_PDF_DIR (6 пар PDF, ~100 страниц в каждом).
# Пишет:  кэш surya $LAYOUT_CACHE_DIR/{fr_geo,fr_nogeo}/ (добивается prefill-surya, ~0.7 с GPU на страницу),
#         $GEOMETRY_LAYOUT_ROOT/{geo,nogeo}/pages/<pdf>_pNNNN.{json,npz}.
# Время: по оценке пака-1 (12 135 полос — ~2 ч на вариант) — около 10 мин на вариант; GPU (surya, DeepSeek, CRAFT).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/scan_markup/pack2/run_9_page_layout_fr.sh > log 2>&1 < /dev/null & PID=$!

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

JOBS="${JOBS:-16}"
VARIANTS="${VARIANTS:-fr_nogeo fr_geo}"

for variant in $VARIANTS; do
    if [ "$variant" = fr_geo ]; then pdf_dir="$GEO_PDF_DIR"; else pdf_dir="$NOGEO_PDF_DIR"; fi
    echo "=== $variant: $pdf_dir → $GEOMETRY_LAYOUT_ROOT/${variant#fr_}"
    uv run python -m ocr_utils.page_layout prefill-surya --cache "$LAYOUT_CACHE_DIR" --variant "$variant" \
        --pdf-dir "$pdf_dir" --jobs "$JOBS"
    uv run python -m ocr_utils.page_layout analyze-pack \
        --pdf-dir "$pdf_dir" --variant "$variant" --cache "$LAYOUT_CACHE_DIR" \
        --out-dir "$GEOMETRY_LAYOUT_ROOT/${variant#fr_}" --jobs "$JOBS" \
        --no-orientation --no-raster --overlays none --axis body --gutter-mode short "$@"
done
