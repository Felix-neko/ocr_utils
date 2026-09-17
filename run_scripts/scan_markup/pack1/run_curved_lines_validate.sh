#!/usr/bin/env bash
#
# Проверка детекторов кривых строк на 14 размеченных полосах пака-1.
#
# ЗАЧЕМ. Эталон назвал пользователь: 8 полос, которым коррекция геометрии FineReader
# помогает, и 6 с нормальной геометрией (curved_lines_labels.csv). По ним калибруются
# пороги: таблица разделения показывает, какая метрика разводит классы и где лежит зазор.
#
# ВЫХОД — md с таблицей «полоса × детектор» и таблицей разделения, оверлеи на все 14
# полос (что именно увидел каждый детектор), симлинки с вердиктом в имени (ok/missed/false).
# Минуты две вместе с холодным стартом surya.
set -euo pipefail
source "$(dirname "$0")/common.sh"
set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

BASE_NAME="curved_lines_validate_pack1"
CACHE_DIR="$MARKUP_ROOT/curved_lines_cache"

ARGS=(
    run
    --root "$SHARPENED_DIR"
    --link-root "$PACK_DIR"
    --db "$DB_REVIEWED"
    --pack-name "$PACK_NAME"
    --labels run_scripts/scan_markup/pack1/curved_lines_labels.csv
    --labelled-only
    --suggest-thresholds         # таблица разделения по каждой метрике — в stdout и в md
    --jobs 4
    --cache-dir "$CACHE_DIR"
    --link-dir "${BASE_NAME}_links"
    --csv "${BASE_NAME}.csv"
    --md-report "${BASE_NAME}.md"
    --overlay-dir "$MARKUP_ROOT/curved_lines_validate_overlay"
    --overlay all
)

echo "Проверка детекторов кривых строк на размеченных полосах $SHARPENED_DIR"
uv run python -m ocr_utils.scan_markup.curved_lines "${ARGS[@]}" "$@"
