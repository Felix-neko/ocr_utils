#!/usr/bin/env bash
#
# Шаг 1а: полосы оглавления («Содержание» выпуска и указатель статей за год) -> SQLite,
# признаки в CSV, контактные листы окна и проверка по эталону.
#
# Отдельный от run_1_detect.sh запуск ради калибровки: детектор меряет только окно
# выпуска (~2 200 полос на пак), заострённые JPEG на SSD читаются быстро, и весь прогон
# с tesseract на 16 процессах занимает минуты, а не часы. Пороги правятся через --thr без
# правки кода; ниже — значения по умолчанию из toc/decide.py.
#
# Аргументы: всё, что после имени скрипта, уходит команде toc (например --only-year 1975
# --only-issue 12 --dry-run).
#
# ПОСЛЕ ПРОГОНА: список выпусков без «Содержания» в выводе — красный флаг, смотреть
# контактные листы $TOC_DIR/sheets/<год>/<выпуск>.jpg. Списки полос для внешнего OCR
# выгружает run_toc_pages.sh.

set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "$TOC_DIR"
TOC_ARGS=(
    --pack-dir "$PACK_DIR"
    --db "$DB"
    --pack-name "$PACK_NAME"
    --layout-cache "$LAYOUT_CACHE_DIR"
    --jobs 16
    --csv "$TOC_DIR/toc_pack1.csv"
    --debug-dir "$TOC_DIR/sheets"
    # Найденное по выпускам: «содержание/» и отдельно «указатели/» (только декабрьские).
    --found-dir "$TOC_DIR/found"
    # Пороги решения (см. toc/decide.py::Thresholds):
    # --thr weak_min_lines=3,weak_min_ratio=0.2,table_min_area=0.4,surya_min_conf=0.3
    # --thr pair_min_lines=1,window_start=4,window_end=11
)
if [ -d "$SHARPENED_DIR" ]; then
    TOC_ARGS+=(--image-root "$SHARPENED_DIR")
fi
if [ -f "$TOC_LABELS" ]; then
    TOC_ARGS+=(--labels "$TOC_LABELS")
fi
uv run python -m ocr_utils.scan_markup toc "${TOC_ARGS[@]}" "$@"
