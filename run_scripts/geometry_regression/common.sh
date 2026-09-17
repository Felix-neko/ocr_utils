#!/usr/bin/env bash
#
# Общие пути прогонов детектора «FineReader ухудшил геометрию» по паку-1. Сам ничего не
# запускает — подключается через `source` из соседних run_*.sh.

# Пути пака и PDF_ROOT — из общего файла разметки.
source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"

# Два прогона FineReader по одним и тем же промежуточным PDF (SSD, читаются по разу на страницу):
# с коррекцией геометрии (перекос, искажение строк, трапеция) и без неё.
GEO_PDF_DIR="$PDF_ROOT/full_pdfs_binary_no_bg_brightening"
NOGEO_PDF_DIR="$PDF_ROOT/full_pdfs_binary_no_bg_brightening_no_geometry_correction"

# Выход прогона — на SSD: JSON на страницу (cache/), metrics.csv, отчёт, картинки pairs/<год>/.
GEOMETRY_REGRESSION_ROOT="/mnt/SYSTEM/raw/mts/pack1_geometry_regression"

# Эталон: страницы, проверенные глазами (bad — коррекция сделала хуже, good — помогла или не тронула).
GEOMETRY_LABELS="$(dirname "${BASH_SOURCE[0]}")/labels.csv"
