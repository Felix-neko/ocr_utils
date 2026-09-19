#!/usr/bin/env bash
#
# Общие пути прогонов детектора «FineReader ухудшил геометрию» по паку-1. Сам ничего не
# запускает — подключается через `source` из соседних run_*.sh.

# Пути пака и PDF_ROOT — из общего файла разметки.
source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"

# Два прогона FineReader (GEO_PDF_DIR / NOGEO_PDF_DIR) — из common.sh пака.

# Выход прогона — на SSD: JSON на страницу (cache/), metrics.csv, отчёт, картинки pairs/<год>/.
# Подпапка прогона — версия детектора, чтобы прошлый прогон оставался для сравнения;
# перекрыть можно переменной окружения: GEOMETRY_RUN_DIR=... ./run_pack1.sh
GEOMETRY_REGRESSION_ROOT="/mnt/SYSTEM/raw/mts/pack1_geometry_regression"
GEOMETRY_RUN_DIR="${GEOMETRY_RUN_DIR:-$GEOMETRY_REGRESSION_ROOT/pack1_v12}"

# Эталон: папка валидации с fr_correction_bad.csv и fr_correction_good.csv (TSV, вердикты
# пользователя по картинкам «было | стало»); пополнять именно их.
GEOMETRY_LABELS="$(dirname "${BASH_SOURCE[0]}")/../../research/geometry_regression/validation/pack1"
