#!/usr/bin/env bash
#
# Общие пути сборки финальных PDF пака-1. Сам ничего не запускает — подключается через
# `source` из соседних run_*.sh.

# Пути пака, PDF_ROOT, GEO_PDF_DIR/NOGEO_PDF_DIR, BLURRED_DIR, FINAL_PDF_DIR, FINAL_WORK_DIR,
# базы разметки — из общего файла разметки.
source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"

# Прогон детектора порчи геометрии по паку: cache/<pdf>/pNNN.json той же версии детектора
# берётся как есть, страницы без записи меряются на месте (~3 с) и дописываются в этот же кэш.
GEOMETRY_REGRESSION_ROOT="/mnt/SYSTEM/raw/mts/pack1_geometry_regression"
GEOMETRY_RUN_DIR="${GEOMETRY_RUN_DIR:-$GEOMETRY_REGRESSION_ROOT/pack1_v12}"
