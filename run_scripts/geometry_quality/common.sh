#!/usr/bin/env bash
#
# Общие пути стенда v17 детектора порчи геометрии FineReader (research/geometry_quality). Сам ничего не
# запускает — подключается через `source` из соседних run_*.sh.

source "$(dirname "${BASH_SOURCE[0]}")/../geometry_regression/common.sh"

# Разбор page_layout обоих вариантов PDF (run_scripts/page_layout/run_pack1_analysis_v6_fr.sh): geo/, nogeo/.
QUALITY_LAYOUT_ROOT="${QUALITY_LAYOUT_ROOT:-$GEOMETRY_LAYOUT_ROOT}"
# Прогон v16: из его кэша берутся метрики штрихов, line art, фото и поле смещений B → A.
QUALITY_V16_DIR="${QUALITY_V16_DIR:-$GEOMETRY_V16_DIR}"
QUALITY_V14_CSV="${QUALITY_V14_CSV:-$GEOMETRY_REGRESSION_ROOT/pack1_v14/metrics.csv}"
# Выход стенда — домашний SSD (/mnt/system заполнен): кэш страниц, report.md, verdicts.csv, оверлеи.
QUALITY_ROOT="$MARKUP_ROOT/pack1_geometry_quality"
QUALITY_RUN_DIR="${QUALITY_RUN_DIR:-$QUALITY_ROOT/pack1_v17}"
