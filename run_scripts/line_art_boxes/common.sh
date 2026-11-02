#!/usr/bin/env bash
# Пути стенда research/line_art_boxes: входы — из common.sh прогонов v2 и пака-1, выход — свой корень на SSD hotstore.
source "$(dirname "${BASH_SOURCE[0]}")/../scan_cropping/common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"
# Стенд «завышенные рамки line art во втором преобразовании v2»: вырезки холста, разборы, листы — SSD hotstore.
LINE_ART_BOXES_ROOT="${LINE_ART_BOXES_ROOT:-/mnt/hotstore/scan_processing/line_art_boxes}"
