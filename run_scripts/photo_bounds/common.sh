#!/usr/bin/env bash
#
# Общие пути стенда «границы фото» (research/photo_bounds). Сам ничего не запускает — подключается через `source`
# из соседних run_*.sh. Кадры выпрямления — из common.sh scan_cropping.
#
# common.sh пака-1 НЕ подключается: он при каждом подключении делает `cp -f pack1.sqlite pack1.sqlite.bak`, а стенд
# базы только читает (sqlite URI mode=ro) — бэкап ему не нужен и затирал бы прежний .bak. Пути пака-1 повторены
# здесь; при расхождении правы run_scripts/scan_markup/pack1/common.sh и docs/data_layout.md.

source "$(dirname "${BASH_SOURCE[0]}")/../scan_cropping/common.sh"

# База разметки пака-1 после ручной правки в CVAT (системный SSD) — только чтение.
MARKUP_ROOT="/home/felix/Projects/mts_markup"
DB_REVIEWED="$MARKUP_ROOT/pack1_reviewed.sqlite"
# Кэш surya layout пака-1 (вариант scan полон на весь пак) — только чтение.
LAYOUT_CACHE_DIR="$MARKUP_ROOT/pack1_page_layout"
# Выход стенда — SSD hotstore: pages.json (снимок набора из DB_REVIEWED), pages/ (серый 300 dpi, цвет 150 dpi,
# детектор — TIFF с Я.Диска читается один раз), run_vN/ (меры, листы), rectify/ (кадры выпрямления), patches/.
HOT_ROOT="/mnt/hotstore/scan_processing"
PHOTO_BOUNDS_ROOT="${PHOTO_BOUNDS_ROOT:-$HOT_ROOT/photo_bounds}"
