#!/usr/bin/env bash
#
# Общие пути стенда брака разворотов (research/spread_defects). Сам ничего не запускает — подключается
# через `source` из соседних run_*.sh.

# Пак-4 «Планового хозяйства» — исходники на Я.Диске (NTFS-3G, медленно; только чтение, ничего туда не писать).
PH_PACK4_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/в работе/пак-4 (450 dpi)"
# Выход стенда — SSD hotstore: плиточные сетки, сводки, листы глазами.
SPREAD_DEFECTS_ROOT="${SPREAD_DEFECTS_ROOT:-/mnt/hotstore/scan_processing/ph_pack4_defects}"
