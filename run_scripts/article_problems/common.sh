#!/usr/bin/env bash
#
# Общие пути стенда research/article_problems (отбор статей пака-1 о проблемах предприятий).
# Сам ничего не запускает: подключается через `source` из соседних run_*.sh.

# Вход — собранные выпуски внешнего OCR (123 md, 2880 статей). Лежат на Я.Диске: только читать,
# туда ничего не писать (Я.Диск затирает исходники при переименовании).
PACK_MD_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/md-файлы/пак-1"

# Рабочая папка на SSD hotstore (перенесена из ~/Projects/mts_markup 2026-10-01): кэш запросов (cache/), состояния пачек Batch API (batches/), сводки (runs/), логи.
WORK_DIR="/mnt/hotstore/scan_processing/mts_markup/article_problems"
