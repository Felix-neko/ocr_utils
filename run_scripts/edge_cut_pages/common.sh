#!/usr/bin/env bash
#
# Общие пути стенда «строки, срезанные краем листа» (research/edge_cut_pages). Сам ничего не запускает —
# подключается через `source` из соседних run_*.sh.

# Пак-6 МТС/РИСК 1989–1993 «от Лены»: фото-развороты подшивок JPG 5687×4268 — Я.Диск (NTFS-3G, медленно;
# только чтение, ничего туда не писать).
PACK6_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-6 от Лены (1989-1992)"
# Выход стенда — SSD hotstore: scores.csv, lines.jsonl, оверлеи, вырезки, листы, папки симлинков.
EDGE_CUT_PACK6_OUT="${EDGE_CUT_PACK6_OUT:-/mnt/hotstore/scan_processing/edge_cut_pages/pack6}"
