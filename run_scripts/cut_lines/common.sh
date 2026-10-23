#!/usr/bin/env bash
#
# Общие пути стенда дорисовки срезанных строк (research/cut_lines). Подключается через `source` из run_*.sh.

# Два фотосета РИСК 1991/05 — Я.Диск, только чтение: основа (1) пак-6 и донор (2) пак-7.
RISK_1991_05_BASE="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-6 от Лены (1989-1992)/РИСК/1991/05 есть фотографии по страницам"
RISK_1991_05_DONOR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-7 от Бори/РИСК/1991/05"
# Кэш scan_cropping v2 по основе (маски бумаги и страниц) — SSD.
RISK_1991_05_V2_WORK="/mnt/hotstore/scan_processing/scan_cropping_v2/risk_1991_05/work"
# Пары разворотов по распознанному тексту (research/spread_ocr) — SSD.
RISK_1991_05_TEXT_PAIRS="/mnt/hotstore/scan_processing/spread_ocr/risk_1991_05/pairs_compare.csv"
# Выход стенда — SSD.
CUT_LINES_ROOT="${CUT_LINES_ROOT:-/mnt/hotstore/scan_processing/cut_lines}"
