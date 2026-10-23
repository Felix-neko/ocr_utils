#!/usr/bin/env bash
#
# Общие пути стенда «текст разворотов» (research/spread_ocr). Сам ничего не запускает — подключается через `source`.

# РИСК 1991 №5, две съёмки одного переплетённого тома — исходники на Я.Диске (NTFS-3G, только чтение).
# (1) пак-6 от Лены: 110 кадров 5687×4268, чёрный мат, разворот часто снят дважды (фокус на левой, потом на правой).
RISK_1991_05_LENA="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-6 от Лены (1989-1992)/РИСК/1991/05 есть фотографии по страницам"
# (2) пак-7 от Бори: 59 кадров 4622×3584, красно-чёрный фон, у корешка текст повреждён.
RISK_1991_05_BORYA="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-7 от Бори/РИСК/1991/05"
# Пары кадров по ключевым точкам (AKAZE/SIFT) стенда research/cut_lines — для сверки.
RISK_1991_05_AKAZE_PAIRS="/mnt/hotstore/scan_processing/cut_lines/risk_1991_05/pairs.csv"

# Выход стенда — SSD hotstore.
SPREAD_OCR_ROOT="${SPREAD_OCR_ROOT:-/mnt/hotstore/scan_processing/spread_ocr}"
