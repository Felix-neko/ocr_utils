#!/usr/bin/env bash
#
# Общие параметры исследования таблиц с повёрнутым текстом (пак-1).
# Сам ничего не запускает: подключается через `source` из соседних run_*.sh.

# Выгрузка FineReader и распознанные PDF: по ним ищутся испорченные таблицы и их страницы.
DOCX_DIR="/mnt/system/raw/mts/pack1_pdf/docx_форматированный_текст"
RECOGNIZED_PDF_DIR="/mnt/system/raw/mts/pack1_pdf/full_pdfs_binary_no_bg_brightening"

# Заострённые копии сканов, 600 dpi.
SHARPENED_DIR="/mnt/system/raw/mts/pack1_background_blurred_v2/sharpened"

# База разметки — только на чтение, из неё берётся связь «страница PDF → файл скана».
DB="/home/felix/Projects/mts_markup/pack1_reviewed.sqlite"
PACK="пак-1"

# Куда всё складывается. Не в корень /mnt/dump3: его синхронит Яндекс.Диск и переименовывает
# новые файлы поверх исходных. Регистр в /mnt/system значим (с 2026-09-20 том смонтирован как /mnt/system строчными; прежний /mnt/SYSTEM заглавными больше не существует).
OUT_DIR="/mnt/system/raw/mts/pack1_table_research"

# Воркеров. Не 16: каждый воркер разжимает JPEG 600 dpi на 20+ мегапикселей, и прогон
# упирается в память и диск раньше, чем в счёт.
JOBS=12

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
