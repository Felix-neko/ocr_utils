#!/usr/bin/env bash
#
# Общие пути исследования текстового слоя (ocr_utils/text_layer_fix) по паку-1. Сам ничего
# не запускает — подключается через `source` из соседних run_*.sh.

# Пути пака, PDF_ROOT, базы разметки — из общего файла разметки.
source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"

# Распознанные бинаризованные PDF (с коррекцией геометрии) — то, что идёт в финальные PDF,
# и их пара без коррекции геометрии (растр = скан + поля, для сверки с ручной разметкой).
TEXT_LAYER_PDF_DIR="$PDF_ROOT/full_pdfs_binary_no_bg_brightening"
TEXT_LAYER_NOGEO_PDF_DIR="$PDF_ROOT/full_pdfs_binary_no_bg_brightening_no_geometry_correction"

# База-зонд с номерами страниц полос в полных PDF (pages.full_pdf_page_idx); в рабочих
# базах разметки эти номера пусты. Только чтение.
PROBE_DB="$PDF_ROOT/_margin_probe/db.sqlite"

# Прогон rotated_text по таблицам пака: info/<таблица>.json с ячейками и summary.csv.
ROTATED_TABLES_DIR="/mnt/SYSTEM/raw/mts/pack1_rotated_tables"

# CSV прогона line_art_detection по паку (покрытие страницы крупным штрихом).
LINE_ART_CSV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/line_art_pack1.csv"

# Выход — на SSD, подпапка на версию алгоритма; перекрыть: TEXT_LAYER_RUN_DIR=... ./run_x.sh
TEXT_LAYER_FIX_ROOT="/mnt/SYSTEM/raw/mts/pack1_text_layer_fix"
TEXT_LAYER_RUN_DIR="${TEXT_LAYER_RUN_DIR:-$TEXT_LAYER_FIX_ROOT/pack1_v1}"
