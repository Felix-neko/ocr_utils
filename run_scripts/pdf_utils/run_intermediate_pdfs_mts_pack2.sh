#!/usr/bin/env bash
#
# Промежуточные PDF пака-2 (1977, выпуски 01-06) под пакетное распознание в FineReader.
#
# Всё как у пака-1 (run_intermediate_pdfs_mts_pack1.sh — там же разбор, зачем поля и почему две PDF):
# полная PDF с полями 12/6 мм под рост кадра при распрямлении строк и PAGES_WITH_PICS_ONLY без полей.
# Полосы пака-2 апскейлены до 600 dpi и по шагу строк и штриху совпадают с паком-1, поэтому поля
# взяты те же, без пересчёта.
#
# Перед сборкой — сверка заострённых копий с очищенными полосами по содержимому
# (verify_sharpened, пороги пака-1): Capture One выгружает пак плоско, раскладку по выпускам
# делали руками, и подменённая полоса того же размера прошла бы сборку молча.
#
# ОБЪЁМ. 597 полос, заострённые копии — 6,9 ГиБ, столько же примерно займут полные PDF.
# Выпуск держится в памяти целиком (~100 полос по ~12 МиБ), при 6 воркерах — до 10 ГиБ. Минуты.

set -euo pipefail
# Оба шага гоняют пул процессов: при остановке скрипта гасить всю группу, без сирот-воркеров.
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM

# Сверка упирается в декодирование TIFF — по числу физических ядер. Сборка — выпуск на воркер,
# выпусков 6, больше воркеров не нужно (и память: ~1,5 ГиБ на выпуск).
VERIFY_JOBS=16
JOBS=6

source "$(dirname "$0")/../scan_markup/pack2/common.sh"

# Колонки полей промежуточных PDF дописывает шаг 8 (миграция), см. скрипт пака-1.
uv run python - "$DB_REVIEWED" <<'PYTHON'
import sqlite3
import sys

columns = {row[1] for row in sqlite3.connect(sys.argv[1]).execute("PRAGMA table_info(issues)")}
if "full_intermediate_pdf_margin_left_mm" not in columns:
    sys.exit(
        "В базе нет колонок полей промежуточных PDF. Сначала выполнить:\n"
        "    run_scripts/scan_markup/pack2/run_8_migrate_db.sh"
    )
PYTHON

# Сборщик пишет в базу номера страниц полос в PDF и поля выпусков: бэкап разобранной базы перед этим.
cp -f "$DB_REVIEWED" "$DB_REVIEWED.bak"

echo "Оригиналы:   $BLURRED_DIR"
echo "Заострённые: $SHARPENED_DIR"
echo "Выход:       $FULL_PDF_DIR"
echo "             $PICS_ONLY_PDF_DIR"
df -h "$CLEAN_ROOT" | tail -1

VERIFY_ARGS=(
    --db "$DB_REVIEWED"
    --pack-name "$PACK_NAME"
    --originals-dir "$BLURRED_DIR"
    --sharpened-dir "$SHARPENED_DIR"
    --jobs "$VERIFY_JOBS"
    --report-csv "$CLEAN_ROOT/verify_sharpened.csv"
    --log-level INFO
)
uv run python -m ocr_utils.pdf_utils.verify_sharpened "${VERIFY_ARGS[@]}"

PDF_ARGS=(
    --db "$DB_REVIEWED"
    --pack-name "$PACK_NAME"
    --originals-dir "$BLURRED_DIR"
    --sharpened-dir "$SHARPENED_DIR"
    --full-pdf-dir "$FULL_PDF_DIR"
    --pics-only-pdf-dir "$PICS_ONLY_PDF_DIR"

    # Проба одним выпуском — вернуть строку для быстрого круга.
    # --only-issue 01

    # Качество врезанных иллюстраций — как у пака-1: полуфабрикат, глазами по нему проверяют картинки.
    --picture-quality 90

    # Поля полной PDF — замер вылета кадра FineReader по паку-1 (см. скрипт пака-1).
    --page-margin-x-mm 12
    --page-margin-y-mm 6

    --jobs "$JOBS"
    --log-level INFO
)
uv run python -m ocr_utils.pdf_utils.intermediate_pdfs "${PDF_ARGS[@]}" "$@"
