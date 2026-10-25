#!/usr/bin/env bash
#
# Шаг 8: промежуточные PDF (1970, выпуски 07-09) под пакетное распознание в FineReader.
#
# Всё как у паков 1-2 (run_scripts/pdf_utils/run_intermediate_pdfs_mts_pack1.sh — там разбор, зачем поля и почему две
# PDF): полная PDF с полями 12/6 мм под рост кадра при распрямлении строк и PAGES_WITH_PICS_ONLY без полей. Поля
# заданы в мм бумаги — к 450 dpi не пересчитываются.
#
# ДО СБОРКИ закрыть вопрос о DPI (common.sh): физический размер страницы в PDF = пиксели / dpi полосы. При теге 450 и
# настоящих ~425 страница выйдет на 6 % меньше оригинала, и FineReader оценит кегль по этой мере.
#
# Перед сборкой — сверка заострённых копий с очищенными полосами по содержимому (verify_sharpened, пороги пака-1):
# Capture One выгружает пак плоско, раскладку по выпускам делают руками, подменённая полоса прошла бы сборку молча.
#
# ОБЪЁМ. 297 полос; выпуск держится в памяти целиком. Минуты.

set -euo pipefail
# Оба шага гоняют пул процессов: при остановке скрипта гасить всю группу, без сирот-воркеров.
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Сверка упирается в декодирование TIFF — по числу физических ядер. Сборка — выпуск на воркер, выпусков 3.
VERIFY_JOBS="${VERIFY_JOBS:-16}"
JOBS="${JOBS:-3}"

source "$(dirname "$0")/common.sh"

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
    # --only-issue 07

    # Качество врезанных иллюстраций — как у паков 1-2: полуфабрикат, глазами по нему проверяют картинки.
    --picture-quality 90

    # Поля полной PDF — замер вылета кадра FineReader по паку-1 (см. скрипт пака-1).
    --page-margin-x-mm 12
    --page-margin-y-mm 6

    --jobs "$JOBS"
    --log-level INFO
)
uv run python -m ocr_utils.pdf_utils.intermediate_pdfs "${PDF_ARGS[@]}" "$@"
