#!/usr/bin/env bash
#
# Общие параметры обработки досканов пака-1 (журнал «Материально-техническое снабжение», 1970, выпуски 07-09).
#
# Что это. Пересъёмка трёх выпусков, которые в паке-1 (Viisan S21, 300 dpi с апскейлом до 600) вышли плохо, на
# другой док-камере — Viisan K48, без апскейла. 3 выпуска по 99 полос (297), TIFF RGB LZW ~2690x4240 px, 3,3 ГБ.
# Первая полоса выпуска — обложка (проверено глазами на всех трёх), оглавление — третья.
#
# РАЗРЕШЕНИЕ. Тег TIFF — 450 dpi, его и берёт весь конвейер (мм -> px по dpi полосы, docs/data_layout.md,
# «Разрешение сканов»): делитель превью CVAT round(450/75) = 6, копия растрового детектора — 150 dpi. Замер
# 2026-10-18 (сессия 16ab6160: шаг строк, x-height и размеры текстового блока по ~290 полосам против тех же
# выпусков пака-1) даёт относительно S21 = 300 dpi ~425 dpi, то есть тег завышен на 5-6 %. Кто из двух камер
# отклонился, по паку не определить — нужна линейка по полосе (ширина текстового блока 1970 г.: 130 мм -> 425 dpi,
# 123 мм -> 450). Для разметки и очистки 6 % в размерах, заданных в мм, несущественны; для промежуточных PDF
# (физический размер страницы) вопрос стоит закрыть до сборки.
#
# ДЕРЕВО — {год}/{выпуск}/{полоса} (папку 1970/ пользователь завёл руками 2026-10-19: обход scan_markup требует
# уровень года). Имена полос — NNNN_1L.tif / NNNN_2R.tif, файлы .ScanTailor пропускаются по расширению.
#
# Сам ничего не запускает: подключается через `source` из соседних run_*.sh. Пути собраны здесь потому, что база
# связывает шаги между собой — разъехавшееся имя в одном из них заведёт ещё один пак вместо продолжения этого.

PACK_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/Готовое/досканы-1 1970-07 - 1970-09 (450 DPI)"
# Имя пака в базе, проекта в CVAT и подпапки в share. ЗАДАВАТЬ ЯВНО во всех командах: без --pack-name оно берётся
# из имени папки. С паком-1 не пересекается: база своя, проект CVAT свой.
PACK_NAME="досканы-1"

# Рабочие файлы разметки — на SSD рядом с проектом (превью CVAT листаются постранично, NTFS-3G для этого медленный).
# Своя подпапка, как у пака-2: плоско в корне лежит пак-1.
MARKUP_ROOT="/home/felix/Projects/mts_markup"
PACK_WORK="$MARKUP_ROOT/pack1_rescan"

DB="$PACK_WORK/pack1_rescan.sqlite"
DB_REVIEWED="$PACK_WORK/pack1_rescan_reviewed.sqlite"
DEBUG_DIR="$PACK_WORK/debug"

# Промежуточные файлы — на SSD /mnt/hotstore, все под одним корнем пака.
HOT_PACK_ROOT="/mnt/hotstore/scan_processing/mts/pack1_rescan"

# Кэш surya layout (ocr_utils.page_layout.surya.SuryaCache): <корень>/<вариант>/<полоса>.json. Вариант scan
# набивает detect сам (модель зовётся на полосу без записи, ~1 с GPU); fr_geo / fr_nogeo — run_9.
LAYOUT_CACHE_DIR="$HOT_PACK_ROOT/page_layout"

# Кэш разбора полос DeepSeek-OCR-2 (тот же формат, свой корень): блоки image — затравки line art
# (page_layout.line_art.deepseek.page_cache). Набивает run_0_prefill_deepseek.sh; рабочая папка — кадры PNG и
# ответы воркера vLLM, после набивки её можно удалить.
DEEPSEEK_CACHE_DIR="$HOT_PACK_ROOT/deepseek_layout"
DEEPSEEK_WORK_DIR="$HOT_PACK_ROOT/deepseek_work"

# Очистка пака (шаг 7): закрас разметки и размытие фона. Не на /mnt/dump3: Яндекс.Диск переименовывает новые
# файлы поверх исходных.
CLEAN_ROOT="$HOT_PACK_ROOT/background_blurred"
BLURRED_DIR="$CLEAN_ROOT/blurred"
CLEAN_DEBUG_DIR="$CLEAN_ROOT/debug"

# Заострённые копии из Capture One. Он выгружает их плоско ({выпуск}{полоса}.jpg), их раскладывают руками в ту же
# схему {год}/{выпуск}/полоса, что у blurred: имя — как у очищенной полосы, только .jpg.
SHARPENED_DIR="$CLEAN_ROOT/sharpened"

# Промежуточные PDF под FineReader — по паку одной папкой (Hot Folder берёт папку целиком).
PDF_ROOT="$HOT_PACK_ROOT/pdf"
FULL_PDF_DIR="$PDF_ROOT/full_intermediate_pdfs"
PICS_ONLY_PDF_DIR="$PDF_ROOT/intermediate_pdfs_pages_with_pics_only"

# Куда FineReader положил распознанное: два прогона по одним полным промежуточным PDF — с коррекцией геометрии и
# без неё. Имена файлов — как у промежуточных (full_1970_NN.pdf).
GEO_PDF_DIR="$PDF_ROOT/binarized_geometry_on"
NOGEO_PDF_DIR="$PDF_ROOT/binarized_geometry_off"

# Финальные PDF ({год}/{год}_{выпуск}.pdf) и рабочий каталог сборщика (JSON анализа на страницу, CSV, превью).
FINAL_PDF_DIR="$PDF_ROOT/final_pdfs"
FINAL_WORK_DIR="$HOT_PACK_ROOT/final_pdfs_work"

# Детектор порчи геометрии FineReader v18 (ocr_utils.geometry_regression.quality): разбор page_layout обоих PDF
# (run_9), кэш движка v16 и прогон v18 (run_10), оверлеи по вердиктам (run_12).
GEOMETRY_LAYOUT_ROOT="${GEOMETRY_LAYOUT_ROOT:-$HOT_PACK_ROOT/page_analysis_fr}"
GEOMETRY_QUALITY_ROOT="$HOT_PACK_ROOT/geometry_quality"
GEOMETRY_V16_DIR="${GEOMETRY_V16_DIR:-$GEOMETRY_QUALITY_ROOT/v16}"
GEOMETRY_V18_DIR="${GEOMETRY_V18_DIR:-$GEOMETRY_QUALITY_ROOT/v18}"

# Должен лежать ВНУТРИ IMAGES_DIR из docker/.env (или совпадать с ним), иначе cvat_server не увидит картинок.
# Общий с паками 1-2: to-cvat сам кладёт превью в подпапку $PACK_NAME и заводит отдельный проект.
SHARE_ROOT="$MARKUP_ROOT/cvat_share"

# Все шаги запускаются из корня репозитория (pyproject.toml). BASH_SOURCE, а не $0: путь нужен именно этого файла.
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

mkdir -p "$PACK_WORK"

echo "Пак:      $PACK_DIR"
echo "База:     $DB"

# Бэкап базы перед любым шагом, который её правит: внутри — ручная разметка из CVAT, пересобрать её нельзя.
if [ -f "$DB" ]; then
    cp -f "$DB" "$DB.bak"
fi
