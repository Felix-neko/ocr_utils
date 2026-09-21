#!/usr/bin/env bash
#
# Общие параметры разметки пака-1 (журнал «Материально-техническое снабжение», 1966-1976).
#
# Пак: 11 годовых комплектов, 123 выпуска, 12 136 полос, TIFF RGB 600 dpi по ~40 МБ.
# Оригиналы лежат на /mnt/dump3 (NTFS-3G на шпинделе), полное чтение — около полутерабайта.
#
# Сам ничего не запускает: подключается через `source` из соседних run_*.sh. Пути собраны
# здесь потому, что база связывает три шага между собой — разъехавшееся имя в одном из них
# заведёт второй пак вместо продолжения первого, и заметить это можно очень нескоро.

PACK_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/Готовое/пак-1"
PACK_NAME="пак-1"

# Рабочие файлы разметки — на SSD рядом с проектом, а не на /mnt/dump3. Оригиналы читаются
# по разу, а вот уменьшенные копии CVAT листает постранично, и на шпиндельном NTFS-3G это
# ощутимо медленнее. Объём небольшой: ~12 тыс. превью по 75 dpi плюс отладочные оверлеи.
MARKUP_ROOT="/home/felix/Projects/mts_markup"

DB="$MARKUP_ROOT/pack1.sqlite"
DB_REVIEWED="$MARKUP_ROOT/pack1_reviewed.sqlite"
DEBUG_DIR="$MARKUP_ROOT/debug"

# Кэш surya layout (ocr_utils.page_layout.surya.SuryaCache): <корень>/<вариант картинки>/<полоса>.json.
# Вариант — часть ключа: scan (сырые TIFF «Готовое»), sharpened (заострённые JPEG), fr_geo и
# fr_nogeo (страницы PDF FineReader с коррекцией геометрии и без — их читают детектор порчи
# геометрии и правка текстового слоя). Варианты scan и sharpened перенесены из старых pickle
# (pack1_table_research/layout_surya_готовое и layout_surya, 2026-09-21) записями legacy —
# полны на все 12 135 полос; fr_* набиваются `page_layout prefill-surya` (~0.7 с GPU на страницу).
# detect берёт разметку отсюда и модель не зовёт, а полосу без записи размечает и дописывает сюда же.
LAYOUT_CACHE_DIR="/mnt/system/raw/mts/pack1_page_layout"

# Оглавления (шаг 1, команда toc): признаки полос окна, контактные листы для разметки эталона
# и списки полос оглавления по выпускам для внешнего OCR (--pages / --skip-pages).
TOC_DIR="$MARKUP_ROOT/toc"
TOC_LISTS_DIR="$TOC_DIR/lists"
TOC_LABELS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/toc_labels.csv"

# Валидационная выборка: папки с примерами неправильной разметки, надёрганными из DEBUG_DIR
# глазами. Имя папки называет тип дефекта, имена файлов внутри — имена оверлеев.
CASES_DIR="$MARKUP_ROOT/некоторые проблемные картинки"
VALIDATE_DIR="$MARKUP_ROOT/validate"

# Очистка пака (шаги 5-7): закрас разметки и размытие фона. Результат — на SSD, а не
# на /mnt/dump3: выход весит примерно столько же, сколько вход (~300 ГиБ), и писать его
# на шпиндельный NTFS-3G значило бы упереться в диск на всём прогоне. Плюс корень
# /mnt/dump3 синхронит Яндекс.Диск, а он переименовывает новые файлы поверх исходных.
# Регистр в /mnt/system ЗНАЧИМ: с 2026-09-20 том смонтирован как /mnt/system строчными; прежний /mnt/SYSTEM заглавными больше не существует.
CLEAN_ROOT="/mnt/system/raw/mts/pack1_background_blurred_v2"
BLURRED_DIR="$CLEAN_ROOT/blurred"
CLEAN_DEBUG_DIR="$CLEAN_ROOT/debug"

# Заострённые копии: Capture One выгружает их в подпапки blurred/{год}/{выпуск}/sharpened,
# а run_collect_sharpened переносит сюда, в привычную раскладку {год}/{выпуск}/полоса.
SHARPENED_DIR="$CLEAN_ROOT/sharpened"

# Промежуточные PDF под FineReader: по паку, а не по годам — распознание идёт пакетом по
# папке, и раскладка по годам означала бы одиннадцать отдельных заданий вместо одного.
PDF_ROOT="/mnt/system/raw/mts/pack1_pdf"
FULL_PDF_DIR="$PDF_ROOT/full_intermediate_pdfs"
PICS_ONLY_PDF_DIR="$PDF_ROOT/intermediate_pdfs_pages_with_pics_only"

# Куда FineReader положил распознанное: два прогона по одним полным промежуточным PDF —
# с коррекцией геометрии (перекос, искажение строк, трапеция) и без неё. Без коррекции
# геометрия страницы = скан + поля, только туда можно точно вернуть иллюстрации. Финальный
# PDF собирается постранично из обоих (run_final_pdfs.sh).
GEO_PDF_DIR="$PDF_ROOT/full_pdfs_binary_no_bg_brightening"
NOGEO_PDF_DIR="$PDF_ROOT/full_pdfs_binary_no_bg_brightening_no_geometry_correction"

# Финальные PDF ({год}/{год}_{выпуск}.pdf, по папке на год) и рабочий каталог сборщика
# (JSON анализа на страницу, CSV, превью) — SSD. Сборка — run_final_pdfs.sh здесь же.
FINAL_PDF_DIR="$PDF_ROOT/final_pdfs"
FINAL_WORK_DIR="/mnt/system/raw/mts/pack1_final_pdfs_work"

# Прогон детектора порчи геометрии по паку (run_scripts/geometry_regression): сборщик финальных
# PDF берёт cache/<pdf>/pNNN.json той же версии детектора как есть, страницы без записи меряет
# на месте (~3 с) и дописывает в этот же кэш. Перекрыть можно переменной окружения:
# GEOMETRY_RUN_DIR=... ./run_final_pdfs.sh
GEOMETRY_REGRESSION_ROOT="/mnt/system/raw/mts/pack1_geometry_regression"
GEOMETRY_RUN_DIR="${GEOMETRY_RUN_DIR:-$GEOMETRY_REGRESSION_ROOT/pack1_v14}"

# Сравнения параметров — рядом с рабочими файлами разметки: их смотрят глазами, они
# невелики и живут ровно до выбора параметров.
COMPARE_DIR="$MARKUP_ROOT/compare"

# Должен лежать ВНУТРИ IMAGES_DIR из docker/.env (или совпадать с ним), иначе cvat_server
# не увидит картинок: задачи заводятся из смонтированного share, по сети файлы не передаются.
SHARE_ROOT="$MARKUP_ROOT/cvat_share"

# Все шаги запускаются из корня репозитория: там pyproject.toml, по которому uv собирает
# окружение. BASH_SOURCE, а не $0: путь нужен именно этого файла, а не того, кто его подключил.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

echo "Пак:      $PACK_DIR"
echo "База:     $DB"

# Бэкап базы перед любым шагом, который её правит. Файл маленький (единицы мегабайт), а
# внутри лежит ручная разметка из CVAT, которой на диске больше нигде нет: пересобрать её
# нельзя, она делается руками неделями.
if [ -f "$DB" ]; then
    cp -f "$DB" "$DB.bak"
fi
