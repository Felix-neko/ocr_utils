#!/usr/bin/env bash
#
# Скрининг фокуса ЭГ 1968 04-06 после подмены пересъёмкой: копия «ЭГ/в работе/1968/04-06 готово ХП» на SSD
# (/mnt/hotstore/scan_processing/newspapers/eg/1968_04-06, 626 RAF), где 54 страницы № 14–17 с текстом под
# корешком заменены пересъёмкой (*.replacement.RAF, 4 из них *.replacement.gutter_loss.light.RAF) —
# replace_reshoots.py, run_scripts/select_best_raws/run_replace_eg_68_04_06.sh. Алгоритм DOM, без GPU.
#
# Читает: RAF копии; скрытые подпапки .gutter_loss_for_debugging/ (54 заменённых оригинала) и
#         .defocus_for_debugging/ (2) не обходятся.
# Пишет:  defocus_dom_eg_68_04_06_replaced.{txt,md,csv}, _retake.txt и папку симлинков _worst/ в корне репо.
#
# ПОСТРАНИЧНАЯ СЪЁМКА (кадры вертикальные, EXIF Rotate 270 CW у оригиналов): текст в кадре вдвое крупнее,
# чем при съёмке разворотом. Пресет dom-si калиброван на разворотах — абсолютные теги ориентировочные
# (ср. ЭГ 1962 сентябрь, run_defocus_tagged_eg_279_62_9.sh: медиана DOM 3.160). Верх списка займут крышки,
# форзацы, первые полосы с фото и рекламные полосы — отсматривать глазами. Пересъёмка снята в другом месте
# и при другом свете: её баллы стоит сравнить с соседними оригиналами (в CSV — по именам *.replacement*).
#
# Настройки и их обоснование — как в run_defocus_tagged_eg_1982_1_3.sh (подробная шапка там): пресет
# dom-si (тяжёлый < 2.92, средний < 2.96, лёгкий < 3.00, ультралёгкий < 3.04), худшие 30 %, зональный
# отчёт 30 % по всем направлениям.
#
set -euo pipefail

# Скрипт лежит в run_scripts/<подсистема>/, пути внутри отсчитываются от корня репозитория.
cd "$(dirname "$0")/../.."

INPUT_DIR="/mnt/hotstore/scan_processing/newspapers/eg/1968_04-06"
BASE_NAME="defocus_dom_eg_68_04_06_replaced"

echo "defocus_detection (DOM, абсолютные пороги + худшие 30%):"
echo "  input = $INPUT_DIR"
echo "  отчёты = ${BASE_NAME}.{txt,md,csv}, ${BASE_NAME}_retake.txt, ${BASE_NAME}_worst/"

# Аргументы массивом — так каждую опцию можно прокомментировать
ARGS=(
    "$INPUT_DIR"
    # Абсолютные пороги тяжести; сам ставит --algorithm dom и --aggregate best
    --tag-preset dom-si
    # Отчёт 1: худшие 30 % папки, от самых плохих сверху
    --worst-percent 30
    # Отчёт 2: зональный расфокус (поплыла часть кадра), все четыре направления
    --zonal-percent 30
    --zonal-axis all
    # Отчёты: текст, markdown с ссылками, CSV по ВСЕМ кадрам, плоский список на пересъёмку
    --txt-report "${BASE_NAME}.txt"
    --md-report "${BASE_NAME}.md"
    --csv "${BASE_NAME}.csv"
    --retake-list "${BASE_NAME}_retake.txt"
    # Симлинки для XnView: overall/, zonal/, defocus_heavy|medium|light|ultralight/
    --link-dir "${BASE_NAME}_worst"
    # Сколько ядер оставить системе (воркеры с nice +10). Данные на SSD — диск не узкое место
    --reserve-cores 4
)
uv run python -m ocr_utils.defocus_detection "${ARGS[@]}"
