#!/usr/bin/env bash
#
# Прогон детектора просвечивающей бумаги по всему паку Сафронова «Плановое хозяйство»
# (22951 кадр, 1931–1971, разложены год/выпуск). Результат — список разворотов,
# которые стоит пересканировать с другого экземпляра.
#
# ВРЕМЯ. Замерено: около 0.5 с на полосу в один поток при 300 dpi, то есть ~1 с на кадр.
# На 12 воркерах это порядка 35 минут счёта. Реально дольше: пак лежит на /mnt/dump3
# (NTFS-3G на шпинделе), и 39 ГБ JPEG читаются оттуда не мгновенно. Прогон стоит
# запускать фоном и ждать ПО СОХРАНЁННОМУ PID:
#
#     bash run_scripts/show_through_detection/run_show_through_ph_pack.sh & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 60; done
#
# ПОРОГ. По умолчанию — калибровочный (см. run_show_through_ph_1931_02_03.sh). Бумага
# в паке меняется от года к году, поэтому первым делом стоит посмотреть на распределение
# в CSV по годам: если доля выше порога в каком-то году резко выбивается, это либо
# действительно плохая бумага того года, либо повод подвинуть --threshold.
#
set -euo pipefail

cd "$(dirname "$0")/../.."

INPUT_DIR="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/пак Сафронова/для нарезки разворотов (300 DPI)"
BASE_NAME="show_through_ph_pack"
REPORT_TXT="${BASE_NAME}.txt"
REPORT_MD="${BASE_NAME}.md"
REPORT_CSV="${BASE_NAME}.csv"
LINK_DIR="${BASE_NAME}_worst"

echo "show_through_detection (весь пак):"
echo "  input = $INPUT_DIR"
echo "  csv   = $REPORT_CSV"

ARGS=(
    "$INPUT_DIR"

    # Пак разложен год/выпуск — без рекурсии не будет найдено вообще ничего.
    --recursive

    # Метрика по умолчанию: доля межстрочий, переживающая предварительную бинаризацию.
    # Порог откалиброван на паре «1931/02-03 (брак) против 1955/03 (просвет есть, но
    # обработка его снимает)»: см. show_through_detection_report.md.
    --algorithm ghost_ink
    --threshold 1.0

    # Отбор не задаём: в отчёт идут ровно те полосы, что перешагнули порог. На 23 тысячах
    # кадров любой процентный отбор дал бы список, который никто не станет смотреть.

    --txt-report "$REPORT_TXT"
    --md-report "$REPORT_MD"
    --csv "$REPORT_CSV"
    --link-dir "$LINK_DIR"

    --reserve-cores 2
)

uv run python -m ocr_utils.show_through_detection "${ARGS[@]}"
