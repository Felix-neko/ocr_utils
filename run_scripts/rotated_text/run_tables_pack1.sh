#!/usr/bin/env bash
#
# Таблицы пака-1 с повёрнутым текстом: найти боковые ячейки, прочитать, набрать прямо,
# посчитать требуемый DPI; листы «было-стало» для проверки глазами.
#
# ВХОД — детекции таблиц из базы разметки (rect_regions.kind='table', 903 штуки) и
# заострённые копии полос на SSD (те же, что уходят в промежуточные PDF).
#
# ВЫХОД — на SSD, не в корне Яндекс.Диска: after/ (результат в конечном dpi), pairs/
# (пары «было-стало»), info/ (JSON по таблице и ячейкам), summary.csv, sheets/, run.log.
#
# ВРЕМЯ. Замер: 2-10 с на таблицу в воркере (сетка + tesseract под каждым углом-кандидатом),
# 903 таблицы при --jobs 16 — около 7 минут; с --surya ещё ~8 минут на GPU.
# Запускать в фоне и ждать ПО СОХРАНЁННОМУ PID (см. CLAUDE.md):
#
#   ./run_scripts/rotated_text/run_tables_pack1.sh & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 30; done
set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

# Снятие ВСЕЙ группы процессов при выходе: пул на forkserver переживает смерть хозяина,
# и убитый по одному PID прогон оставляет forkserver и воркеров сиротами (см. CLAUDE.md).
set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

OUT_DIR="/mnt/system/raw/mts/pack1_rotated_tables"

ARGS=(
    run
    --db "$DB_REVIEWED"          # только чтение: рамки таблиц и пути к полосам
    --pack-name "$PACK_NAME"
    --images-root "$SHARPENED_DIR"
    --out-dir "$OUT_DIR"
    --jobs 16                    # tesseract и морфология — чистый CPU, GPU только у surya
                                 # в родителе между фазами
    --angles 0,90,180,270        # все четыре, а не 0,90,270 пака: перевёрнутый текст в
                                 # ячейке FineReader ломает так же, как боковой
    --lang rus                   # латиница в журнале только в марках; eng подменяет
                                 # похожие кириллические буквы
    --work-dpi 300               # разрешение сетки и чтения: при 150 петит шапки 7 px
    --min-letters 3              # меньше букв — подменять нечем («№№ п/п»)
    --max-dpi 1350               # потолок DPI страницы по условию задачи
    --min-font-px 40             # em шрифта в пикселях, ниже которого распознаватель
                                 # читает неуверенно (10 pt при 300 dpi ≈ 42 px)
    --no-surya                   # второе мнение surya — в бэклоге: читает курсивные бланки,
                                 # которые tesseract не берёт («с 7 по 12», 1968/03 IMG_0114),
                                 # но выдумывает на коротких надписях и стоит ~0,8 с на ячейку;
                                 # фильтры есть (second_opinion), включать --surya
    --pairs-width 1400           # ширина панели в паре «было-стало»
    "$@"
)

echo "Таблицы с повёрнутым текстом → $OUT_DIR"
uv run python -m ocr_utils.rotated_text.tables "${ARGS[@]}"
uv run python -m ocr_utils.rotated_text.tables sheets --out-dir "$OUT_DIR" --per-sheet 4 --width 2000
