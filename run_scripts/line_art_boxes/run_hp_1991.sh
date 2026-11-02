#!/usr/bin/env bash
#
# Стенд research/line_art_boxes на ХиП 1991 (выпуски 01–07 прогона v2 с растяжением у корешка, остановлен 2026-10-25):
# холсты первого преобразования (освещение + Кунс) и вырезки страниц, как их видит детектор line art в
# rectify.text_field.page_regions → разбор line art боевым кодом и вариантами прототипа → листы отсмотра и «было | стало».
# Плюс регрессия на паке-1 с surya из кэша (путь scan_markup / final_pdfs): 30 PDF, каждая 4-я страница.
# Читает: Я.Диск (кадры, только чтение), кэш и итоги v2 $GUTTER_DECOMPRESSION_ROOT/hp_1991, PDF пака-1 и кэш surya.
# Пишет: $LINE_ART_BOXES_ROOT/{hp_1991,pack1_regress}.
# Время: canvas ~20 мин на 559 кадров (8 воркеров, ~8 с на кадр), detect ~5 мин на вариант-пару, regress ~15 мин.
# Плотность ХиП 1991 по выпускам — шапка run_v2_hp_1991.sh (как в run_v2_gutter_decompression.sh).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/line_art_boxes/common.sh

# Не больше 3: рядом идут прогоны v2 и замеры основной сессии (2026-10-25: стенду — 2–3 ядра и до 16 ГБ ОЗУ).
JOBS="${JOBS:-3}"
OUT="$LINE_ART_BOXES_ROOT/hp_1991"
PX="01=17.9,02=17.1,03=17.3,04=17.2,05=17.0,06=17.3,07=16.9,08=16.6,09=16.6,10=16.6,11=16.3,12=16.3"
VARIANTS="legacy,no_strokes,base,grow,ring,close"

uv run python -m research.line_art_boxes canvas --photos "$HP_1991_ROOT" --run "$GUTTER_DECOMPRESSION_ROOT/hp_1991" \
    --out "$OUT/canvas" --px-per-mm "$PX" --jobs "$JOBS"
uv run python -m research.line_art_boxes detect --canvas "$OUT/canvas" --out "$OUT/detect" --variants "$VARIANTS" \
    --jobs "$JOBS"
uv run python -m research.line_art_boxes sheets --canvas "$OUT/canvas" --detect "$OUT/detect" --out "$OUT/sheets/base" \
    --variant base
for AFTER in ring close; do
    uv run python -m research.line_art_boxes sheets --canvas "$OUT/canvas" --detect "$OUT/detect" \
        --out "$OUT/sheets/compare_$AFTER" --variant base --compare "$AFTER" --columns 6 --per-sheet 6
done
uv run python -m research.line_art_boxes report --detect "$OUT/detect" --labels research/line_art_boxes/sets/hp_1991_base.csv
# Полнота рамок: листы разметки (компоненты кольца с номерами) и мера по sets/hp_1991_gt.csv.
uv run python -m research.line_art_boxes gt-sheets --canvas "$OUT/canvas" --labels research/line_art_boxes/sets/hp_1991_base.csv \
    --out "$OUT/sheets/gt" --objects "$OUT/gt_objects.json"
uv run python -m research.line_art_boxes coverage --canvas "$OUT/canvas" --detect "$OUT/detect" --objects "$OUT/gt_objects.json" \
    --gt research/line_art_boxes/sets/hp_1991_gt.csv --variants no_strokes,base,ring
# Перенос областей второго преобразования за содержимым (TextParams.carry_regions): 24 кадра, «было | стало».
uv run python -m research.line_art_boxes carry --photos "$HP_1991_ROOT" --run "$GUTTER_DECOMPRESSION_ROOT/hp_1991" \
    --out "$OUT/carry" --px-per-mm "$PX" --jobs "$JOBS" --frames "01/IMG_1050,01/IMG_1071,02/IMG_1175,03/IMG_0144,\
01/IMG_0986,01/IMG_1004,01/IMG_1013,01/IMG_1042,01/IMG_1045,03/IMG_0119,04/IMG_0202,04/IMG_0222,04/IMG_0236,\
05/IMG_0308,05/IMG_0325,05/IMG_0332,06/IMG_0419,06/IMG_0457,02/IMG_1091,02/IMG_1149,02/IMG_1188,04/IMG_0154,\
05/IMG_0313,06/IMG_0377"

uv run python -m research.line_art_boxes regress --pdf-dir "$NOGEO_PDF_DIR" --surya-cache "$LAYOUT_CACHE_DIR" \
    --out "$LINE_ART_BOXES_ROOT/pack1_regress/detect" --variants legacy,base --step 4 --pdf-limit 0 --jobs "$JOBS"
uv run python -m research.line_art_boxes sheets --detect "$LINE_ART_BOXES_ROOT/pack1_regress/detect" \
    --out "$LINE_ART_BOXES_ROOT/pack1_regress/sheets" --variant legacy --compare base --columns 6 --per-sheet 6
uv run python -m research.line_art_boxes report --detect "$LINE_ART_BOXES_ROOT/pack1_regress/detect"
