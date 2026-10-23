#!/usr/bin/env bash
#
# Стенд research/page_rectify text-run по итогам прогона v2 «spline_frames» (run_v2_spline_bends.sh, 2026-10-22):
# выравнивание освещения не-цветных страниц итога v2 (как Equalize illumination в ScanTailor: полином фона 8×5),
# выпрямление подгоночных областей страниц поверхностью Кунса, затем второе преобразование — 3 прохода «разбор текста →
# поле» со сложением смещений. Разбор — page_layout.text_blocks, движок по краске (CPU), по копии с утолщёнными на
# 0.25 мм штрихами (тонкий шрифт ПХ 1975 иначе уходит в «черты»). Поле — своё у каждой страницы (разрыв по корешку),
# тонкая пластина на сетке 3 мм: вдоль строк длина изгиба 10 мм, у корешка 3 мм, поперёк строк 8 мм; выборки — оси
# строк (в горизонталь, на концах гуще) и выровненные стороны блоков (общая вертикаль отбивки колонки). Пиксели — из
# итога v2, одна выборка по составной карте. Числа и случаи — research/page_rectify/README.md.
# Целиком цветные страницы (FULL) — без освещения и без поля. Наборы те же: ph_gutter_strip, hp_1991, ph_1975.
#
# SAMPLE=N — по N кадров на набор (ХиП — по N/12 на выпуск, не меньше 2), выход $TEXT_RECTIFIED_SAMPLE_WORK; без
# SAMPLE — все кадры в $TEXT_RECTIFIED_WORK. Внутри: output/<набор>/…tiff, debug/<шаг>/<набор>/…jpg (01_illumination,
# 02_analysis, 03_field, 04_result), summary/<набор>.csv (ХиП — <набор>/<выпуск>).
# Читает: Я.Диск (оригиналы, только чтение), кэш и итоги v2 на SSD. Пишет: SSD hotstore.
# JOBS=12, не 16: кадр читает оригинал с Я.Диска (NTFS-3G) и держит ~2 ГБ картинок, карт и разборов.
# Время: разворот ~25 с в одном процессе (разбор текста 4×2 страниц — 17 с, области по кэшу — 3 с, составные карты
# на сетке 4 px — 2 с); под нагрузкой 12 воркеров — больше. 2016 кадров — ~1.5–2 ч.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-12}"
SAMPLE="${SAMPLE:-0}"
if [[ "$SAMPLE" -gt 0 ]]; then
    OUT_ROOT="${OUT_ROOT:-$TEXT_RECTIFIED_SAMPLE_WORK}"
    HP_SAMPLE=$(( SAMPLE / 12 > 2 ? SAMPLE / 12 : 2 ))
else
    OUT_ROOT="${OUT_ROOT:-$TEXT_RECTIFIED_WORK}"
    HP_SAMPLE=0
fi

# Все наборы — одним вызовом: кадры всех наборов идут в один пул (при поочерёдных наборах выборка ХиП по 2 кадра на
# выпуск занимала 2 воркера из 12). Набор: подпись (подпуть в output/, debug/<шаг>/, summary/), вход, кэш v2, итоги v2,
# сколько кадров (0 — все).
SETS=(--set ph_gutter_strip "$PH_GUTTER_STRIP_SET" "$PH_GUTTER_STRIP_WORK/cache" "$SPLINE_FRAMES_WORK/ph_gutter_strip/output" "$SAMPLE")
for DIR in "$HP_1991_ROOT"/*/; do
    NAME="$(basename "$DIR")"
    SETS+=(--set "hp_1991/$NAME" "$DIR" "$HP_1991_WORK/cache/$NAME" "$SPLINE_FRAMES_WORK/hp_1991/$NAME/output" "$HP_SAMPLE")
done
SETS+=(--set ph_1975 "$PACK4_PH_ROOT/1975" "$PACK4_PH_WORK/cache/1975" "$SPLINE_FRAMES_WORK/ph_1975/output" "$SAMPLE")

mkdir -p "$OUT_ROOT"
echo "=== page_rectify text-run → $OUT_ROOT"
uv run python -m research.page_rectify text-run \
    "${SETS[@]}" \
    --root "$OUT_ROOT" \
    --field-step-mm 3 \
    --gutter-smooth-mm 3 \
    --skip-if-exists \
    --jobs "$JOBS"
