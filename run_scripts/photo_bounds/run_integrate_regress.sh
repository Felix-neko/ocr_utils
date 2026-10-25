#!/usr/bin/env bash
#
# Регрессия встраивания фото в выпрямление (scan_cropping/v2/rectify: область фото многоугольником, стороны фото в опорах
# с ограничением веса у текста). Каждый кадр — две настройки боевого text_frame: off (TextParams.photos=False — прежнее
# поведение) и limited (photos=True, PHOTO_NEAR_MM 5, PHOTO_NEAR_WEIGHT 0.5 — умолчания text_field). Меры: текст
# (measure.text_measures) и прямизна сторон фото на итоге (research/photo_bounds/rectify.straightness).
#
# Наборы: ПХ 1974/01 (84 кадра, прогон v2 pages_try/1974/01_оплачено, итоги 01_оплачено_b), ПХ 1975/03, 06, 10 (256 кадров,
# кэш ph/pack_4/cache/1975, итоги и вход pages_try/1975_sel), ХиП 1991 — 3 кадра с фото, МТС 1990–91 пака-6 — 7 кадров
# с фото (run_rectify_demo.sh). Плотность бумаги: ПХ — 13.8 px/мм, ХиП — по выпускам, пак-6 — 13.8.
# Параметры поля — как у прогона v4: шаг 3 мм, у корешка 3 мм.
# Читает: Я.Диск (исходные кадры ПХ 1974 и пака-6, ХиП — только чтение), кэши и итоги v2 на SSD.
# Пишет: $PHOTO_BOUNDS_ROOT/integrate/regress (итоги JPEG, summary.json, summary.csv, листы compare_*).
# Время: кадр ~1.5 мин на две настройки в одном процессе; 350 кадров при JOBS=12 — ~45 мин.
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/photo_bounds/common.sh

# 12, не 16: кадр держит ~3 ГБ карт и копий; ядра — вровень с прочими CPU-прогонами машины.
JOBS="${JOBS:-12}"
OUT="${OUT:-$PHOTO_BOUNDS_ROOT/integrate/regress}"

FRAMES=()
PH74_IN="$PACK4_PH_ROOT/1974/01 оплачено"
PH74_WORK="$PH_PAGES_WORK/1974/01_оплачено"
for F in "$PH74_WORK"_b/output/*.tiff; do
    K="$(basename "$F" .tiff)"
    SRC="$(ls "$PH74_IN/$K".* | head -1)"
    FRAMES+=(--frame ph_1974_01 "$SRC" "$PH74_IN" "$PH74_WORK/cache" "${PH74_WORK}_b/output" 13.8)
done
PH75="$PH_PAGES_WORK/1975_sel"
for SRC in "$PH75"/input/*/*; do
    FRAMES+=(--frame ph_1975 "$SRC" "$PH75/input" "$PACK4_PH_WORK/cache/1975" "$PH75/output" 13.8)
done
for SPEC in "02 оплачено:IMG_1091:17.1" "04 оплачено:IMG_0154:17.2" "06 оплачено:IMG_0377:17.3"; do
    IFS=: read -r N K P <<< "$SPEC"
    FRAMES+=(--frame "hp_1991/$N" "$HP_1991_ROOT/$N/$K.jpg" "$HP_1991_ROOT/$N" "$HP_1991_WORK/cache/$N"
             "$SPLINE_FRAMES_WORK/hp_1991/$N/output" "$P")
done
for REL in "1990/02/IMG_0906" "1990/03 есть фотографии по страницам/IMG_1101" "1990/05/IMG_1273" "1990/09/IMG_0310" \
           "1990/10 есть фотографии по страницам/IMG_0558" "1990/11 есть фотографии по страницам/IMG_0701" \
           "1991/01 есть фотографии по страницам/IMG_0615"; do
    FRAMES+=(--frame pack6 "$PACK6_450_ROOT/МТС/$REL.jpg" "$PACK6_450_ROOT" "$PACK6_V6_WORK/cache" "$PACK6_V6_WORK/output" 13.8)
done

mkdir -p "$OUT"
echo "кадров: $(( ${#FRAMES[@]} / 7 ))"
uv run python -m research.photo_bounds integrate "${FRAMES[@]}" \
    --setting off false 10 0.25 --setting limited true 5 0.5 --out "$OUT" --jobs "$JOBS"
uv run python -m research.photo_bounds integrate-sheets --out "$OUT" --before off --after limited --threshold-mm 0.1
