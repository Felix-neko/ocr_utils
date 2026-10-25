#!/usr/bin/env bash
#
# Стенд research/photo_bounds: стороны фото как опоры второго преобразования выпрямления (боевой код не меняется —
# подмены text_run в процессе стенда). Каждый кадр — три варианта: base (как в бою), quad (область фото кривым
# четырёхугольником вместо рамки растра — преграда разбора и отсев выборок), sides (quad плюс стороны фото в опорах).
#
# Кадры — все кадры с фото среди итогов стенда page_rectify (rectify-scan по 2016 кадрам v4: ХиП 1991 — 3 кадра
# с фото и двумя страницами, ПХ 1975 — только мраморные форзацы) плюс 7 разворотов МТС 1990–1991 пака-6 (прогон v2
# pack6_v6, rectify-scan по 1390 кадрам МТС: 52 с растром, из них выбраны глазами 7 с фото на текстовых полосах).
# Плотность бумаги: ХиП — по выпускам (run_page_rectify_text.sh), пак-6 — 13.8 px/мм (v2_pack6_args.sh).
# Параметры поля — как у прогона v4: шаг 3 мм, у корешка 3 мм.
# Читает: Я.Диск (исходные кадры, только чтение), кэши и итоги v2 на SSD. Пишет: $PHOTO_BOUNDS_ROOT/rectify/demo.
# Время: кадр ~2.5 мин (три варианта по ~40 с плюс поиск фото на каждом проходе), 10 кадров — ~8 мин при JOBS=4.
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/photo_bounds/common.sh

# 4, не 12: кадр держит ~3 ГБ (три варианта выпрямления с картами).
JOBS="${JOBS:-4}"
OUT="${OUT:-$PHOTO_BOUNDS_ROOT/rectify/demo}"

FRAMES=()
for SPEC in "02 оплачено:IMG_1091:17.1" "04 оплачено:IMG_0154:17.2" "06 оплачено:IMG_0377:17.3"; do
    IFS=: read -r N K P <<< "$SPEC"
    FRAMES+=(--frame "hp_1991/$N" "$HP_1991_ROOT/$N/$K.jpg" "$HP_1991_ROOT/$N" "$HP_1991_WORK/cache/$N"
             "$SPLINE_FRAMES_WORK/hp_1991/$N/output" "$P")
done
for REL in "1990/02/IMG_0906" "1990/03 есть фотографии по страницам/IMG_1101" "1990/05/IMG_1273" "1990/09/IMG_0310" \
           "1990/10 есть фотографии по страницам/IMG_0558" "1990/11 есть фотографии по страницам/IMG_0701" \
           "1991/01 есть фотографии по страницам/IMG_0615"; do
    FRAMES+=(--frame "pack6" "$PACK6_450_ROOT/МТС/$REL.jpg" "$PACK6_450_ROOT" "$PACK6_V6_WORK/cache"
             "$PACK6_V6_WORK/output" 13.8)
done

mkdir -p "$OUT"
uv run python -m research.photo_bounds rectify-demo "${FRAMES[@]}" --out "$OUT" --jobs "$JOBS"
