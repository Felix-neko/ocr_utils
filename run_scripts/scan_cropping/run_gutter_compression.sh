#!/usr/bin/env bash
#
# Стенд research/gutter_compression «сжатие букв у корешка» по ХиП 1991 (фото-развороты, туго сшитая подшивка).
# Шаг — первый аргумент:
#   scan     — мера по всем выпрямленным итогам v4 ($TEXT_RECTIFIED_V4_WORK): раскладка холста по кэшу v2, строки по
#              краске (после поля горизонтальны), окна 5 мм вдоль строки от края текста у корешка (шаг 1 мм до 25 мм,
#              дальше 2.5 мм), пять мер окна к опоре своей строки (окна дальше 25 мм от края). ~15 мин, 966 кадров.
#   select   — пояса сжатия по сторонам (sides.csv) и листы вырезок у корешка на пояс. ~30 с.
#   synth    — проверка мер: нулевая у внешнего края текста и восстановление синтетического сжатия
#              c(d) = 1 − A·exp(−(d − e)/L), A 0–0.45, L 5 и 12 мм, на 40 кадрах, у которых обе стороны без сжатия. ~5 мин.
#   outer    — поправка на место в строке: те же меры у ВНЕШНИХ краёв текста всех кадров (начала строк левой страницы,
#              концы правой; корешка там нет) по поясам расстояния от края → position_bias.csv. ~15 мин.
#   letters  — сверка со словами tesseract (ширина слова к ожидаемой по буквам), по 8 сторон на пояс. ~5 мин.
#   stretch  — выпрямление в два преобразования (освещение, Кунс, 3 прохода поля, как run_page_rectify_text.sh, но с
#              настоящей плотностью бумаги) плюс растяжение у корешка по модели; меры до/после и листы «было | стало»
#              по множествам research/gutter_compression/sets/*.txt. ~40 с на кадр (разбор строк 4 раза + 1 повторный).
#   shape    — форма зависимости амплитуды сжатия от высоты страницы B(y) по окнам scan (только CPU, без картинок):
#              константа, линия, полиномы Лежандра 2–3, P-сплайн, логистическая ступень; перекрёстная проверка по
#              группам строк, листы кривых. ~2 мин, 1853 стороны, 16 воркеров.
#   edge-profile — профиль сжатия у края текста с разрешением в букву: ширина каждой буквы к опоре своей строки и узкие
#              окна 2 мм с шагом 0.5 мм по всем выпрямленным итогам v4; то же от внешнего края — ноль. ~15 мин.
# Плотность бумаги ХиП 1991 — по выпускам (run_v2_hp_1991.sh); выпуски 11–12 без итогов spline_frames — пропускаются.
# Числа модели и пределов растяжения — reports/gutter_compression.md.
# Читает: Я.Диск (оригиналы, только чтение — области страниц по кэшу), кэш и итоги v2, итоги v4 на SSD.
# Пишет: $GUTTER_COMPRESSION_ROOT/{scan,select,synth,outer,letters,stretch*}.
# JOBS=8: кадр читает оригинал с Я.Диска (NTFS-3G) и держит 1–2 ГБ картинок и карт; машина делит ядра с прогоном v2.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

STEP="${1:?шаг: scan|select|synth|outer|letters|stretch|shape|edge-profile}"
JOBS="${JOBS:-8}"
ROOT="$GUTTER_COMPRESSION_ROOT"
SETS_DIR=research/gutter_compression/sets

declare -A HP_PX_PER_MM=(
    [01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2 [05]=17.0 [06]=17.3
    [07]=16.9 [08]=16.6 [09]=16.6 [10]=16.6 [11]=16.3 [12]=16.3
)
SETS=()
for DIR in "$HP_1991_ROOT"/*/; do
    NAME="$(basename "$DIR")"
    RECT="$TEXT_RECTIFIED_V4_WORK/output/hp_1991/$NAME"
    V2="$SPLINE_FRAMES_WORK/hp_1991/$NAME/output"
    [[ -d "$RECT" && -d "$V2" ]] || { echo "пропуск $NAME: нет итогов v2 или v4"; continue; }
    SETS+=(--set "hp_1991/$NAME" "$DIR" "$HP_1991_WORK/cache/$NAME" "$V2" "$RECT" "${HP_PX_PER_MM[${NAME:0:2}]}")
done

mkdir -p "$ROOT/$STEP"
echo "=== gutter_compression $STEP → $ROOT/$STEP"
case "$STEP" in
    scan)
        uv run python -m research.gutter_compression scan "${SETS[@]}" --out-root "$ROOT/scan" --jobs "$JOBS" ;;
    select)
        uv run python -m research.gutter_compression select --scan-root "$ROOT/scan" \
            --rectified-root "$TEXT_RECTIFIED_V4_WORK/output" --out-dir "$ROOT/select" ;;
    synth)
        uv run python -m research.gutter_compression synth "${SETS[@]}" --sides-csv "$ROOT/select/sides.csv" \
            --out-dir "$ROOT/synth" --jobs "$JOBS" ;;
    outer)
        uv run python -m research.gutter_compression outer "${SETS[@]}" --out-dir "$ROOT/outer" --jobs "$JOBS" ;;
    letters)
        uv run python -m research.gutter_compression letters "${SETS[@]}" --sides-csv "$ROOT/select/sides.csv" \
            --out-dir "$ROOT/letters" --jobs "$JOBS" ;;
    stretch)
        uv run python -m research.gutter_compression stretch-run "${SETS[@]}" \
            --keys "$SETS_DIR/compressed.txt" --keys "$SETS_DIR/control.txt" \
            --out-dir "$ROOT/stretch${VARIANT:+_$VARIANT}" ${STRETCH_ARGS:-} --jobs "$JOBS" ;;
    shape)
        uv run python -m research.gutter_compression shape --scan-root "$ROOT/scan" \
            --position-bias "$ROOT/outer/position_bias.csv" \
            --keys "$SETS_DIR/compressed.txt" --out-dir "$ROOT/shape" --jobs "${SHAPE_JOBS:-16}" ;;
    edge-profile)
        uv run python -m research.gutter_compression edge-profile "${SETS[@]}" --out-root "$ROOT/edge-profile" \
            --jobs "$JOBS" ;;
    *)
        echo "неизвестный шаг: $STEP"; exit 2 ;;
esac
