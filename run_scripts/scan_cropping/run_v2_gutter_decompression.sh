#!/usr/bin/env bash
#
# scan_cropping v2 целиком с выходом выпрямленных страниц и растяжением у корешка (--gutter-decompression, по умолчанию
# включено: модель по ширине букв, страница шире со стороны корешка) по трём наборам: «Хозяйство и право» 1991
# (12 выпусков, 966 кадров), пак-6 (МТС и РИСК, ~2500 кадров), «Плановое хозяйство» 1975 (пак-4, 1027 кадров) —
# проверка растяжения и ложных срабатываний на паке, где сжатия у корешка почти нет. Полные отладочные картинки
# (--debug-steps: шаги сегментации 01–07, выход страниц 08_rectify, 09_pages, растяжение «было | стало» 10_gutter).
# Ключи — боевые: v2_pack6_args.sh; ХиП 1991 — как run_v2_hp_1991.sh (плотность по выпускам, тег 434 dpi, тень пальца);
# ПХ 1975 — как run_v2_ph_pack4_1975.sh (тень пальца). Флаги кадров — сайдкары рядом с фото (по умолчанию читаются).
#
# Кэш шагов (пальцы, SAM 3, surya) — копия кэша боевого прогона набора (cp -a при первом запуске): v2 пишет в кэш
# summary.csv и досчитанные шаги, боевой кэш не трогается.
# Читает: Я.Диск (кадры и сайдкары, только чтение), кэши $HP_1991_WORK, $PACK6_450_WORK, $PACK4_PH_WORK.
# Пишет: $GUTTER_DECOMPRESSION_ROOT/<набор>/{output,pages,pages_meta,cache,debug} на SSD hotstore.
# Место: итог TIFF ~70 МБ на кадр плюс страницы — ~600 ГБ на три набора; кэши — 52 ГБ.
# Время: с готовым кэшем — CPU выпрямления ~25 с на кадр в пуле 12; ~3–5 ч на три набора.
# Запуск: run_v2_gutter_decompression.sh [hp_1991|pack6|ph_1975 …] (без аргументов — все три по очереди).
# Перезапуск безопасен: готовые итоги пропускаются (--skip-if-exists).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

OUT="$GUTTER_DECOMPRESSION_ROOT"
JOBS="${JOBS:-12}"  # как у боевых прогонов: выпрямление держит 1–2 ГБ на кадр и читает кадры с NTFS-3G
SETS=("$@")
[[ ${#SETS[@]} -gt 0 ]] || SETS=(hp_1991 pack6 ph_1975)

# Плотность бумаги ХиП 1991 по выпускам, px/мм (замер — шапка run_v2_hp_1991.sh).
declare -A HP_PX_PER_MM=(
    [01]=17.9 [02]=17.1 [03]=17.3 [04]=17.2 [05]=17.0 [06]=17.3
    [07]=16.9 [08]=16.6 [09]=16.6 [10]=16.6 [11]=16.3 [12]=16.3
)

# Копия кэша набора: $1 — боевой кэш, $2 — кэш прогона.
copy_cache() {
    if [[ ! -d "$2" ]]; then
        echo "копия кэша: $1 → $2"
        mkdir -p "$(dirname "$2")"
        cp -a "$1" "$2"
    fi
}

# Прогон v2 по папке: $1 — вход, $2 — корень выхода набора, $3 — подпапка (выпуск или пусто), дальше — свои ключи.
run_v2() {
    local input="$1" root="$2" sub="$3"
    shift 3
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$input" \
        --output "$root/output$sub" \
        --output-pages "$root/pages$sub" \
        --output-pages-meta "$root/pages_meta$sub" \
        --cache "$root/cache$sub" \
        --debug "$root/debug$sub" \
        --debug-steps \
        "${V2_PACK6_ARGS[@]}" \
        "$@" \
        --jobs "$JOBS" \
        --log-level INFO
}

for SET in "${SETS[@]}"; do
    ROOT="$OUT/$SET"
    mkdir -p "$ROOT"
    case "$SET" in
        hp_1991)
            copy_cache "$HP_1991_WORK/cache" "$ROOT/cache"
            for DIR in "$HP_1991_ROOT"/*/; do
                NAME="$(basename "$DIR")"
                PX="${HP_PX_PER_MM[${NAME:0:2}]}"
                echo "=== $SET/$NAME ($PX px/мм) → $ROOT"
                run_v2 "$DIR" "$ROOT" "/$NAME" --px-per-mm "$PX" --force-dpi 434 \
                    --finger-zone-light-increment 0,0 --finger-shadow
            done ;;
        pack6)
            copy_cache "$PACK6_450_WORK/cache" "$ROOT/cache"
            echo "=== $SET → $ROOT"
            run_v2 "$PACK6_450_ROOT" "$ROOT" "" ;;
        ph_1975)
            copy_cache "$PACK4_PH_WORK/cache/1975" "$ROOT/cache/1975"
            echo "=== $SET → $ROOT"
            run_v2 "$PACK4_PH_ROOT/1975" "$ROOT" "/1975" --finger-zone-light-increment 0,0 --finger-shadow ;;
        *)
            echo "неизвестный набор: $SET (hp_1991|pack6|ph_1975)" >&2
            exit 2 ;;
    esac
done
