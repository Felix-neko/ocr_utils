#!/usr/bin/env bash
#
# scan_cropping v2 целиком с выходом выпрямленных страниц и растяжением у корешка (--gutter-decompression, по умолчанию
# включено: модель по ширине букв, страница шире со стороны корешка) по трём наборам: «Хозяйство и право» 1991
# (12 выпусков, 966 кадров), пак-6 (МТС и РИСК, ~2500 кадров), «Плановое хозяйство» 1975 (пак-4, 1027 кадров) —
# проверка растяжения и ложных срабатываний на паке, где сжатия у корешка почти нет. Полные отладочные картинки
# (--debug-steps: шаги сегментации 01–07, выход страниц 08_rectify, 09_pages, растяжение «было | стало» 10_gutter).
# --refine-ragged — уточнение рваных внешних сторон страниц (детектор извилистости сырой маски SAM 3, порог по умолчанию
# 1.05; с 2026-10-25 включено по умолчанию, ключ оставлен явным): ХиП 1991/01 IMG_1017 L — перескок кромки в «веере» страниц, извилистость 1.22.
# Ключи — боевые: v2_pack6_args.sh; ХиП 1991 — как run_v2_hp_1991.sh (плотность по выпускам, тег 434 dpi, тень пальца);
# ПХ 1975 — как run_v2_ph_pack4_1975.sh (тень пальца). Флаги кадров — сайдкары рядом с фото (по умолчанию читаются).
#
# Области страниц без складки карты у корешка (2026-10-25, rectify/pages.unfolded): полоса, где поле снова брало концы
# строк левой страницы (ХиП 1991/01 IMG_0982 L, «кафедрой» → лишняя «й» за корешком), закрашивается. Выход — в
# $GUTTER_DECOMPRESSION_FOLD_ROOT; прежний прогон без этой правки — в $GUTTER_DECOMPRESSION_ROOT (для сравнения).
#
# С 2026-10-26 выход — в $GUTTER_DECOMPRESSION_CRAFT_ROOT (прогон *_fold остановлен на 540 кадрах и оставлен для
# сравнения): фильтр псевдоглифов разбора страниц (кромка листа, тень сгиба, нитки, сор; --pseudo-glyphs) с CRAFT на GPU
# в главном процессе (--pseudo-craft; замер на 20 кадрах — пик видеопамяти 12.1 ГБ из 16 вместе с фоном 2.3 ГБ),
# серии строк без абзацных отступов (text_blocks SKIP_INDENT), line art с добором оторванных штрихов (LINE_ART 9) и
# рамки областей, перенесённые за полем по тексту. Кэш ХиП 1991 — копия кэша прогона *_fold (GPU-шаги не повторяются).
#
# С 2026-10-26 (вечер) выход — в $GUTTER_DECOMPRESSION_TABLES_ROOT (прогон *_craft остановлен на 648 кадрах ХиП 1991):
# плюс таблицы с изогнутыми после Кунса линейками (признаки таблицы по выпрямленной вырезке, ХиП 1991/02 IMG_1161 L) и
# деление текстовых блоков по смене стороны выключки (подпись под абзацем, «г. Город» — свои блоки, 01/IMG_1008 R).
#
# С 2026-10-26 (ночь) выход — в $GUTTER_DECOMPRESSION_TAILS_ROOT (прогон *_tables остановлен на 286 кадрах ХиП 1991):
# короткий хвост строки без замеренных глифов («143 с.—») сливается со своей строкой по жирности — левая сторона списка
# литературы выровнена (01/IMG_1030 R). Кэш ХиП 1991 — копия кэша прогона *_tables.
# Разбор итога страницы — без чужого у корешка (rectify/fold.own_view: полоса повтора концов строк и соседняя страница
# закрашиваются способом --fill; ХиП 1991/01 IMG_0982 R — оси больше не пришивают повторённые «й», «В,»), лист
# 08_rectify — итог по записанным страницам с зоной докраса.
#
# Кэш шагов (пальцы, SAM 3, surya) — копия кэша боевого прогона набора (cp -a при первом запуске): v2 пишет в кэш
# summary.csv и досчитанные шаги, боевой кэш не трогается. ХиП 1991 — копия кэша прежнего прогона, если он есть: там
# уже посчитаны уточнения рваных сторон (GPU).
# Читает: Я.Диск (кадры и сайдкары, только чтение), кэши $HP_1991_WORK, $PACK6_450_WORK, $PACK4_PH_WORK.
# Пишет: $GUTTER_DECOMPRESSION_TAILS_ROOT/<набор>/{output,pages,pages_meta,cache,debug} на SSD hotstore.
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

OUT="$GUTTER_DECOMPRESSION_TAILS_ROOT"
JOBS="${JOBS:-9}"  # 9 — 75 % ядер по просьбе пользователя (2026-10-26), параллельно идут стенды; как у боевых прогонов: выпрямление держит 1–2 ГБ на кадр и читает кадры с NTFS-3G
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
        --refine-ragged \
        "$@" \
        --jobs "$JOBS" \
        --log-level INFO
}

for SET in "${SETS[@]}"; do
    ROOT="$OUT/$SET"
    mkdir -p "$ROOT"
    case "$SET" in
        hp_1991)
            PREV="$GUTTER_DECOMPRESSION_TABLES_ROOT/$SET/cache"
            if [[ -d "$PREV" ]]; then copy_cache "$PREV" "$ROOT/cache"; else copy_cache "$HP_1991_WORK/cache" "$ROOT/cache"; fi
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
