#!/usr/bin/env bash
#
# Фильтр псевдоглифов во втором преобразовании scan_cropping.v2 (reports/pseudo_glyphs.md): прогон v2 с выходом страниц
# на 20 кадрах — ХиП 1991 (07/IMG_0099, 0090, 0082, 0101; 02/IMG_1106 — таблица, 1110; 05/IMG_0370, 0371, 0369, 0346),
# пак-6 (МТС 1990/02, 5 кадров), ПХ 1975 (01/IMG_0051–0055) — «было» (код базы, снимок $BASE_SRC) и «стало» (текущий
# код: серии без отступов, фильтр псевдоглифов + CRAFT), листы 08_rectify обоих; замер видеопамяти — отдельный прогон
# «стало» с пустым кэшем на 4 кадрах ХиП 1991/07 (все GPU-шаги заново); nocraft — «стало» без CRAFT (время на кадр).
# CRAFT фильтра псевдоглифов — сервер на GPU в главном процессе v2 (text_blocks.craft_service).
#
# Ключи — как run_scripts/scan_cropping/run_v2_gutter_decompression.sh (v2_pack6_args.sh, --refine-ragged; ХиП —
# плотность по выпускам, тег 434 dpi, тень пальца; ПХ — тень пальца). Вход — деревья ссылок на кадры (Я.Диск — только
# чтение), кэш — копии кэшей боевых прогонов по этим кадрам (боевые кэши не трогаются); готовит их
# research/pseudo_glyphs (см. README стенда, раздел v2).
# Читает: $V/input/<набор>, $V/cache_<набор>; пишет: $V/<before|after|vram>/<набор>/{output,pages,pages_meta,debug} и
# $V/<метка>.vram.txt (видеопамять раз в 0.5 с) — всё на SSD hotstore.
# JOBS=2: машина нужна пользователю (пределы координатора); выпрямление держит 1–2 ГБ на кадр.
# Время: с готовым кэшем ~1 мин на кадр в пуле 2; прогон с пустым кэшем — ~2 мин на кадр (SAM 3, LaMa, surya).
#
#     setsid bash run_scripts/text_blocks/run_pseudo_glyphs_v2.sh [before|after|vram …] > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Корень репозитория прогона (код «стало» и относительные пути моделей finger_models/): по умолчанию — свой; при
# запуске копии скрипта из другого дерева — REPO_ROOT (python -m берёт пакет из текущей папки раньше PYTHONPATH).
cd "${REPO_ROOT:-$(dirname "$0")/../..}"
source run_scripts/scan_markup/pack1/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

V="${V:-$HOT_ROOT/pseudo_glyphs/v2_vram}"
BASE_SRC="${BASE_SRC:-$V/base_src}"
JOBS="${JOBS:-2}"
PY="${PY:-$(pwd)/.venv/bin/python}"
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(before after vram)

declare -A HP_PX_PER_MM=([07]=16.9 [02]=17.1 [05]=17.0)

# Видеопамять раз в 0.5 с в файл, пока жив процесс-сборщик; $1 — файл.
# Рядом — ОЗУ всех процессов группы скрипта (главный v2, воркеры пула, forkserver), сумма RSS в МБ, в файл <имя>.rss.
MY_PGID="$(ps -o pgid= $$ | tr -d ' ')"
vram_start() {
    ( while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        ps -eo pgid=,rss= | awk -v g="$MY_PGID" '$1 == g {s+=$2} END {print int(s/1024)}' >> "${1%.txt}.rss"
        sleep 0.5
    done ) > "$1" 2>/dev/null &
    VRAM_PID=$!
    VRAM_T0=$(date +%s)
}
vram_stop() {
    kill "$VRAM_PID" 2>/dev/null || true
    echo "видеопамять: пик $(sort -n "$1" | tail -1) МБ, фон $(head -1 "$1") МБ; ОЗУ v2: пик $(sort -n "${1%.txt}.rss" | tail -1) МБ; $(( $(date +%s) - VRAM_T0 )) с ($1)"
}

# Прогон v2: $1 — код (PYTHONPATH), $2 — вход, $3 — корень выхода, $4 — кэш, дальше — свои ключи.
run_v2() {
    local code="$1" input="$2" root="$3" cache="$4"
    shift 4
    PYTHONPATH="$code" "$PY" -m ocr_utils.scan_cropping.v2 \
        --input-dir "$input" --output "$root/output" --output-pages "$root/pages" \
        --output-pages-meta "$root/pages_meta" --cache "$cache" --debug "$root/debug" --debug-steps \
        "${V2_PACK6_ARGS[@]}" --refine-ragged "$@" --jobs "$JOBS" --log-level INFO
}

# Все наборы одним кодом: $1 — метка, $2 — код, дальше — ключи для всех наборов.
run_sets() {
    local label="$1" code="$2"
    shift 2
    local extra=("$@")
    for ISSUE in "$V"/input/hp/*/; do
        NAME="$(basename "$ISSUE")"
        run_v2 "$code" "$ISSUE" "$V/$label/hp/$NAME" "$V/cache_hp/$NAME" \
            --px-per-mm "${HP_PX_PER_MM[${NAME:0:2}]}" --force-dpi 434 --finger-zone-light-increment 0,0 --finger-shadow \
            "${extra[@]}"
    done
    run_v2 "$code" "$V/input/pack6" "$V/$label/pack6" "$V/cache_pack6" "${extra[@]}"
    run_v2 "$code" "$V/input/ph1975" "$V/$label/ph1975" "$V/cache_ph" --finger-zone-light-increment 0,0 --finger-shadow \
        "${extra[@]}"
}

for STAGE in "${STAGES[@]}"; do
    case "$STAGE" in
        before)
            vram_start "$V/before.vram.txt"; run_sets before "$BASE_SRC"; vram_stop "$V/before.vram.txt" ;;
        after)
            vram_start "$V/after.vram.txt"; run_sets after "${AFTER_SRC:-$(pwd)}"; vram_stop "$V/after.vram.txt" ;;
        nocraft)
            # Тот же код, фильтр одними правилами — для сравнения времени на кадр.
            vram_start "$V/nocraft.vram.txt"; run_sets nocraft "${AFTER_SRC:-$(pwd)}" --no-pseudo-craft
            vram_stop "$V/nocraft.vram.txt" ;;
        stress)
            # Пул пошире (JOBS=5): нет ли взаимоблокировок клиентов и сервера CRAFT при пяти воркерах; ХиП 1991/05, 07.
            vram_start "$V/stress.vram.txt"
            for NAME in "05 оплачено есть фотографии по страницам" "07 оплачено есть фотографии по страницам"; do
                run_v2 "${AFTER_SRC:-$(pwd)}" "$V/input/hp/$NAME" "$V/stress/hp/$NAME" "$V/cache_hp/$NAME" \
                    --px-per-mm "${HP_PX_PER_MM[${NAME:0:2}]}" --force-dpi 434 --finger-zone-light-increment 0,0 --finger-shadow
            done
            vram_stop "$V/stress.vram.txt" ;;
        vram)
            # Пустой кэш: все GPU-шаги заново (SAM 3, SAM 2, LaMa, surya, уточнение рваных сторон) + выход страниц.
            NAME="07 оплачено есть фотографии по страницам"
            rm -rf "$V/cache_vram"
            vram_start "$V/vram.vram.txt"
            run_v2 "${AFTER_SRC:-$(pwd)}" "$V/input/hp/$NAME" "$V/vram/hp/$NAME" "$V/cache_vram" \
                --px-per-mm 16.9 --force-dpi 434 --finger-zone-light-increment 0,0 --finger-shadow
            vram_stop "$V/vram.vram.txt" ;;
        *) echo "неизвестный этап: $STAGE (before|after|nocraft|stress|vram)" >&2; exit 2 ;;
    esac
done
