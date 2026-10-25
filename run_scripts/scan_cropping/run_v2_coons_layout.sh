#!/usr/bin/env bash
#
# surya layout по холсту Кунса (--layout-warp coons) против холста по трапециям (--layout-warp keystone) на одном
# наборе: два прогона v2 по полным копиям кэша, затем меры стенда research/surya_coons и склейки листов «было | стало».
# Холст Кунса — первое преобразование выпрямления (scan_cropping/v2/coons_layout.py): подгоночные области страниц без
# текста (layout_frames.npz, шаг страниц) → поверхность Кунса → прямоугольники с полем 25 мм; блоки surya переносятся
# обратно на кадр кривыми многоугольниками (сгущение 2 мм, упрощение 0.1 мм).
#
# Наборы (SET=…):
#   ph_1974_01  — ПХ пак-4 1974/01 (86 кадров, 13.8 px/мм; обложка IMG_0003, карточки-вклейки IMG_0005/0006,
#                 обложка №2 IMG_0087; ключи — как run_v2_pages_ph_1974_01.sh);
#   hp_1991_01, hp_1991_02 — ХиП 1991/01 (85 кадров, 17.9 px/мм) и 02 (17.1 px/мм; сильный изгиб IMG_1049, 1052,
#                 1057, 1168), плотности — из run_v2_hp_1991.sh;
#   ph_gutter_strip — ПХ 1975 с полосой-вклейкой у корешка (23 кадра, $PH_GUTTER_STRIP_SET, IMG_0021).
#
# Кэш: ПОЛНАЯ копия (cp -r, не cp -al: запись по жёсткой ссылке переписала бы исходный кэш) без layout.json и
# summary.csv — surya считается заново в обоих режимах (лист debug/surya есть у обоих). Прогон keystone — с
# --save-fingers, и его кэш копируется в coons: пальцы второй раз не считаются.
# Первый проход — без --output-pages (дёшево); выход страниц — повторным вызовом с OUTPUT_PAGES=1 (surya и итог
# кадра берутся из кэша, лог — run_pages.log; меры прямизны строк до и после поля — в колонке pages_text сводки summary.csv).
#
# Читает: кадры с Я.Диска (только чтение), исходный кэш (только чтение, копия).
# Пишет: $OUT_ROOT/{keystone,coons}/{cache,output,debug,run.log}, меры — $SURYA_COONS_ROOT/<набор>/, склейки —
#   $SURYA_COONS_ROOT/<набор>/sheets/ (SSD hotstore).
# Время (оценка): keystone — пальцы из кэша нет (clean.png не хранился), ~10 с GPU на кадр; coons — только surya
#   (~1.1 с GPU) и финал в пуле; ПХ 1974/01 ~15 + 5 мин.
# JOBS=12: финал держит ~1 ГБ на кадр и читает кадры с NTFS-3G (как run_v2_pages_ph_1974_01.sh).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

JOBS="${JOBS:-12}"
SET="${SET:-ph_1974_01}"
OUTPUT_PAGES="${OUTPUT_PAGES:-0}"
FORCE_DPI=450
case "$SET" in
    ph_1974_01)
        INPUT="$PACK4_PH_ROOT/1974/01 оплачено"
        SRC_CACHE="$PH_PAGES_WORK/1974/01_оплачено/cache"
        RUN_ROOT="$SURYA_COONS_PH_RUNS/1974_01"
        PX=13.8
        ;;
    hp_1991_01)
        INPUT="$HP_1991_ROOT/01 оплачено"
        SRC_CACHE="$HP_1991_WORK/cache/01 оплачено"
        RUN_ROOT="$SURYA_COONS_HP_RUNS/01"
        PX=17.9
        FORCE_DPI=434
        ;;
    hp_1991_02)
        INPUT="$HP_1991_ROOT/02 оплачено"
        SRC_CACHE="$HP_1991_WORK/cache/02 оплачено"
        RUN_ROOT="$SURYA_COONS_HP_RUNS/02"
        PX=17.1
        FORCE_DPI=434
        ;;
    ph_gutter_strip)
        INPUT="$PH_GUTTER_STRIP_SET"
        SRC_CACHE="$PH_GUTTER_STRIP_WORK/cache"
        RUN_ROOT="$SURYA_COONS_PH_RUNS/gutter_strip"
        PX=13.8
        ;;
    *)
        echo "неизвестный набор: $SET" >&2
        exit 2
        ;;
esac
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT}"

# Полная копия кэша без layout.json и summary.csv (и областей surya — их пишет шаг страниц).
copy_cache() {
    local src="$1" dst="$2"
    if [[ -d "$dst" ]]; then
        echo "кэш уже есть: $dst"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"
    find "$dst" \( -name layout.json -o -name layout_frames.npz \) -delete
    rm -f "$dst/summary.csv"
}

run_mode() {
    local mode="$1"
    local root="$OUT_ROOT/$mode"
    local pages=()
    local log="$root/run.log"
    mkdir -p "$root"
    if [[ "$OUTPUT_PAGES" == 1 ]]; then
        # Второй проход — только выход страниц (surya и итог из кэша); лог отдельно, run.log первого прохода цел.
        pages=(--output-pages "$root/pages")
        log="$root/run_pages.log"
    fi
    echo "=== $SET: --layout-warp $mode → $root"
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$INPUT" \
        --output "$root/output" \
        --cache "$root/cache" \
        --debug "$root/debug" \
        --debug-steps \
        "${pages[@]}" \
        "${V2_PACK6_ARGS[@]}" \
        --px-per-mm "$PX" \
        --force-dpi "$FORCE_DPI" \
        --finger-zone-light-increment 0,0 \
        --finger-shadow \
        --save-fingers \
        --layout-warp "$mode" \
        --jobs "$JOBS" \
        --log-level INFO > "$log" 2>&1
}

copy_cache "$SRC_CACHE" "$OUT_ROOT/keystone/cache"
run_mode keystone
copy_cache "$OUT_ROOT/keystone/cache" "$OUT_ROOT/coons/cache"
run_mode coons

uv run python -m research.surya_coons measure --base "$OUT_ROOT/keystone" --new "$OUT_ROOT/coons" --label "$SET" \
    --out-dir "$SURYA_COONS_ROOT" --jobs 16
uv run python -m research.surya_coons report --out-dir "$SURYA_COONS_ROOT"
uv run python -m research.surya_coons sheets --base "$OUT_ROOT/keystone" --new "$OUT_ROOT/coons" \
    --select "$SURYA_COONS_ROOT/$SET/select.txt" --out-dir "$SURYA_COONS_ROOT/$SET/sheets"
