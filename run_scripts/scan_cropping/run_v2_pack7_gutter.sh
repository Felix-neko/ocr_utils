#!/usr/bin/env bash
#
# scan_cropping v2 с выходом страниц по трём выпускам пака-7 (МТС 1991/05, 1991/06, 1990/07; ~180 кадров) — материал
# стенда research/gutter_cut («линия разреза страниц у корешка впритык к тексту одной страницы»). Пак-7 через v2 раньше
# не прогонялся: кэша нет, все GPU-шаги (пальцы, SAM 3, surya) считаются заново.
#
# Числа:
#   --px-per-mm 12.2 — замер 2026-10-26 по шагу строк (автокорреляция профиля краски): пак-7 МТС 1991 — 48 и 59 px
#       (два кегля), 1990/07 — 48 px; пак-6 МТС тех же лет — 54 и 67 px при 13.8 px/мм. Отношение 0.88 → 12.2 px/мм.
#       1974/01 (другой набор) и 1991/04 (фото по страницам) сюда не взяты.
#   Остальные ключи — боевые v2_pack6_args.sh (тот же журнал МТС той же поры), --force-dpi 450.
# Читает: Я.Диск пак-7 (только чтение). Пишет: $GUTTER_CUT_ROOT/pack7/{output,pages,pages_meta,cache,debug} на SSD.
# Время: ~1–2 мин на кадр с GPU-шагами; ~3 ч. Перезапуск безопасен (--skip-if-exists).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

PACK7_MTS_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-7 от Бори/МТС"
ROOT="$GUTTER_CUT_ROOT/pack7"
JOBS="${JOBS:-9}"
ISSUES=("$@")
[[ ${#ISSUES[@]} -gt 0 ]] || ISSUES=(1991/05 1991/06 1990/07)

for ISSUE in "${ISSUES[@]}"; do
    SUB="/${ISSUE//\//__}"
    echo "=== $ISSUE → $ROOT (12.2 px/мм)"
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$PACK7_MTS_ROOT/$ISSUE" \
        --output "$ROOT/output$SUB" \
        --output-pages "$ROOT/pages$SUB" \
        --output-pages-meta "$ROOT/pages_meta$SUB" \
        --cache "$ROOT/cache$SUB" \
        --debug "$ROOT/debug$SUB" \
        --debug-steps \
        "${V2_PACK6_ARGS[@]}" \
        --px-per-mm 12.2 \
        --refine-ragged \
        --jobs "$JOBS" \
        --log-level INFO
done
