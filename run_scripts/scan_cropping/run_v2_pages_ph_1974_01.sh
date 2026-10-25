#!/usr/bin/env bash
#
# scan_cropping v2 с выходом выпрямленных страниц (--output-pages) по выпуску ПХ пак-4 1974/01 (86 кадров) — проба
# флагов кадров и нарезки. Флаги — сайдкары <кадр>.flags.json рядом с фото (навык frame-flags, 2026-10-23):
# IMG_0001, IMG_0002 — ignore-both (обложка и форзац подшивки), IMG_0003 — ignore-left (форзац), IMG_0005 —
# ignore-right / IMG_0006 — ignore-left (карточка-вклейка в разные стороны), IMG_0087 — ignore-right (обложка №2).
# Страницы: итог кадра → освещение (ScanTailor) → Кунс по подгоночным областям → 3 прохода «разбор текста → поле»
# (ядро research/page_rectify, теперь scan_cropping/v2/rectify) → разрез по кривой линии → закрас каждой страницы до
# прямоугольника тем же pull_push_seam 5 мм.
# Ключи — боевые пака-6 (v2_pack6_args.sh: 13.8 px/мм, итог TIFF 450 dpi) + осветление тени пальца, как у прогонов пака-4.
# Паспорта страниц (исходный кадр, флаги, dpi, тон, контуры закраса пальцев) и маски закраса — в pages_meta, отдельно от
# страниц. Тег dpi страниц — настоящая плотность 13.8 px/мм ≈ 351 dpi, а не --force-dpi 450 основного итога.
# Читает: Я.Диск (кадры и сайдкары, только чтение). Пишет: $OUT_ROOT/{output,pages,pages_meta,cache,debug} на SSD.
# Время: без кэша ~11 с GPU на кадр + ~25 с CPU выпрямления на кадр в пуле; выпуск ~10–15 мин.
# JOBS=12: выпрямление держит 1–2 ГБ на кадр и читает кадры с NTFS-3G (как у run_page_rectify_text.sh).
# Повторный прогон рядом с прежним: OUT_ROOT=…/01_оплачено_b CACHE=…/01_оплачено/cache.
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

JOBS="${JOBS:-12}"
ISSUE="${ISSUE:-1974/01 оплачено}"
OUT_ROOT="${OUT_ROOT:-$PH_PAGES_WORK/${ISSUE// /_}}"
# Кэш шагов можно взять у прежнего прогона (пальцы, SAM 3, surya от выхода страниц не зависят).
CACHE="${CACHE:-$OUT_ROOT/cache}"

mkdir -p "$OUT_ROOT"
uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$PACK4_PH_ROOT/$ISSUE" \
    --output "$OUT_ROOT/output" \
    --output-pages "$OUT_ROOT/pages" \
    --output-pages-meta "$OUT_ROOT/pages_meta" \
    --cache "$CACHE" \
    --debug "$OUT_ROOT/debug" \
    --debug-steps \
    "${V2_PACK6_ARGS[@]}" \
    --finger-zone-light-increment 0,0 \
    --finger-shadow \
    --jobs "$JOBS" \
    --log-level INFO
