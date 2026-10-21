#!/usr/bin/env bash
#
# Стенд research/page_cutting, сквозная нарезка: рамка spline со сгибом и краем у корешка по цвету бумаги →
# выпрямление (гомография + Coons-патч) → TIFF страницы с dpi кадра → scantailor-cli (поля 15/10/5/5 мм, серый вывод,
# Equalize Illumination; цветные страницы — без полей, без Equalize, без dewarp) дважды: dewarp выкл / авто → судья
# (число слов и уверенность tesseract, прямизна строк page_layout.text_blocks, сход кривых модели ScanTailor).
# Наборы: research/page_cutting/sets/{tight_gutter,curved,ragged,flat,colour}.txt (~44 кадра, ~88 страниц).
# Прогоны и px/мм — как в run_page_cutting_select.sh. scantailor-cli — ветка cli в ~/Projects/scantailor-advanced.
# Читает: кэш и output прогонов v2 (SSD). Пишет: $PAGE_CUTTING_ROOT/cut/<RUN_NAME>/{pages,st,sheets,profiles,cut.csv}.
# Время: ~25 с на кадр на воркер (ScanTailor ×2 ~6 с, tesseract ×2, разбор строк ×2), ~5 мин на 6 воркерах.
# JOBS=6: полноразмерные страницы в remap, ScanTailor и разборе строк — до ~3 ГБ ОЗУ на воркер, плюс tesseract.
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

JOBS="${JOBS:-6}"
OUT="$PAGE_CUTTING_ROOT/cut/${RUN_NAME:-v1}"
SETS_DIR=research/page_cutting/sets
RUNS=(
    --run "pack6=$PACK6_V6_WORK,,13.8"
    --run "ph4=$PACK4_PH_WORK,1975,13.8"
    --run "hp91=$HP_1991_02_WORK,,17.1"
)

uv run python -m research.page_cutting cut "${RUNS[@]}" --out-dir "$OUT" --cli "$SCANTAILOR_CLI" --jobs "$JOBS" \
    --set "$SETS_DIR/tight_gutter.txt" --set "$SETS_DIR/curved.txt" --set "$SETS_DIR/ragged.txt" \
    --set "$SETS_DIR/flat.txt" --set "$SETS_DIR/colour.txt"
