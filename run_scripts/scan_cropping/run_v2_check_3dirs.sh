#!/usr/bin/env bash
#
# Проверка правок scan_cropping v2 (уточнение склейки 5 мм, защита контента surya плоских страниц, трапеции и класс
# «плоско / приподнята», высветление зоны пальца) на трёх папках пака-6: МТС 1990/03 (90 кадров), МТС 1991/05 (59),
# РИСК 1991/05 (110) — 259 кадров, ~1 ч. Ключи — как у полного прогона (v2_pack6_args.sh).
# Читает: $PACK6_450_ROOT (Я.Диск, только чтение). Пишет: $SCAN_CROPPING_V2_ROOT/check_2026-10-12/{output,cache,debug},
# --save-fingers: кадр без пальцев остаётся в кэше (~5 ГБ), чтобы финал после калибровки порогов пересчитывался
# без GPU-шагов. Структура подпапок — от корня пака (как у полного прогона: оверлеи сравнимы с pack6_stopped_2026-10-12/debug).
#
set -euo pipefail

set -m
trap 'kill -- -$$' EXIT INT TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

# Папка прогона: по умолчанию первая проверка; повтор — в отдельную, `CHECK_WORK=... ./run_v2_check_3dirs.sh`.
WORK="${CHECK_WORK:-$SCAN_CROPPING_V2_ROOT/check_2026-10-12}"
JOBS=12
DIRS=(
    "МТС/1990/03 есть фотографии по страницам"
    "МТС/1991/05"
    "РИСК/1991/05 есть фотографии по страницам"
)

mkdir -p "$WORK"
for dir in "${DIRS[@]}"; do
    echo "=== $dir"
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$PACK6_450_ROOT/$dir" \
        --output "$WORK/output/$dir" \
        --cache "$WORK/cache/$dir" \
        --debug "$WORK/debug/$dir" \
        "${V2_PACK6_ARGS[@]}" \
        --save-fingers \
        --jobs "$JOBS" \
        --log-level INFO
done
