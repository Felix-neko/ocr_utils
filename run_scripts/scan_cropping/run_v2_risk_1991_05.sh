#!/usr/bin/env bash
#
# Прогон scan_cropping v2 на подшивке РИСК 1991/05 (пак-6, 110 кадров разворотов на чёрном коврике, пальцы почти
# на каждом кадре, у многих страниц край текста подходит к обрезу):
# пальцы (LaMa inkback) → фон → SAM 3 1288 px → snap_bg → склейка hull_bridge → уточнение у края (~1.5 см)
# → copy back surya в книге → закрас до осевой рамки бумаги. Два способа закраса — telea и replicate (v1) — по
# общему кэшу шагов: второй прогон считает только финал.
# Читает кадры с Я.Диска (только чтение), пишет на SSD: $OUT_ROOT/{telea,replicate,work,debug_*}.
# Время: ~10–15 с на кадр (GPU: пальцы, surya, SAM 3), ~25–30 мин на 110 кадров.
#
# Числа:
#   PX_PER_MM=13.8 — калибровка v2 (подшивки МТО 1990, ~6.9 px/мм на ½). Замер 2026-10-11: высота книги в кадре
#                    РИСК 1991/05 — медиана ~3570 px (10 кадров), МТО 1990 — ~3600 px (10 кадров): та же съёмка и
#                    тот же формат, отношение 0.99 — оставлено калибровочное значение.
#   LAMA_HOLE_MAX_PX=300 — предел рамки дыры в масштабе LaMa (research/lama_params: разметка CVAT — 300).
#
set -euo pipefail

# Пул процессов: при остановке скрипта гасим всю группу (forkserver и воркеры), иначе остаются сироты.
set -m
trap 'kill -- -$$' EXIT INT TERM

# Скрипт лежит в run_scripts/<подсистема>/, пути внутри отсчитываются от корня репозитория.
cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh

INPUT_DIR="$PACK6_ROOT/РИСК/1991/05 есть фотографии по страницам"
OUT_ROOT="$SCAN_CROPPING_V2_ROOT/risk_1991_05"
PX_PER_MM=13.8
# Размеры ниже — в мм бумаги, пересчитаны из прежних px при 13.8 px/мм: σ заливки 32 px = 2.318841 мм,
# припуск блоков surya 18,72 px = 1.3043,5.2174 мм.
LAMA_HOLE_MAX_PX=300
JOBS=16

for FILL in telea replicate; do
    echo "scan_cropping v2: закрас до прямоугольника $FILL → $OUT_ROOT/$FILL"
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$INPUT_DIR" \
        --output "$OUT_ROOT/$FILL" \
        --cache "$OUT_ROOT/work" \
        --debug "$OUT_ROOT/debug_$FILL" \
        --px-per-mm "$PX_PER_MM" \
        --lama-hole-max-px "$LAMA_HOLE_MAX_PX" \
        --fill "$FILL" \
        --crop-fill-blur-mm 2.318841 \
        --crop-fill-fade 0.0 \
        --layout-pad-mm 1.3043,5.2174 \
        --jobs "$JOBS" \
        --log-level INFO
done
