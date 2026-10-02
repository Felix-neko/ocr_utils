#!/usr/bin/env bash
#
# Стенд соринок у края строки (research/text_block_specks, отчёт reports/text_block_specks.md).
#
# Шаг scan: 9 943 полосы «только текст» пака-1 (по разбору v3, без поворота), страницы бинаризованного PDF
# FineReader без коррекции геометрии; детектор текстовых блоков (ink, вторая ось); по каждому концу строки
# у вертикальной стороны блока — отклонение от устойчивой (ROBUST) кривой стороны и признаки крайнего глифа.
# Выход: $OUT/scan/ends.csv (~100 концов на полосу), done.txt, errors.txt. Идемпотентно.
#
# ВРЕМЯ: ~5–8 с/полосу на воркер, 16 воркеров → ~50 мин.
#
#     setsid bash run_scripts/text_blocks/run_specks.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
OUT="${OUT:-/mnt/hotstore/scan_processing/mts/text_block_specks}"
JOBS="${JOBS:-16}"

PYTHONPATH=. uv run python -W ignore -m research.text_block_specks scan --out-dir "$OUT/scan" --jobs "$JOBS" "$@"
