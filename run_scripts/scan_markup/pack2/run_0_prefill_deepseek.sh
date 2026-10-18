#!/usr/bin/env bash
#
# Шаг 0: кэш разбора полос DeepSeek-OCR-2 — затравки line art для шага 1 (--deepseek-cache).
#
# Зачем: surya пропускает тонкие графики (1977/04 0180_2R, «Рис. 4» — блок Caption 0.50), а DeepSeek
# в режиме grounding видит их как image. Модель живёт в своём окружении vLLM
# (ocr_utils/page_layout/line_art/deepseek/vllm_env), поэтому её ответы кладутся на диск заранее.
#
# Читает: оригиналы пака ($PACK_DIR, 597 полос, по разу — только заголовки у уже размеченных).
# Пишет: $DEEPSEEK_CACHE_DIR/scan/<год>/<выпуск>/<полоса>.json (формат кэша surya), рабочую папку
# $DEEPSEEK_WORK_DIR (кадры surya 150 dpi в PNG, задания и ответы воркера; прерванный прогон
# продолжается, после набивки папку можно удалить).
#
# Время: подготовка кадров упирается в чтение TIFF с /mnt/dump3 (минуты), модель — пуск движка
# ~2,5 мин и ~1 с на полосу пакетом. Нужен GPU (~75 % видеопамяти — кэш vLLM): ничего другого на
# GPU в это время не гонять. Воркер vLLM идёт под сторожем памяти (scripts/memory_watchdog.py).
#
# JOBS=8, а не 16: подготовка кадров упирается в медленный NTFS-3G, а не в CPU.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM
source "$(dirname "$0")/common.sh"
JOBS="${JOBS:-8}"

uv run python -m ocr_utils.page_layout prefill-deepseek \
    --cache "$DEEPSEEK_CACHE_DIR" \
    --variant scan \
    --scan-dir "$PACK_DIR" \
    --work-dir "$DEEPSEEK_WORK_DIR" \
    --jobs "$JOBS" \
    "$@"
