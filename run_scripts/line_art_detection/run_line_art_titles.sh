#!/usr/bin/env bash
#
# Стенд research/line_art_titles: как часто line art пака-1 — стилизованный заголовок.
#
# ЧТО ДЕЛАЕТ. Единый детектор line art (page_layout) по всем 12 135 полосам пака-1 —
# заострённые JPEG, surya только из кэша, растр и таблицы из проверенной базы как известные
# исключения (как у scan_markup detect). Каждая область вырезается при 300 dpi, читается
# tesseract (rus, psm 11 и psm 6), по словам и краске считаются признаки, и по поясам доли
# краски под уверенными словами собираются контактные листы для просмотра глазами.
# Отчёт — reports/line_art_titles.md.
#
# ВРЕМЯ (замер 2026-09-26, --jobs 16): детектор ~0.1 с на полосу в пуле — минуты на пак;
# tesseract — секунды на вырезку.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/line_art_detection/run_line_art_titles.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 10; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="/mnt/system/raw/mts/pack1_line_art_titles"
JOBS=16

uv run python -m research.line_art_titles detect \
    --db "$DB_REVIEWED" --sharpened-dir "$SHARPENED_DIR" --layout-cache "$LAYOUT_CACHE_DIR" \
    --out-dir "$OUT" --jobs "$JOBS"
# Вырезки с полем 2 мм со всех сторон пишет detect (detect.CROP_PAD_MM); другое поле — recrop --side-pad-mm.
uv run python -m research.line_art_titles ocr --out-dir "$OUT" --jobs "$JOBS"
uv run python -m research.line_art_titles features --out-dir "$OUT"
uv run python -m research.line_art_titles sheets --out-dir "$OUT" --per-band 60
uv run python -m research.line_art_titles overlays --out-dir "$OUT" --jobs "$JOBS"

# DeepSeek-OCR-2 батчем через vLLM (отдельное окружение research/line_art_titles/vllm_env, ~12 мин вместе
# с пуском движка; HF-воркер deepseek_worker.py даёт тот же вывод за ~2.5 ч). Эксперты — Triton
# (--moe-backend triton по умолчанию): ядро FlashInfer CUTLASS собирается nvcc на лету и без MAX_JOBS
# съедает всю память. Запуск под сторожем памяти: systemd-run --user MemoryMax здесь не работает.
setsid uv run --project research/line_art_titles/vllm_env python research/line_art_titles/deepseek_vllm_worker.py \
    --out-dir "$OUT" --prompts markdown,ocr --prefix deepseek_vllm > "$OUT/deepseek_vllm.log" 2>&1 < /dev/null & DS=$!
uv run python scripts/memory_watchdog.py --sid "$DS" --limit-gb 90 --min-available-gb 12 --log "$OUT/deepseek_watchdog.log" &
while kill -0 "$DS" 2>/dev/null; do sleep 10; done
uv run python -m research.line_art_titles deepseek-features --out-dir "$OUT" --prefix deepseek_vllm
uv run python -m research.line_art_titles evaluate --out-dir "$OUT" --engine deepseek_vllm
uv run python -m research.line_art_titles overlays --out-dir "$OUT" --engine deepseek_vllm --jobs "$JOBS"

# Слияние с вердиктом DeepSeek и второй проход по «надписям» (слова залиты, остаток — DeepSeek и классика).
uv run python -m research.line_art_titles merge --db "$DB_REVIEWED" --out-dir "$OUT" --sharpened-dir "$SHARPENED_DIR" --jobs "$JOBS"
uv run python -m research.line_art_titles pass2-prepare --out-dir "$OUT" --jobs "$JOBS"
setsid uv run --project research/line_art_titles/vllm_env python research/line_art_titles/deepseek_vllm_worker.py \
    --out-dir "$OUT/pass2" --prompts markdown --prefix deepseek_vllm > "$OUT/pass2_vllm.log" 2>&1 < /dev/null & DS=$!
uv run python scripts/memory_watchdog.py --sid "$DS" --limit-gb 90 --min-available-gb 12 --log "$OUT/pass2_watchdog.log" &
while kill -0 "$DS" 2>/dev/null; do sleep 10; done
uv run python -m research.line_art_titles pass2-finish --out-dir "$OUT" --jobs "$JOBS"
