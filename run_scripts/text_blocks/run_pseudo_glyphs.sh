#!/usr/bin/env bash
#
# Стенд «псевдоглифы» (research/pseudo_glyphs, reports/pseudo_glyphs.md): компоненты краски, которые разбор текстовых
# блоков принимает за буквы, — тень сгиба и кромка листа у корешка, полоса обложки, нитки прошивки, карандаш, сор.
#
# Что делает: features — разбор вырезок холста Кунса ХиП 1991 как во втором преобразовании v2 (серый → 300 dpi →
#   утолщение → гашение линеек → analyse_gray), кандидаты (крайние компоненты рядов + компоненты у края вырезки) с
#   признаками; maps — карты CRAFT и pero ParseNet (GPU, по одному процессу, строго по очереди); scores — оценки
#   детекторов у кандидатов; evaluate — полнота и вред правил по разметке research/pseudo_glyphs/sets/labels.csv,
#   с CRAFT и без; compare — разбор «было (без фильтра, с правилом 0.3 мм и без) | стало (правила, правила + CRAFT)»
#   по всем вырезкам, оверлеи страниц с закрасом; pack1-features — регрессия на размеченных наборах пака-1
#   (edge_marks, text_block_specks).
# Читает: вырезки $CANVAS (PNG + JSON, line_art_boxes/hp_1991/canvas), бинаризованные nogeo PDF пака-1.
# Пишет: $OUT/{candidates.csv,scores.csv,features/,measures/,gray300/,maps/,eval/,compare/,sheets/} — SSD, вне git.
# Сколько идёт (2026-10-25, 2 воркера): features ~25 мин (1 096 страниц, 2.6 с на разбор), maps ~11 мин (CRAFT 0.27 с,
#   pero 0.34 с на страницу), scores ~1 мин, compare ~2 ч (до трёх разборов на страницу и режим), pack1-features ~10 мин.
# Пороги — research/pseudo_glyphs/rules.py (RuleParams), откуда — reports/pseudo_glyphs.md. CRAFT_DOUBT=0.4 — порог
#   CRAFT для спорных серых кандидатов (сетка 0.1–0.6: вред постоянен до 0.55).
# JOBS=2: машина нужна пользователю, рядом идут другие задачи; разбор держит ~0.45 ГБ на воркер.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_pseudo_glyphs.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
source "$(dirname "$0")/../scan_markup/pack1/common.sh"
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CANVAS="${CANVAS:-$HOT_ROOT/line_art_boxes/hp_1991/canvas}"
OUT="${OUT:-$HOT_ROOT/pseudo_glyphs/hp1991}"
JOBS="${JOBS:-2}"
CRAFT_DOUBT="${CRAFT_DOUBT:-0.4}"
export PYTHONPATH=.

uv run python -m research.pseudo_glyphs features --canvas-root "$CANVAS" --out-dir "$OUT" --jobs "$JOBS"
uv run python -m research.pseudo_glyphs maps --out-dir "$OUT"
uv run python -m research.pseudo_glyphs scores --out-dir "$OUT" --jobs "$JOBS"
uv run python -m research.pseudo_glyphs evaluate --out-dir "$OUT"
uv run python -m research.pseudo_glyphs compare --canvas-root "$CANVAS" --out-dir "$OUT/compare" \
    --craft-dir "$OUT/maps/craft" --craft-doubt "$CRAFT_DOUBT" --jobs "$JOBS"
uv run python -m research.pseudo_glyphs pack1-features --out-dir "$HOT_ROOT/pseudo_glyphs/pack1_sets" --jobs "$JOBS"
uv run python -m research.pseudo_glyphs pack1-check --out-dir "$HOT_ROOT/pseudo_glyphs/pack1_sets"
