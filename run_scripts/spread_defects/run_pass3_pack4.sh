#!/usr/bin/env bash
#
# Третий проход по паку-4 ПХ: строки у сгиба (research/spread_defects/lines.py) и признаки концевых букв строки
# против опорных букв той же строки (glyph_quality.py): сжатие, глубина ядра, доля ядра штриха (порча черт),
# куски на половине глубины, резкость (градиент, лапласиан, Tenengrad, DOM), полутона; положение концов строк
# (буквы под переплётом) и глубина опорных букв (пересвет).
#
# Читает: все кадры $PH_PACK4_ROOT (полное разрешение).
# Пишет: $SPREAD_DEFECTS_ROOT/{pass3.csv, lines/<выпуск>/<кадр>.npz} — таблица строк кадра. Досчитывает недостающие.
# Время: ~2 с на кадр в воркере, ~1 ч на пак при 14 воркерах. v2: + сжатие по шагу штрихов (pitch). v3 (2026-10-04): концевые знаки той же высоты, что опорные (без заглавных, инициалов, цифр). v4: группа у сгиба за широким пробелом (номер, обрывок за сгибом) пропускается.

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null || true' EXIT INT TERM
source "$(dirname "$0")/common.sh"

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-14}"
EXTRA="${EXTRA:-}"

# shellcheck disable=SC2086
uv run python -m research.spread_defects pass3 --root "$PH_PACK4_ROOT" --out-root "$SPREAD_DEFECTS_ROOT" --jobs "$JOBS" $EXTRA
