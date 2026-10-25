#!/usr/bin/env bash
#
# Четвёртый проход по паку-4 ПХ: пересвет по плиткам (research/spread_defects/exposure.py) — толщина штриха
# (2·площадь/периметр маски краски на уровне 0.55 отражения бумаги), светлота и доля краски по текстовым плиткам.
#
# Читает: все кадры $PH_PACK4_ROOT (полное разрешение).
# Пишет: $SPREAD_DEFECTS_ROOT/{exposure.csv, exposure/<выпуск>/<кадр>.npz}. Досчитывает недостающие.
# Время: ~1.2 с счёта на кадр (замер 2026-10-04), ~45 мин на пак при 14 воркерах.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-14}"
EXTRA="${EXTRA:-}"

# shellcheck disable=SC2086
uv run python -m research.spread_defects pass4 --root "$PH_PACK4_ROOT" --out-root "$SPREAD_DEFECTS_ROOT" --jobs "$JOBS" $EXTRA
