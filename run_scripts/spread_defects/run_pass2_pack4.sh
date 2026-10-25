#!/usr/bin/env bash
#
# Второй проход по паку-4 ПХ: буквы у сгиба (сжатие у корешка, мера по глифам — research/spread_defects/gutter.py,
# glyph_edge) и номера страниц из колонтитула (tesseract, sequence.py) за одно чтение кадра.
#
# Читает: все кадры $PH_PACK4_ROOT (полное разрешение; номера — по уменьшенной вдвое копии).
# Пишет: $SPREAD_DEFECTS_ROOT/{glyphs.csv, pages/{frames.csv, tokens/…json}, midtone/…npz (доля полутонов по плиткам)}. Досчитывает недостающие кадры.
# Время: ~5.5 кадр/с при 14 воркерах, ~45 мин на пак (замер 2026-10-02).

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

# Ниже 16: диск NTFS-3G и однопоточный tesseract на кадр.
JOBS="${JOBS:-12}"
EXTRA="${EXTRA:-}"

# shellcheck disable=SC2086
uv run python -m research.spread_defects pass2 --root "$PH_PACK4_ROOT" --out-root "$SPREAD_DEFECTS_ROOT" --jobs "$JOBS" $EXTRA
