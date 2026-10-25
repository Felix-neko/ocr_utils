#!/usr/bin/env bash
#
# Общий проход по паку-4 ПХ: признаки смаза (относительный кепстр плиток, research/spread_defects/motion.py) и
# корешка (шаг штрихов у сгиба, gutter.py) за одно чтение кадра.
#
# Читает: все кадры $PH_PACK4_ROOT (14 465 JPEG ~5000×3900, полное разрешение).
# Пишет: $SPREAD_DEFECTS_ROOT/{features.csv, motion/{frames.csv, tiles/…npz}, gutter/{frames.csv, profiles/…npz}}.
# Досчитывает недостающие кадры. Время: ~50–60 мин на пак при 12 воркерах (смаз отдельно шёл 45 мин, 5.3 кадр/с).
# Плитка 256 px и кольцо 3–25 px — под 450 dpi: двоение на 1985/02 IMG_0017 ≈ 5 px.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

# Ниже 16: чтение ~10 МБ JPEG с NTFS-3G упирается в диск.
JOBS="${JOBS:-8}"
EXTRA="${EXTRA:-}"

# shellcheck disable=SC2086
uv run python -m research.spread_defects features --root "$PH_PACK4_ROOT" --out-root "$SPREAD_DEFECTS_ROOT" \
    --jobs "$JOBS" $EXTRA
