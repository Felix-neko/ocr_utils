#!/usr/bin/env bash
#
# Партия «ФИН» 2026-10-07: журнальные развороты, декабрьские выпуски 1960–1979 (JPG 5813×4363, 434 dpi, 1 012 кадров),
# проверка на пропуски страниц и зональный смаз (research/spread_defects).
#   1. pages + sequence — номера колонтитула (tesseract) и разрывы ряда номеров по выпускам;
#   2. features + pass2 + motion-rank — плиточные признаки смаза (относительный кепстр), доля полутонов (ореол смаза)
#      и оценки кадров; правило пака-4: agree_peak ≥ 6 и mid_ratio ≥ 1.15;
#   3. motion-sheets — листы крупов самых смазанных кадров для отсмотра.
#
# Читает: $FIN_2026_10_07_ROOT/<год>/12*/IMG_*.jpg (Я.Диск, только чтение).
# Пишет: $FIN_2026_10_07_OUT/{pages/,pages_footer/,motion/,midtone/,glyphs.csv,motion/scores.csv,motion/sheets/}.
# Номер страницы: до 1964 — в верхнем колонтитуле, с 1965 — в нижнем (pages_footer/).
# Время (замер 2026-10-08, 1 012 кадров): pages ~7 кадр/с — 2.5 мин; pages footer ~4 кадр/с — 4 мин; features ~3 кадр/с — 6 мин.

set -euo pipefail
set -m
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-12}"
ROOT="$FIN_2026_10_07_ROOT"
OUT="$FIN_2026_10_07_OUT"

uv run python -m research.spread_defects pages --root "$ROOT" --out-dir "$OUT/pages" --jobs "$JOBS"
uv run python -m research.spread_defects sequence --run-dir "$OUT/pages"
uv run python -m research.spread_defects pages --root "$ROOT" --out-dir "$OUT/pages_footer" --place footer --jobs "$JOBS"
uv run python -m research.spread_defects sequence --run-dir "$OUT/pages_footer"
uv run python -m research.spread_defects features --root "$ROOT" --out-root "$OUT" --jobs "$JOBS"
uv run python -m research.spread_defects pass2 --root "$ROOT" --out-root "$OUT" --jobs "$JOBS"
uv run python -m research.spread_defects motion-rank --run-dir "$OUT/motion"
uv run python -m research.spread_defects motion-sheets --run-dir "$OUT/motion" --count 48 --out-dir "$OUT/motion/sheets"
