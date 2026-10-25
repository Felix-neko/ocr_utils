#!/usr/bin/env bash
#
# «Хозяйство и право» (пробный пак), 1977–1980: пропуски и смаз — 48 выпусков, ~2 900 кадров JPG 434 dpi
# (папки «NN оплачено…»; в 1977–1978 есть выпуски с отдельными фото по страницам).
#   1. pages (верхний и нижний колонтитул) + sequence — номера страниц и разрывы ряда по выпускам;
#   2. features + pass2 — плиточные сетки кепстра, полутонов, сгиб (glyphs.csv);
#   3. motion-rank — оценка по пику кепстра (правило пака-4, для сверки);
#   4. motion-zones + motion-zone-sheets — карта смаза по зонам всего кадра (ореол к медиане выпуска ≥ 1.6, двоение
#      по кепстру), отсев растра и цветных плашек; лист на кадр с зонами — отсматривать эти листы.
#
# Читает: $HP_1991_PACK_ROOT/{1977,1978,1979,1980}/<выпуск>/IMG_*.jpg (Я.Диск, только чтение).
# Пишет: $HP_1977_1980_OUT/{pages/,pages_footer/,features.csv,glyphs.csv,motion/,midtone/,gutter/,motion/zone_sheets/}.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"
cd "$(dirname "$0")/../.."

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-12}"
ROOT="$HP_1991_PACK_ROOT"
OUT="$HP_1977_1980_OUT"
SD=(uv run python -m research.spread_defects)
ONLY=(--only 1977 --only 1978 --only 1979 --only 1980)

"${SD[@]}" pages --root "$ROOT" "${ONLY[@]}" --out-dir "$OUT/pages" --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages"
"${SD[@]}" pages --root "$ROOT" "${ONLY[@]}" --out-dir "$OUT/pages_footer" --place footer --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages_footer"

"${SD[@]}" features --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass2 --root "$ROOT" "${ONLY[@]}" --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-rank --run-dir "$OUT/motion"
"${SD[@]}" motion-zones --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-zone-sheets --out-root "$OUT" --jobs "$JOBS"
