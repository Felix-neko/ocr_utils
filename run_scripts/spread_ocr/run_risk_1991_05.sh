#!/usr/bin/env bash
#
# РИСК 1991 №5: OCR левой и правой полос каждого кадра двух съёмок (tesseract rus+eng, psm 3) и пары кадров по тексту.
# Читает: кадры JPG обеих съёмок (Я.Диск, только чтение), пары по ключевым точкам research/cut_lines.
# Пишет: $OUT/pages.csv, text/<съёмка>/<кадр>_<сторона>.txt, json/ (кэш, повторный запуск не перечитывает),
#        pairs_by_text.csv (по полосам), pairs_compare.csv (по кадрам, сверка с ключевыми точками), similarity.npy.
# Время: ~3 с tesseract на полосу; 169 кадров при JOBS=12 — ~100 с, сопоставление — секунды.
# Ещё пишет donor_sequence.csv — нумерация кадров донора по порядку (левая = 2·k + сдвиг, сдвиг — мода прочитанных номеров).
# JOBS=12, а не 16: параллельно идёт другая тяжёлая CPU-задача.

set -euo pipefail
set -m
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$(dirname "$0")/../.."
source run_scripts/spread_ocr/common.sh

JOBS="${JOBS:-12}"
OUT="$SPREAD_OCR_ROOT/risk_1991_05"
mkdir -p "$OUT"

uv run python -m research.spread_ocr ocr \
    --set "lena=$RISK_1991_05_LENA" \
    --set "borya=$RISK_1991_05_BORYA" \
    --out "$OUT" --jobs "$JOBS"

uv run python -m research.spread_ocr match \
    --out "$OUT" --base lena --donor borya \
    --akaze-pairs "$RISK_1991_05_AKAZE_PAIRS"
