#!/usr/bin/env bash
#
# Дорисовка строк, срезанных краем листа, в кадрах РИСК 1991/05 пака-6 (основа, 110 кадров) по кадрам пака-7 (донор,
# 59 кадров). Читает оба сета с Я.Диска (только чтение), кэш scan_cropping v2 и пары по тексту (research/spread_ocr);
# пишет на SSD: frames/ (все кадры основы, дорисованные — JPEG q95), sheets/ («было | донор | стало»), patches.csv.
# Время: ~20 мин на 110 кадров при 12 воркерах (SIFT, DIS-поток, TPS — CPU).
# Числа порогов и окон — в шапках research/cut_lines/{register,lines,compose}.py (масштаб ½ кадра, ~6.9 px/мм).
#
set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/cut_lines/common.sh

JOBS=12  # не 16: параллельно идут другие CPU-прогоны, а чтение с NTFS упирается в диск
OUT="$CUT_LINES_ROOT/risk_1991_05"

uv run python -m research.cut_lines pairs-from-text --compare-csv "$RISK_1991_05_TEXT_PAIRS" --out-root "$OUT"
uv run python -m research.cut_lines run \
    --base-dir "$RISK_1991_05_BASE" \
    --donor-dir "$RISK_1991_05_DONOR" \
    --v2-work "$RISK_1991_05_V2_WORK" \
    --out-root "$OUT" \
    --jobs "$JOBS"
