#!/usr/bin/env bash
#
# Паки 2–3 «Вопросов экономики»: пропуски страниц и вставки-ксерокопии (research/spread_defects).
#   1. pages + sequence — номера колонтитула (tesseract) и разрывы ряда номеров по выпускам;
#   2. copies — признаки бумаги/тонера по плиткам полос (цвет, зерно, крап, краска, неровность фона);
#   3. copies-embed — DINOv2-small по 8 вырезкам 224 px текста каждой полосы (GPU, один процесс);
#   4. copies-rank — логистическая регрессия на разметке sets/photocopy.csv, оценка каждой полосы моделью,
#      не видевшей её выпуска (5 складок по выпускам); участки: ядро ≥0.5, расширение ≥0.05;
#   5. copies-quality — читаемость полос-копий (доля слов с conf tesseract ≥80, словарь pymorphy3).
#
# Читает: все кадры $VE_PACK2_ROOT и $VE_PACK3_ROOT кроме focus-test (--only 19).
# Пишет: $VE_DEFECTS_ROOT/{pack2,pack3}/{pages/,copies/,copies.csv,copies_embed.npz},
#        $VE_DEFECTS_ROOT/result/{copies_scores.csv,copies_regions.csv,copies_quality.csv}.
# Время (замер 2026-10-04, 6 503 кадра): pages ~6 кадр/с — 20 мин; copies ~4 кадр/с — 30 мин;
#        copies-embed — 25 мин (упирается в чтение кадров с NTFS); rank — 1 мин; quality (85 полос) — 1 мин.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-14}"

for PACK in pack2 pack3; do
    if [ "$PACK" = pack2 ]; then ROOT="$VE_PACK2_ROOT"; else ROOT="$VE_PACK3_ROOT"; fi
    OUT="$VE_DEFECTS_ROOT/$PACK"
    uv run python -m research.spread_defects pages --root "$ROOT" --only 19 --out-dir "$OUT/pages" --jobs "$JOBS"
    uv run python -m research.spread_defects sequence --run-dir "$OUT/pages"
    uv run python -m research.spread_defects copies --root "$ROOT" --only 19 --out-root "$OUT" --jobs "$JOBS"
    uv run python -m research.spread_defects copies-embed --out-root "$OUT" --jobs 12
done

ROOTS=(--out-root "$VE_DEFECTS_ROOT/pack2" --out-root "$VE_DEFECTS_ROOT/pack3")
uv run python -m research.spread_defects copies-rank "${ROOTS[@]}" --threshold 0.5 --extend 0.05 \
    --result-dir "$VE_DEFECTS_ROOT/result"
uv run python -m research.spread_defects copies-quality "${ROOTS[@]}" --labeled --result-dir "$VE_DEFECTS_ROOT/result" \
    --jobs "$JOBS"
