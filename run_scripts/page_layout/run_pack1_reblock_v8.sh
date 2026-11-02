#!/usr/bin/env bash
#
# Текстовые блоки пака-1 заново по разбору v8 (reblock-pack): объекты, надписи, ответы DeepSeek — как в v8, блоки —
# текущим детектором (TEXT_BLOCKS_VERSION 10, 2026-11-01).
#
# Зачем. Итог v8 ($HOT_MTS_ROOT/pack1_page_analysis_v8, таблицы TABLES_VERSION 8) посчитал блоки разбором v9. После
# него в text_blocks: разрез ряда по набору «кегль + жирность» (blocks.RowStyleSplit.STYLE), продолжение висячего списка
# не подпись (blocks.HangingList.ALIGN), подпись по центру своей оси — свой блок (flush_split.FlushChange.TO_CENTRE).
# A/B на 622 полосах P∪N∪C: 16 + 6 полос изменилось, все лучше; ХиП 1991 — 1 полоса лучше
# (ocr_utils/page_layout/text_blocks/README.md, «Разрез ряда по набору и висячие списки (v10)»).
#
# Читает: $SHARPENED_DIR, $FROM/pages/*.json (только чтение). Пишет: $OUT (pages/, overlays/).
# Второй проход защиты сторон (CRAFT + pero, GPU) — по умолчанию, как в v8.
#
# ВРЕМЯ (оценка по итогу v8: 12 135 полос за ~50 мин на 12 воркерах плюс защита сторон ~30 мин) ≈ 1.5 ч.
# JOBS=12 — потолок пользователя 75 % CPU (16 физических ядер).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/page_layout/run_pack1_reblock_v8.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

FROM="${FROM:-$HOT_MTS_ROOT/pack1_page_analysis_v8}"
OUT="${OUT:-$HOT_MTS_ROOT/pack1_page_analysis_v8_blocks_v10}"
JOBS="${JOBS:-12}"

uv run python -m ocr_utils.page_layout reblock-pack \
    --sharpened-dir "$SHARPENED_DIR" --from "$FROM" --out-dir "$OUT" --jobs "$JOBS" --axis body --gutter-mode short "$@"
