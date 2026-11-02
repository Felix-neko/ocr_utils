#!/usr/bin/env bash
#
# Разбор страниц пака-1 v8: таблицы и линейки-сироты текущим детектором (TABLES_VERSION 8, 2026-11-01).
#
# Зачем. Полный разбор v5 ($HOT_MARKUP_ROOT/pack1_page_analysis_v5, 2026-09-29) посчитан детектором таблиц v4 (фильтр
# ложных линеек-сирот tables/loose_filter.py). С тех пор детектор менялся: v5 — стык линейки через узел с допуском
# 1 мм, v6 — график со штриховкой — рисунок, v7 — тень у края вырезки не штриховая краска, v8 — признаки по вырезке,
# выпрямленной вдоль изогнутых линеек (page_layout/__init__.py, TABLES_VERSION). Стенды текстовых блоков до сих пор
# брали объекты из v4 (сироты до фильтра): на 622 полосах P∪N∪C стенда heading_merge отброс ложных сирот дал
# 30 лучше / 3 хуже из 48 заметно изменённых (2026-11-01).
#
# Как. scripts/recompute_pack_candidates.py: растр — из $DB_REVIEWED (как v3–v5); стадия кандидатов (растр, таблицы,
# сироты, кандидаты line art, вырезки) — заново по всем 12 135 полосам в $OUT/work; ответы DeepSeek-OCR-2 первого прохода из
# $PREVIOUS/work переносятся только для кандидатов с тем же id И той же рамкой вырезки (та же картинка — тот же
# ответ), новые вырезки досчитывает vLLM (GPU); затем второй проход, итог (текстовые блоки по второй оси,
# межколонники short) и защита сторон CRAFT + pero (GPU). $PREVIOUS только читается.
#
# Текстовые блоки считаются текущим кодом рабочего дерева — заново их можно пересобрать быстро (reblock-pack).
#
# ВРЕМЯ (оценка по v5: весь разбор 1 ч 50 мин на 16 воркерах): на 12 воркерах ≈ 2.5 ч плюс пуск vLLM.
# JOBS=12 — потолок пользователя 75 % CPU (16 физических ядер).
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/page_layout/run_pack1_analysis_v8.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/../scan_markup/pack1/common.sh"

OUT="${OUT:-$HOT_MTS_ROOT/pack1_page_analysis_v8}"
PREVIOUS="${PREVIOUS:-$HOT_MARKUP_ROOT/pack1_page_analysis_v5}"
JOBS="${JOBS:-12}"

PYTHONPATH=. uv run python scripts/recompute_pack_candidates.py \
    --sharpened-dir "$SHARPENED_DIR" --cache "$LAYOUT_CACHE_DIR" --previous "$PREVIOUS" --out-dir "$OUT" \
    --raster-db "$DB_REVIEWED" --pack-name "$PACK_NAME" --jobs "$JOBS" --axis body "$@"
