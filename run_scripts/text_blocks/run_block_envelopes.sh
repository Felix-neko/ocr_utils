#!/usr/bin/env bash
#
# Стенд границ и объединения текстовых блоков (research/block_envelopes): тестовое множество пака-1
# (105 проблемных полос: примеры пользователя + кандидаты по мерам; 80 чистых), чужие движки на нём,
# переключаемые пары алгоритмов «группировка__граница» с мерами и оверлеями.
#
# Вход — тот же, что у разбора v3 (объекты и запреты из pack1_page_analysis_v3, вторая ось строки).
# Кэш входа блоковой стадии (``capture``) — ~0.9 МБ и ~3 с на полосу (при свободной машине);
# сами алгоритмы по кэшу — 0.5–2 с на полосу.
#
# Шаги (ШАГ=all — capture, engines, run подряд):
#   select  — меры по JSON прогона research.gutter_crossing legacy → select/measures.csv, candidates.csv;
#             множества собраны из них и лежат в research/block_envelopes/sets (в git);
#   capture — кэш входа по полосам множеств (идемпотентно);
#   engines — чужие движки ПОСЛЕДОВАТЕЛЬНО (GPU один): surya layout и pero — на всех полосах,
#             kraken, chronicling, eynollah — на подмножестве sets/engines_subset.txt (50 полос;
#             при загруженной машине pero ~35 с/полосу, eynollah — минуты);
#   run     — сетка пар алгоритмов → $OUT/algos/<пара>/{overlays,pages,summary.csv};
#   pack    — весь пак «только текст» (~9.9 тыс. полос): кэш входа в тот же $OUT/cache (записи кандидатов —
#             $RECORDS: work/pages разбора v3 с 2026-09-28 пересобирается другой сессией, v3 строился на
#             work/pages_before_rules_v2), затем reach__fit с оверлеями и боевой legacy__legacy только
#             с мерами → $PACK_OUT/algos/<пара>; кэш ~2 МБ/полосу, оверлей ~0.9 МБ.
#
# Ждать по сохранённому PID:
#     setsid bash run_scripts/text_blocks/run_block_envelopes.sh > log 2>&1 < /dev/null & PID=$!
#     while kill -0 "$PID" 2>/dev/null; do sleep 30; done

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/../scan_markup/pack1/common.sh"
cd "$REPO"

OUT="${OUT:-/mnt/system/raw/mts/pack1_block_envelopes}"
PACK="${PACK:-/mnt/system/raw/mts/pack1_page_analysis_v3}"
GUTTER_PAGES="${GUTTER_PAGES:-/mnt/system/raw/mts/pack1_gutter_crossing/legacy/pages}"
SETS="$REPO/research/block_envelopes/sets"
JOBS="${JOBS:-8}"
STEP="${STEP:-all}"
PACK_OUT="${PACK_OUT:-/mnt/system/raw/mts/pack1_block_envelopes_pack}"
RECORDS="${RECORDS:-$PACK/work/pages_before_rules_v2}"
ALGOS="${ALGOS:-legacy__legacy legacy__bands legacy__fit legacy__alpha reach__legacy reach__fit reach__bands graph__fit}"

stand() {
    uv run python -m research.block_envelopes "$@"
}

mkdir -p "$OUT"
cat "$SETS/problem.txt" "$SETS/clean.txt" > "$OUT/sets_all.txt"

if [[ "$STEP" == select ]]; then
    stand select --pack-dir "$PACK" --pages-dir "$GUTTER_PAGES" --out-dir "$OUT/select"
fi
if [[ "$STEP" == all || "$STEP" == capture ]]; then
    stand capture --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --cache-dir "$OUT/cache" \
        --keys "$OUT/sets_all.txt" --jobs "$JOBS"
fi
if [[ "$STEP" == all || "$STEP" == engines || "$STEP" == engines_fast ]]; then
    stand engines --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --cache-dir "$OUT/cache" --out-dir "$OUT" \
        --keys "$OUT/sets_all.txt" --engine surya_layout --engine pero
fi
if [[ "$STEP" == all || "$STEP" == engines || "$STEP" == engines_slow ]]; then
    stand engines --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --cache-dir "$OUT/cache" --out-dir "$OUT" \
        --keys "$SETS/engines_subset.txt" --engine kraken --engine chronicling --engine eynollah
fi
if [[ "$STEP" == all || "$STEP" == run ]]; then
    ARGS=()
    for ALGO in $ALGOS; do ARGS+=(--algo "$ALGO"); done
    stand run --cache-dir "$OUT/cache" --out-dir "$OUT" --keys "$OUT/sets_all.txt" "${ARGS[@]}" --jobs "$JOBS"
fi
if [[ "$STEP" == pack ]]; then
    mkdir -p "$PACK_OUT"
    stand capture --pack-dir "$PACK" --sharpened-dir "$SHARPENED_DIR" --cache-dir "$OUT/cache" \
        --records-dir "$RECORDS" --jobs "${CAPTURE_JOBS:-16}"
    # Ключи — все полосы «только текст» без поворота, у которых есть кэш.
    uv run python -c "
from pathlib import Path
from research.block_envelopes.cli import only_text_keys
from research.block_envelopes.capture import cache_path
keys = [k for k in only_text_keys(Path('$PACK')) if cache_path(Path('$OUT/cache'), k).exists()]
Path('$PACK_OUT/keys.txt').write_text(chr(10).join(keys) + chr(10), encoding='utf-8')
print('полос с кэшем:', len(keys))
"
    stand run --cache-dir "$OUT/cache" --out-dir "$PACK_OUT" --keys "$PACK_OUT/keys.txt" --algo reach__fit \
        --jobs "$JOBS"
    stand run --cache-dir "$OUT/cache" --out-dir "$PACK_OUT" --keys "$PACK_OUT/keys.txt" --algo legacy__legacy \
        --no-overlays --jobs "$JOBS"
fi
