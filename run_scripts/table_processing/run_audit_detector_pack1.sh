#!/usr/bin/env bash
#
# Разбор находок детектора таблиц: что принято, что отклонено и почему.
#
# Пишет оверлей каждой полосы с находками, вырезки по папкам «принятые» и «отклонённые»,
# контактные листы для просмотра глазами и CSV со всеми признаками. Пороги проверки стоят
# в research/legacy/table_processing/detection/verify.py и калиброваны по 61 находке, размеченной
# глазами (research/legacy/table_processing/labels/detector.csv).
#
# Замер до и после проверки на выборке в 396 полос всех одиннадцати лет пака:
#   было:  25 принято (из них 5 не таблицы), 8 отклонено (из них 2 настоящие таблицы);
#   стало: 22 принято (все таблицы), 11 отклонено (все не таблицы).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

uv run python -m research.legacy.table_processing audit-detector \
    --sharpened-dir "$SHARPENED_DIR" \
    --out-dir "$OUT_DIR" \
    --sample 400 \
    --jobs "$JOBS" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
wait "$PID"

echo "листы для просмотра: $OUT_DIR/detector_audit/листы"
