#!/usr/bin/env bash
#
# Выпрямление ВОСЬМИ эталонных кривых полос пака-1 всеми движками dewarp.
#
# ЗАЧЕМ. FineReader выправляет кривые строки сам, но его коррекция геометрии портит
# прямые полосы. Если кривые полосы выпрямлять заранее отдельным движком, FineReader
# можно гонять без коррекции вовсе. Какой движок годится для флэтбед-скана журнала —
# заранее не известно: нейросети обучены на фотографиях документов, классика — на
# книжных разворотах. Поэтому все движки прогоняются по одним и тем же полосам, и
# результат смотрится глазами по парам «было | стало».
#
# ВХОД — заострённые копии на SSD (то, что ушло в FineReader), полное разрешение 600 dpi.
# ВЫХОД — папка на движок в /mnt/SYSTEM/raw/mts/pack1_dewarp/validate/ плюс compare/ и
# quality.{csv,md}. На /mnt/dump3 не писать: там Яндекс.Диск.
#
# ВРЕМЯ. textline — секунды на полосу, pagedewarp — до минуты (Powell на CPU), нейросети —
# секунды плюс загрузка весов. Всего минут десять. Запускать в фоне и ждать ПО PID:
#   ./run_scripts/scan_markup/pack1/run_dewarp_validate.sh & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 10; done
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Снятие всей группы процессов при выходе: пул на forkserver переживает смерть хозяина.
set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

OUT_DIR="/mnt/SYSTEM/raw/mts/pack1_dewarp/validate"

# Восемь полос, про которые известно, что строки кривые и FineReader их обычно улучшает.
PAGES=(
    1966/06/IMG_0144_1L.jpg
    1966/05/0440_2R.jpg
    1966/05/0350_2R.jpg
    1966/02/IMG_0086_1L.jpg
    1966/01/IMG_0049_1L.jpg
    1966/01/IMG_0048_2R.jpg
    1966/01/IMG_0051_2R.jpg
    1972/03/IMG_0151_1L.jpg
)

ARGS=(
    run
    --root "$SHARPENED_DIR"
    --out-dir "$OUT_DIR"
    --engines all
    --jobs 8                     # CPU-движки: полоса textline ~2 с, pagedewarp ~40 с
    # --out-dpi не задан: эталон смотрится в полном разрешении
)
for page in "${PAGES[@]}"; do
    ARGS+=(--only "$page")
done

echo "Dewarp эталонных полос → $OUT_DIR"
uv run python -m ocr_utils.dewarp "${ARGS[@]}" "$@"
