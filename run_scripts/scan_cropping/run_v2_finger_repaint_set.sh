#!/usr/bin/env bash
#
# База стенда research/finger_repaint (тёмные пятна LaMa на месте пальцев): прогон v2 с ключами пака-6 на множествах
# sets/dark_spots.txt (28 кадров с серым пятном в pack6_v3) и sets/control.txt (15 кадров с пальцем на бумаге без пятна),
# плюс --save-fingers: кадр без пальцев (clean.png, первый проход LaMa) остаётся в кэше — вход второго прохода.
# Кэш копируется из pack6_v3: SAM 3 и бумага не пересчитываются; пальцы (LaMa) — заново (clean.png в pack6_v3 не хранился),
# surya — заново (layout.json версии 3 без типов блоков). ~43 кадра, ~10 мин.
# Читает: $PACK6_450_ROOT (Я.Диск, только чтение), кэш pack6_v3. Пишет: $FINGER_REPAINT_ROOT/{input (симлинки), $BASE_NAME/{output,cache,debug}}.
#
# BASE_NAME — имя папки базы (по умолчанию base); EXTRA_ARGS — ключи v2 после ключей пака-6 (перекрывают их), например
# база без высветления зоны пальца перед LaMa:
#   BASE_NAME=base_nolight EXTRA_ARGS="--finger-zone-light-increment 0,0" ./run_scripts/scan_cropping/run_v2_finger_repaint_set.sh
#
set -euo pipefail

# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

FINGER_REPAINT_ROOT="${FINGER_REPAINT_ROOT:-/mnt/hotstore/scan_processing/finger_repaint}"
WORK="$FINGER_REPAINT_ROOT/${BASE_NAME:-base}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
CACHE_FROM="${CACHE_FROM:-/mnt/hotstore/scan_processing/scan_cropping_v2/pack6_v3/cache}"
INPUT="$FINGER_REPAINT_ROOT/input"
JOBS=12

mkdir -p "$INPUT" "$WORK/cache"
for set in research/finger_repaint/sets/dark_spots.txt research/finger_repaint/sets/control.txt; do
    grep -v '^#' "$set" | sed '/^$/d' | while IFS= read -r rel; do
        mkdir -p "$INPUT/$(dirname "$rel")"
        ln -sfn "$PACK6_450_ROOT/$rel" "$INPUT/$rel"
        key="$(echo "${rel%.*}" | sed 's#/#__#g')"
        if [ ! -d "$WORK/cache/$key" ]; then
            cp -r "$CACHE_FROM/$key" "$WORK/cache/$key"
        fi
    done
done

uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$INPUT" \
    --output "$WORK/output" \
    --cache "$WORK/cache" \
    --debug "$WORK/debug" \
    "${V2_PACK6_ARGS[@]}" \
    "${EXTRA[@]}" \
    --save-fingers \
    --jobs "$JOBS" \
    --log-level INFO
