#!/usr/bin/env bash
#
# Проверка правок закраса (--fill-before-copy-back, --fill pull_push) на тестовом
# множестве стенда research/border_fill: sets/trails.txt (55 кадров со шлейфом закраса в pack6_v2) и sets/control.txt
# (27 кадров без шлейфа, цветные полосы) — 82 кадра, ~15 мин. Ключи — как у полного прогона (v2_pack6_args.sh), плюс
# --save-fingers: кадр без пальцев остаётся в кэше — вход стенда.
# Читает: $PACK6_450_ROOT (Я.Диск, только чтение), кэш $PACK6_450_WORK/cache (маски, SAM 3, surya — копируются,
# чтобы пересчитывались только пальцы и финал). Пишет: $BORDER_FILL_ROOT/{input (симлинки), v2_after/{output,cache,debug}}.
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

BORDER_FILL_ROOT="${BORDER_FILL_ROOT:-/mnt/hotstore/scan_processing/border_fill}"
# Папка прогона и откуда брать кэш: повтор с другими ключами — `RUN_NAME=v2_pull_push CACHE_FROM=$BORDER_FILL_ROOT/v2_after/cache ./...`
# (кэш с clean.png: пересчитывается только финал).
WORK="$BORDER_FILL_ROOT/${RUN_NAME:-v2_after}"
CACHE_FROM="${CACHE_FROM:-$PACK6_450_WORK/cache}"
INPUT="$BORDER_FILL_ROOT/input"
JOBS=12

mkdir -p "$INPUT" "$WORK/cache"
for set in research/border_fill/sets/trails.txt research/border_fill/sets/control.txt; do
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
    --save-fingers \
    --jobs "$JOBS" \
    --log-level INFO
