#!/usr/bin/env bash
#
# scan_cropping v2 по «Хозяйство и право» 1991/02 (пробный пак, 85 фото-разворотов JPG 5392×4394 на чёрном коврике,
# пальцы) — вход стенда research/page_cutting (группа curved: страницы изогнуты сверху, снизу и по бокам, IMG_1162,
# IMG_1168). Ключи — боевые пака-6 (v2_pack6_args.sh) плюс осветление тени пальца, как у run_v2_ph_pack4_1975.sh.
# Числа:
#   --px-per-mm 17.1 — плотность бумаги на кадре: здесь «434 dpi» из EXIF совпадает с замером (рамка книги с обложкой
#       4444×3928 px на кадре IMG_1168, страница ~130×205 мм ≈ 2200×3500 px), в отличие от пака-6 (13.8 при «450 dpi»).
#   --force-dpi 434 — dpi в метаданных итога.
#
# Читает: $HP_1991_02_ROOT (Я.Диск, только чтение).
# Пишет: $HP_1991_02_WORK/{output,cache,debug}/ на SSD hotstore.
# Время: ~10–11 с на кадр (упор в GPU), ~15 мин на выпуск; видеопамять — пик ~7.4 ГБ.
# Перезапуск безопасен: готовые итоги пропускаются (--skip-if-exists), кэш шагов берётся с диска.
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

JOBS="${JOBS:-12}"

mkdir -p "$HP_1991_02_WORK"
echo "=== scan_cropping v2: $HP_1991_02_ROOT → $HP_1991_02_WORK/output"
# Ключи после V2_PACK6_ARGS перекрывают одноимённые оттуда (click берёт последнее значение).
uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$HP_1991_02_ROOT" \
    --output "$HP_1991_02_WORK/output" \
    --cache "$HP_1991_02_WORK/cache" \
    --debug "$HP_1991_02_WORK/debug" \
    "${V2_PACK6_ARGS[@]}" \
    --px-per-mm 17.1 \
    --force-dpi 434 \
    --finger-zone-light-increment 0,0 \
    --finger-shadow \
    --jobs "$JOBS" \
    --log-level INFO
