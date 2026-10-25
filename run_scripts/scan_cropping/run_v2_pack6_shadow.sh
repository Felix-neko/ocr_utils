#!/usr/bin/env bash
#
# Полный прогон scan_cropping v2 по паку-6 с осветлением тени пальца по карте («A0 + P», research/finger_repaint):
# ключи пака-6 (v2_pack6_args.sh) плюс
#   --finger-zone-light-increment 0,0 — без высветления зоны пальца перед LaMa: оно само оставляло светлый ореол
#       (контроль РИСК 1991/02 IMG_0790), а тень теперь снимает карта;
#   --finger-shadow — после copy back и уровней: зона поиска тени (ядро пальца + 15 мм, в книге), карта «pull-push по
#       кольцу бумаги 10 мм − сглаженная бумага без краски, кожи и печати», прибавка к итогу. На 28 кадрах с пятном
#       остаток тени 17.8 тыс. → 3.2 тыс. мм², на 15 контрольных 338 → 9 мм², ~0.5 с CPU на кадр.
# Кэш копируется из pack6_v3 (фон, SAM 3, бумага, страницы не пересчитываются); пальцы (LaMa) — заново: кадра без
# пальцев в кэше pack6_v3 нет; surya — заново: layout.json версии 3 без типов блоков. ~2480 кадров, ориентир 3–4 ч.
# Читает: $PACK6_450_ROOT (Я.Диск, только чтение), кэш pack6_v3. Пишет: $WORK/{output,cache,debug,run.log}; на кадр в
# debug — оверлей main/<кадр>_main.jpg и картинка тени shadow/<кадр>_shadow.jpg (зона поиска | карта прибавки).
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

WORK="${PACK6_SHADOW_WORK:-$SCAN_CROPPING_V2_ROOT/pack6_v4_shadow}"
CACHE_FROM="${CACHE_FROM:-$SCAN_CROPPING_V2_ROOT/pack6_v3/cache}"
JOBS=12  # как run_v2_pack6.sh: CPU-шаги идут параллельно GPU и упираются в чтение с NTFS

mkdir -p "$WORK"
if [ ! -d "$WORK/cache" ]; then
    echo "копирую кэш $CACHE_FROM → $WORK/cache"
    cp -r "$CACHE_FROM" "$WORK/cache"
fi
echo "scan_cropping v2 (тень пальца по карте): $PACK6_450_ROOT → $WORK/output"
uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$PACK6_450_ROOT" \
    --output "$WORK/output" \
    --cache "$WORK/cache" \
    --debug "$WORK/debug" \
    "${V2_PACK6_ARGS[@]}" \
    --finger-zone-light-increment 0,0 \
    --finger-shadow \
    --jobs "$JOBS" \
    --log-level INFO
