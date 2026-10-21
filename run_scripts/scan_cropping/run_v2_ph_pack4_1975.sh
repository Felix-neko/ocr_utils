#!/usr/bin/env bash
#
# scan_cropping v2 по паку-4 ПХ, 1975, выпуски 01–12 (1027 кадров разворотов на чёрном коврике, пальцы) — с ключами
# боевого прогона пака-6 (v2_pack6_args.sh) и осветлением тени пальца по карте, как у run_v2_pack5_1990_q1.sh:
#   --finger-zone-light-increment 0,0 — без высветления зоны пальца перед LaMa (оно оставляло светлый ореол);
#   --finger-shadow — карта тени пальца (ядро + 15 мм, в книге) и прибавка к итогу после copy back.
# Плотность — 13.8 px/мм пака-6: у кадров пака-4 та же (research/edge_cut_pages, ~6.9 px/мм на ½ кадра).
#
# Читает: $PACK4_PH_ROOT/1975/<выпуск>/ (Я.Диск, только чтение; --recursive по папке года).
# Пишет: $PACK4_PH_WORK/{output,cache,debug}/1975/<выпуск>/ на SSD hotstore; в «Готовое» на Я.Диске — после отсмотра.
# Время: ~10–11 с на кадр (упор в GPU: пальцы, SAM 3 ×2, surya), ~3 часа на год. Видеопамять — пик ~7.4 ГБ.
# Место: TIFF ~70 МБ на кадр, ~75 ГБ на год.
# Перезапуск безопасен: готовые итоги пропускаются (--skip-if-exists), кэш шагов берётся с диска.
#
set -euo pipefail

set -m
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

WORK="${PACK4_WORK:-$PACK4_PH_WORK}"
YEAR="${YEAR:-1975}"
JOBS="${JOBS:-12}"  # как у пака-6: CPU-шаги идут параллельно GPU и упираются в чтение с NTFS

mkdir -p "$WORK"
echo "=== scan_cropping v2: $PACK4_PH_ROOT/$YEAR → $WORK/output/$YEAR"
uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$PACK4_PH_ROOT/$YEAR" \
    --output "$WORK/output/$YEAR" \
    --cache "$WORK/cache/$YEAR" \
    --debug "$WORK/debug/$YEAR" \
    "${V2_PACK6_ARGS[@]}" \
    --finger-zone-light-increment 0,0 \
    --finger-shadow \
    --jobs "$JOBS" \
    --log-level INFO
