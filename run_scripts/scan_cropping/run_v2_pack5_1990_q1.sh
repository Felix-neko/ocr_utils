#!/usr/bin/env bash
#
# scan_cropping v2 по паку-5 МТС, 1990, выпуски 01–03 (182 кадра разворотов на чёрном коврике, пальцы) — с ключами
# боевого прогона пака-6 (v2_pack6_args.sh + осветление тени пальца по карте, как run_v2_pack6_shadow.sh):
#   --finger-zone-light-increment 0,0 — без высветления зоны пальца перед LaMa (оно оставляло светлый ореол);
#   --finger-shadow — карта тени пальца (ядро + 15 мм, в книге) и прибавка к итогу после copy back.
# Плотность — 13.8 px/мм пака-6: у пака-5 по шагу строк на 1–2 % ниже, расхождение не критично (решение пользователя
# 2026-10-19, замер — в common.sh).
#
# Выпуски идут по одному (--input-dir — папка выпуска, без --recursive): в папке 1990 пака-5 лежат и 04–06, их не
# трогаем. Структура выхода — 1990/<выпуск>/ в каждой из output, cache, debug.
# Читает: $PACK5_ROOT/1990/{01,02,03} (Я.Диск, только чтение).
# Пишет: $PACK5_1990_Q1_WORK/{output,cache,debug}/1990/<выпуск>/ на SSD hotstore; в «Готовое» на Я.Диске — после отсмотра.
# Время: ~10–11 с на кадр (упор в GPU: пальцы, SAM 3 ×2, surya), ~35 мин на три выпуска. Видеопамять — пик ~7.4 ГБ.
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

WORK="${PACK5_WORK:-$PACK5_1990_Q1_WORK}"
ISSUES="${ISSUES:-01 02 03}"
JOBS="${JOBS:-12}"  # как у пака-6: CPU-шаги идут параллельно GPU и упираются в чтение с NTFS

mkdir -p "$WORK"
for issue in $ISSUES; do
    rel="1990/$issue"
    echo "=== scan_cropping v2: $PACK5_ROOT/$rel → $WORK/output/$rel"
    uv run python -m ocr_utils.scan_cropping.v2 \
        --input-dir "$PACK5_ROOT/$rel" \
        --output "$WORK/output/$rel" \
        --cache "$WORK/cache/$rel" \
        --debug "$WORK/debug/$rel" \
        "${V2_PACK6_ARGS[@]}" \
        --finger-zone-light-increment 0,0 \
        --finger-shadow \
        --jobs "$JOBS" \
        --log-level INFO
done
