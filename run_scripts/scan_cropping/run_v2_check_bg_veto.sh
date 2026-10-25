#!/usr/bin/env bash
#
# Проверка вето фона scan_cropping v2 (--bg-veto, по умолчанию включено): фон flood отдаётся книге, где SAM 3 видит
# страницу, а SAM 2 по точкам коврика — не фон. Кадры — из стенда research/bound_volumes runs/bg_bays (2026-10-21):
#   пак-6, 9 кадров, где фон flood отнимал бумагу у страницы (рукописные номера, логотипы, линейки у края листа):
#       РИСК 1991/03 IMG_0347 (2.0 тыс. мм²), IMG_0382, РИСК 1991/02 IMG_0802, МТС 1991/07 IMG_0801, МТС 1991/04 p78,
#       МТС 1990/05 IMG_1219, РИСК 1991/05 IMG_0664, IMG_0611, IMG_0662;
#   пак-6, контроль: РИСК 1991/05 IMG_0661, IMG_0665 (полоска поля под линейкой, +38–43 мм²), РИСК 1993/04-05
#       IMG_0543 (палец, вето не должно ничего менять);
#   hp91: IMG_1091 (надпись «30 84» над заголовком, −361 мм² бумаги), IMG_1146 (яма от SAM 3, вето ни при чём).
# Ключи — как у полных прогонов (v2_pack6_args.sh; hp91 — как run_v2_hp_1991_02.sh). Сравнивать с
# $PACK6_V6_WORK и $HP_1991_02_WORK: итог, debug и cache/<кадр>/paper.npz.
#
# Читает: $PACK6_450_ROOT, $HP_1991_02_ROOT (Я.Диск, только чтение) — через ссылки в $WORK/input.
# Пишет: $WORK/{input,pack6,hp91}/{output,cache,debug}/ на SSD hotstore.
# Время: ~1 мин на загрузку моделей + ~10–15 с на кадр; видеопамять — пик ~7.4 ГБ + SAM 2 (~1 ГБ).
#
set -euo pipefail

trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/scan_cropping/common.sh
source run_scripts/scan_cropping/v2_pack6_args.sh

JOBS="${JOBS:-12}"
WORK="${CHECK_WORK:-$SCAN_CROPPING_V2_ROOT/check_bg_veto}"
PACK6_FRAMES=(
    "РИСК/1991/03 есть фотографии по страницам/IMG_0347.jpg"
    "РИСК/1991/03 есть фотографии по страницам/IMG_0382.jpg"
    "РИСК/1991/02 есть фотографии по страницам/IMG_0802.jpg"
    "МТС/1991/07 есть фотографии по страницам/IMG_0801.jpg"
    "МТС/1991/04 есть фотографии по страницам/063_pak7_1_0650_p78.jpg"
    "МТС/1990/05/IMG_1219.jpg"
    "РИСК/1991/05 есть фотографии по страницам/IMG_0664.jpg"
    "РИСК/1991/05 есть фотографии по страницам/IMG_0611.jpg"
    "РИСК/1991/05 есть фотографии по страницам/IMG_0662.jpg"
    "РИСК/1991/05 есть фотографии по страницам/IMG_0661.jpg"
    "РИСК/1991/05 есть фотографии по страницам/IMG_0665.jpg"
    "РИСК/1993/04-05/IMG_0543.jpg"
)
HP_FRAMES=(IMG_1091.jpg IMG_1146.jpg)

# Вход — ссылки на исходники с той же структурой папок, что у полного прогона: ключи кэша совпадают с pack6_v6.
for rel in "${PACK6_FRAMES[@]}"; do
    mkdir -p "$WORK/input/pack6/$(dirname "$rel")"
    ln -sfn "$PACK6_450_ROOT/$rel" "$WORK/input/pack6/$rel"
done
mkdir -p "$WORK/input/hp91"
for name in "${HP_FRAMES[@]}"; do
    ln -sfn "$HP_1991_02_ROOT/$name" "$WORK/input/hp91/$name"
done

echo "=== пак-6"
uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$WORK/input/pack6" \
    --output "$WORK/pack6/output" \
    --cache "$WORK/pack6/cache" \
    --debug "$WORK/pack6/debug" \
    "${V2_PACK6_ARGS[@]}" \
    --bg-veto \
    --jobs "$JOBS" \
    --log-level INFO

echo "=== hp91"
uv run python -m ocr_utils.scan_cropping.v2 \
    --input-dir "$WORK/input/hp91" \
    --output "$WORK/hp91/output" \
    --cache "$WORK/hp91/cache" \
    --debug "$WORK/hp91/debug" \
    "${V2_PACK6_ARGS[@]}" \
    --px-per-mm 17.1 \
    --force-dpi 434 \
    --finger-zone-light-increment 0,0 \
    --finger-shadow \
    --bg-veto \
    --jobs "$JOBS" \
    --log-level INFO
