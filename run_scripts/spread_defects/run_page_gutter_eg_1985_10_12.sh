#!/usr/bin/env bash
#
# Уход текста под корешок на ЭГ 1985 10-12 (Химки): 322 кадра RAF, одна полоса на кадр, корешок сверху.
# Шаги: превью из RAF → ориентация (tesseract OSD; вкладки с прямым текстом — строки параллельны сгибу, в surya не
# идут, решены глазами в sets/eg_1985_10_12_gutter.csv) → surya по ленте у сгиба (~8 с/кадр, GPU) → DeepSeek-OCR-2
# по спорному поясу → раскладка по трём папкам симлинков.
#
# Чтение (v4): лента у сгиба режется из полного RAW (rawpy, 5178×7752; строка ~53 px против ~31 на превью),
#   линия сгиба — по превью в масштабе кадра; ~25 с/кадр, ~2 ч на 302 кадра (surya на большой ленте).
# Линия сгиба (v2, 2026-10-06): хребет складки + обход прищепок + RANSAC, запрет проходить сквозь строки surya
#   (кэш fold-layout); маска загиба — выше линии минус 14 px. Эталон линии — sets/eg_1985_10_12_fold_gt.csv.
# Пороги: доля незнакомых словарю крайних у сгиба слов (surya, без слов с заглавной, переносы склеиваются и через
#   дефис): норма < 0.05 (примеры пользователя 0.00–0.05), потеря ≥ 0.16 (примеры 0.17–0.72), между — «непонятно».
#   Отступ строк от сгиба в решение не входит (ложные тревоги 1350_2, 1280_2, 1550_2). Шум OCR > 0.08 — «непонятно».
#
# Решение v4 — по согласию двух чтений: RAW (features.csv) и превью (features_v3.csv, кэш ocr_surya_preview_v2/;
#   получается шагом 4 без --raw-dir в отдельную папку). Одно чтение по RAW: 89/127/76, по превью: 99/125/68.
# v5: + геометрическая норма (касаются складки < 0.4 и отступ p10 ≥ 0.35 высоты строки) и пиксели у складки
#   (page-gutter-pixels: медиана просвета крайнего знака до тени складки ≤ −0.10 — потеря, > −0.03 — буквы целы;
#   между — словарь ≥ 0.10 потеря). Пороги — по двугорбому распределению просвета по папке; разметка пользователя
#   (sets/eg_1985_10_12_gutter_labels.csv, 50 кадров) — проверка: v4 14/50, v5 48/50.
# Итог v5: 139 потеря / 177 норма / 6 непонятно. Итог v4: 95 / 174 / 53.
# Итог v3 (2026-10-06, колонка у сгиба — ширина 3 высоты строки от ближнего края): 103 потеря / 151 норма / 68 непонятно —
# $EG_1985_10_12_OUT/verdicts.csv и links/.
set -euo pipefail
cd "$(dirname "$0")/../.."
source run_scripts/spread_defects/common.sh

OUT="$EG_1985_10_12_OUT"
JOBS="${JOBS:-8}"   # извлечение превью упирается в NTFS-3G, больше 8 не ускоряет
mkdir -p "$OUT/previews"

# 1. Превью из RAF (4416×2944), идемпотентно.
find "$EG_1985_10_12_RAW" -maxdepth 1 -name '*.RAF' -print0 |
  xargs -0 -P "$JOBS" -I{} sh -c 'f="{}"; o="'"$OUT"'/previews/$(basename "$f" .RAF).jpg"; [ -s "$o" ] || exiftool -b -PreviewImage "$f" > "$o"'

# 2. Ориентация: кадры с прямым текстом (OSD Rotate 0, уверенность > 3) — строки параллельны сгибу.
if [ ! -s "$OUT/parallel_to_fold.txt" ]; then
  for f in "$OUT"/previews/*.jpg; do
    r=$(convert "$f" -resize 50% png:- | OMP_THREAD_LIMIT=1 tesseract - - --psm 0 -l osd 2>/dev/null |
      awk '/Rotate:/{r=$2} /Orientation confidence:/{c=$3} END{print r, c}')
    echo "$(basename "$f" .jpg) $r"
  done > "$OUT/osd_tesseract.txt"
  awk '$2==0 && $3>3 {print $1}' "$OUT/osd_tesseract.txt" > "$OUT/parallel_to_fold.txt"
fi

# 3. Разметка ленты у сгиба (surya layout, строки surya, DeepSeek grounding) — рамки для линии сгиба.
uv run python -m research.spread_defects fold-layout --previews "$OUT/previews" --stems <(echo stem; ls "$OUT/previews" | sed 's/\.jpg$//' | grep -vxFf "$OUT/parallel_to_fold.txt") \
  --out-dir "$OUT/fold_variants"

# 4. surya по ленте у сгиба (кэш ocr_surya/), признаки features.csv.
uv run python -m research.spread_defects page-gutter --previews "$OUT/previews" --out-dir "$OUT" \
  --skip "$OUT/parallel_to_fold.txt" --layout-dir "$OUT/fold_variants/layout" --raw-dir "$EG_1985_10_12_RAW" --rec-batch 16 --jobs 6

# 5. DeepSeek-OCR-2 по спорному поясу surya — для сверки (deepseek.csv), в решение не входит; при чтении по RAW
#    (шаг 4 с --raw-dir) кадры пропускаются — вырезки для сверки режутся только по превью.
uv run python -m research.spread_defects page-gutter-deepseek --out-dir "$OUT" --previews "$OUT/previews" \
  --low 0.05 --high 0.16

# 5б. Пиксельные признаки у складки по полному RAW (~3 мин, 12 процессов) и оверлеи «линия разреза + буквы»
#     по размеченным кадрам — $OUT/overlays_cut/.
uv run python -m research.spread_defects page-gutter-pixels --out-dir "$OUT" --raw-dir "$EG_1985_10_12_RAW" \
  --overlay-stems research/spread_defects/sets/eg_1985_10_12_gutter_labels.csv --jobs 12

# 6. Раскладка: links/{есть_потеря_текста_у_корешка,текст_у_корешка_в_норме,непонятно}/ — симлинки на RAF.
uv run python -m research.spread_defects page-gutter-sort --out-dir "$OUT" --raw-dir "$EG_1985_10_12_RAW" \
  --manual research/spread_defects/sets/eg_1985_10_12_gutter.csv \
  --loss-share 0.16 --normal-share 0.05 --second-features "$OUT/features_v3.csv"
