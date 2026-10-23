#!/usr/bin/env bash
#
# Продление нижней кромки листа (обрезанной переплётчиком) и допечатка срезанных нижних строк в кадрах
# РИСК 1991/05 пака-6 (основа) по набору вердиктов research/cut_lines/sets/risk_1991_05_bottom.csv
# (47 кадров, отсмотрены глазами; тексты срезанных строк — по подшивке пака-7).
# Читает: кадры основы и донора с Я.Диска (только чтение), кэш scan_cropping v2 (маски страниц, пальцы,
# LaMa-кадр), pairs.csv стенда (пары кадров — только для сверки текста tesseract).
# Пишет на SSD: frames/ (47 кадров, JPEG q95, холст может быть выше исходного), sheets/ («было | стало» по
# нижним полосам), lines.csv (статусы по страницам и колонкам, перечитанный текст, сверка с донором).
# Время: ~30 с на кадр с допечаткой (tesseract по строкам колонки), ~5 с без неё; при 12 воркерах — ~3 мин.
# Числа: EXTEND_MM — сдвиг кромки (пользователь: 8–10 мм); PX_PER_MM — замер по кадрам пака-6: страница
# 145 мм ≈ 2330 px → 16 px/мм. Остальные размеры — в шапках research/cut_lines/{extend,fragments}.py.
#
set -euo pipefail
set -m
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/cut_lines/common.sh

JOBS="${JOBS:-12}"  # не 16: чтение ~10 МБ JPEG с NTFS упирается в диск
EXTEND_MM="${EXTEND_MM:-9}"
PX_PER_MM="${PX_PER_MM:-16}"
OUT="$CUT_LINES_ROOT/risk_1991_05_extend"

# Общая библиотека букв подшивки (буквы с соседних страниц): ~3 мин при 12 воркерах; есть — не пересобирается.
KEYS="$(python3 -c "import csv; print(','.join(sorted({r['frame'] for r in csv.DictReader(open('research/cut_lines/sets/risk_1991_05_bottom.csv', encoding='utf-8'))})))")"
mkdir -p "$OUT"
[ -f "$OUT/library.npz" ] || uv run python -m research.cut_lines harvest --base-dir "$RISK_1991_05_BASE" \
    --v2-work "$RISK_1991_05_V2_WORK" --keys "$KEYS" --out "$OUT/library.npz" --jobs "$JOBS"

uv run python -m research.cut_lines extend \
    --base-dir "$RISK_1991_05_BASE" \
    --donor-dir "$RISK_1991_05_DONOR" \
    --v2-work "$RISK_1991_05_V2_WORK" \
    --pairs-root "$CUT_LINES_ROOT/risk_1991_05" \
    --set research/cut_lines/sets/risk_1991_05_bottom.csv \
    --out-root "$OUT" \
    --extend-mm "$EXTEND_MM" \
    --px-per-mm "$PX_PER_MM" \
    --library "$OUT/library.npz" \
    --jobs "$JOBS"
