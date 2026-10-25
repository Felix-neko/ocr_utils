#!/usr/bin/env bash
#
# «Хозяйство и право» 1991 (пробный пак): журнальные развороты JPG ~5400×4300, 434 dpi, выпуски 01–12
# (~1 000 кадров на 2026-10-21; в 05, 07 и 12 есть и постраничные кадры, 12 — неполный). Проверка на пропуски
# страниц, смаз, текст у корешка и расфокус. Прогон 2026-10-08 (619 кадров, без корешка и карты зон) — в hp_1991_defects.
#   1. pages (верхний и нижний колонтитул) + sequence — номера страниц и разрывы ряда по выпускам;
#   2. features + pass2 + motion-rank — смаз: относительный кепстр плиток и полутоновый ореол; правило пака-4 ПХ:
#      agree_peak ≥ 6 и mid_at_peak_rel ≥ 1.15; motion-sheets — листы крупов худших кадров;
#      motion-zones + motion-zone-sheets — карта смаза по зонам всего кадра (отсматривать её, навык reshoot-review);
#   2a. pass3 + pass3-rank — корешок: сжатие по шагу штрихов < 0.62, порча черт по 4 худшим строкам (worst_run) < 0.60;
#   3. defocus_detection — расфокус DOM с пресетом dom-si (абсолютные теги), худшие 30 % и зональный отчёт.
#
# Читает: $HP_1991_PACK_ROOT/1991/<выпуск>/IMG_*.jpg (Я.Диск, только чтение).
# Пишет: $HP_1991_OUT/{pages/,pages_footer/,motion/,midtone/,glyphs.csv,pass3_scores.csv,lines/,defocus/}.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"
cd "$(dirname "$0")/../.."

# Ниже 16: диск NTFS-3G.
JOBS="${JOBS:-12}"
ROOT="$HP_1991_PACK_ROOT"
OUT="$HP_1991_OUT"
SD=(uv run python -m research.spread_defects)

"${SD[@]}" pages --root "$ROOT" --only 1991 --out-dir "$OUT/pages" --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages"
"${SD[@]}" pages --root "$ROOT" --only 1991 --out-dir "$OUT/pages_footer" --place footer --jobs "$JOBS"
"${SD[@]}" sequence --run-dir "$OUT/pages_footer"

"${SD[@]}" features --root "$ROOT" --only 1991 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass2 --root "$ROOT" --only 1991 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-rank --run-dir "$OUT/motion"
"${SD[@]}" motion-sheets --run-dir "$OUT/motion" --count 48 --out-dir "$OUT/motion/sheets"
"${SD[@]}" motion-zones --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" motion-zone-sheets --out-root "$OUT" --jobs "$JOBS"

"${SD[@]}" pass3 --root "$ROOT" --only 1991 --out-root "$OUT" --jobs "$JOBS"
"${SD[@]}" pass3-rank --out-root "$OUT"
uv run python -c "from pathlib import Path; from research.spread_defects.worst_run import rank; rank(Path('$OUT')).to_csv(Path('$OUT') / 'worst_run.csv', index=False)"

# Расфокус: пресет dom-si калиброван на разворотах (RAF-превью СИ); здесь JPG того же порядка разрешения.
mkdir -p "$OUT/defocus"
uv run python -m ocr_utils.defocus_detection "$ROOT/1991" --recursive \
    --tag-preset dom-si --worst-percent 30 --zonal-percent 30 --zonal-axis all \
    --txt-report "$OUT/defocus/defocus.txt" --md-report "$OUT/defocus/defocus.md" --csv "$OUT/defocus/defocus.csv" \
    --retake-list "$OUT/defocus/retake.txt" --link-dir "$OUT/defocus/worst" --reserve-cores 4
