#!/usr/bin/env bash
#
# Оверлеи «без коррекции | с коррекцией» детектора порчи геометрии v18 по всем страницам пака-2, разложенные по
# вердикту (research.geometry_quality export --mode verdicts): good/, mixed/, bad/<правило>__<виновник>/ — причина bad:
# правило вердикта (unforgivable — непрощаемая метрика, hard — жёсткий порог, total — совокупность групп, no_gain —
# порча без выигрыша, hysteresis — порча ≥ ratio × выигрыш) и метрика или группа, на которой оно сработало. В шапке
# каждой картинки — вердикт, правило, score групп и таблица метрик. Пороги — те же, что у прогона (по умолчанию).
#
# Сначала — run_10_geometry_v18.sh.
# Читает: $GEOMETRY_V18_DIR, $GEO_PDF_DIR, $NOGEO_PDF_DIR, $GEOMETRY_LAYOUT_ROOT (line art и формулы — в опись).
# Пишет:  $GEOMETRY_OVERLAYS_DIR/{good,mixed,bad/<причина>}/<pdf>_pNNN_<вид>_<вердикт>_<правило>_<виновник>.jpg, index.csv.
# Время: ~1 с на картинку на воркер (рендер двух PDF) — ~600 страниц за пару минут.

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

# Рендер двух PDF и запись JPEG — упор и в CPU, и в диск: как у выгрузок пака-1 (run_pack1_v18_full.sh).
JOBS="${JOBS:-8}"
GEOMETRY_OVERLAYS_DIR="${GEOMETRY_OVERLAYS_DIR:-$GEOMETRY_V18_DIR/overlays}"

uv run python -m research.geometry_quality export --run-dir "$GEOMETRY_V18_DIR" --geo-dir "$GEO_PDF_DIR" \
    --nogeo-dir "$NOGEO_PDF_DIR" --out-dir "$GEOMETRY_OVERLAYS_DIR" --layout-root "$GEOMETRY_LAYOUT_ROOT" \
    --mode verdicts --jobs "$JOBS" "$@"
