#!/usr/bin/env bash
#
# Кадры пака-6 (МТС, РИСК), где строки своей печати срезаны краем листа (лист загибается, строка уходит за
# кромку): низ, верх, наружный край; срез у корешка — отдельно (links_by_side/spine*).
#
# Читает: все кадры $PACK6_ROOT кроме скрытых папок (2 416 JPEG 5687×4268), в ½ разрешения.
# Пишет: $EDGE_CUT_PACK6_OUT/{scores.csv, lines.jsonl, overlays/, crops/, sheets/, cut_pages.csv, links/,
#        links_by_side/}.
# Время: ~6–8 мин на пак при 12 воркерах (≈6 кадр/с, упор в чтение с NTFS).
# Числа: размеры детектора — константы research/edge_cut_pages/detect.py под ½ кадра (~6.9 px/мм); пороги
# отбора MIN_SCORE / MIN_OUTER — по распределению пака и отсмотру поясов (research/edge_cut_pages/README.md).

set -euo pipefail
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "$(dirname "$0")/common.sh"

# Ниже 16: чтение ~10 МБ JPEG с NTFS-3G упирается в диск.
JOBS="${JOBS:-12}"
MIN_SCORE="${MIN_SCORE:-3}"
MIN_OUTER="${MIN_OUTER:-2}"
MIN_SPINE="${MIN_SPINE:-8}"

uv run python -m research.edge_cut_pages run --root "$PACK6_ROOT" --out "$EDGE_CUT_PACK6_OUT" --jobs "$JOBS"
uv run python -m research.edge_cut_pages select --out "$EDGE_CUT_PACK6_OUT" --min-score "$MIN_SCORE" --min-outer "$MIN_OUTER" --min-spine "$MIN_SPINE"
uv run python -m research.edge_cut_pages sheets --out "$EDGE_CUT_PACK6_OUT"
