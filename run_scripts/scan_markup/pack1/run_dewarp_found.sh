#!/usr/bin/env bash
#
# Выпрямление полос, которые детектор кривых строк отметил как кривые (каталог combo/).
#
# ЗАЧЕМ. То же, что run_dewarp_validate.sh, но по НАХОДКАМ детектора, а не по восьми
# эталонным полосам: видно, как движки ведут себя на всём разнообразии кривизны пака.
#
# ВХОД — симлинки curved_lines_pack1_links/combo (их пишет run_curved_lines.sh); имя
# симлинка уже несёт год, выпуск, номер страницы PDF и score, оно и становится именем
# результата. Цель симлинка — оригинальный TIFF на /mnt/dump3.
# ВЫХОД — /mnt/system/raw/mts/pack1_dewarp/found/<движок>/, в 300 dpi: находок сотни,
# движков семь, и по 13 МБ на полосу в 600 dpi это была бы сотня гигабайт.
#
# ВРЕМЯ. При сотнях полос — часы, в основном pagedewarp; при необходимости сузить
# движки: --engines textline,docscanner. Ждать ПО PID (CLAUDE.md).
set -euo pipefail
source "$(dirname "$0")/common.sh"

set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

LINKS_DIR="curved_lines_pack1_links/combo"
OUT_DIR="/mnt/system/raw/mts/pack1_dewarp/found"

if [ ! -d "$LINKS_DIR" ]; then
    echo "Нет каталога $LINKS_DIR — сперва run_curved_lines.sh" >&2
    exit 1
fi

ARGS=(
    run
    --from-links "$LINKS_DIR"
    --out-dir "$OUT_DIR"
    --out-dpi 300
    --engines all
    --jobs 8
)

echo "Dewarp находок из $LINKS_DIR → $OUT_DIR"
uv run python -m ocr_utils.dewarp "${ARGS[@]}" "$@"
