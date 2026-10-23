#!/usr/bin/env bash
#
# Исправленные копии PDF по кэшу прогона run: удаление и усечение слов слоя, вставка
# прочитанного повёрнутого текста → $TEXT_LAYER_RUN_DIR/pdf/<выпуск>.pdf и fix.csv со сверкой.
# Исходные PDF не изменяются. Копия выпуска пишется целиком (все страницы), правятся только
# страницы из кэша.
#
# Читает: $TEXT_LAYER_PDF_DIR, $TEXT_LAYER_RUN_DIR/cache. Пишет: $TEXT_LAYER_RUN_DIR/{pdf/,fix.csv}.
# Идёт ~1–3 с на выпуск (копия ~15 МБ): 8 воркеров → пара минут на все выпуски выборки.

set -euo pipefail
set -m
# Выход: погасить всю группу (воркеры пула), сохранив код выхода; сигналы на время гашения игнорируются,
# иначе свой же TERM снова зовёт ловушку — рекурсия до переполнения стека bash (SIGSEGV, код 139).
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$SCRIPT_DIR/../.."

JOBS="${JOBS:-8}"

uv run python -m ocr_utils.text_layer_fix fix \
    --pdf-dir "$TEXT_LAYER_PDF_DIR" \
    --out-dir "$TEXT_LAYER_RUN_DIR" \
    --jobs "$JOBS" \
    "$@"
