#!/usr/bin/env bash
# PostToolUse-хук на Edit/Write: форматирует изменённый .py и обновляет карту модулей.
#
# Из stdin приходит JSON вызова; нужен только tool_input.file_path. Файлы не из проекта и не
# .py пропускаются молча. Карта docs/modules.md перегенерируется, если правился модуль пакета —
# так она не протухает и её не надо помнить обновлять.

set -u
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

file=$(python3 -c 'import json,sys; print((json.load(sys.stdin).get("tool_input") or {}).get("file_path") or "")' 2>/dev/null)
[ -n "$file" ] || exit 0
case "$file" in
    *.py) ;;
    *) exit 0 ;;
esac
[ -f "$file" ] || exit 0
case "$file" in
    "$root"/*) ;;
    *) exit 0 ;;
esac

cd "$root" || exit 0
uv run --no-sync black -l 120 -C -q "$file" 2>/dev/null

case "$file" in
    "$root"/ocr_utils/*|"$root"/research/*)
        uv run --no-sync python scripts/gen_module_map.py >/dev/null 2>&1
        ;;
esac
exit 0
