#!/bin/bash
# Целостность RAF после поворота (только чтение): файл на месте, тег Orientation, exiftool -validate
# (кроме врождённого у Fuji предупреждения VignettingParams), пиксели превью и блок сырых данных — как «до».
# Аргументы: план (путь\tтег\tmd5 пикселей превью\tmd5 сырых или -), выходной манифест md5 всего файла, число потоков.
plan="$1"; out="$2"; jobs="${3:-4}"; D="$(dirname "$0")"
check() {
  local f o pixm rawm err got v
  IFS=$'\t' read -r f o pixm rawm <<< "$1"; err=""
  if [ ! -f "$f" ]; then echo "НЕТ ФАЙЛА: $f"; return; fi
  got=$(exiftool -q -n -s3 -Orientation "$f"); [ "$got" = "$o" ] || err+=" тег=$got(ждали $o)"
  v=$(exiftool -q -q -validate -warning -error -a "$f" 2>&1 | grep -E '^(Warning|Error)' | grep -v 'VignettingParams overlaps')
  [ -z "$v" ] || err+=" validate:[$v]"
  [ "$(bash "$D/pixmd5.sh" "$f")" = "$pixm" ] || err+=" пиксели превью изменились"
  if [ "$rawm" != "-" ]; then [ "$(bash "$D/rawmd5.sh" "$f")" = "$rawm" ] || err+=" сырые данные изменились"; fi
  if [ -n "$err" ]; then echo "ПЛОХО: $f:$err"; else printf 'OK\t%s\t%s\n' "$(md5sum < "$f" | cut -d' ' -f1)" "$f"; fi
}
export -f check; export D
tr '\n' '\0' < "$plan" | xargs -0 -P"$jobs" -I{} bash -c 'check "$1"' _ {} > "$out.raw"
grep '^OK' "$out.raw" | cut -f2,3 > "$out"; grep -v '^OK' "$out.raw"
echo "проверено $(wc -l < "$out.raw"), в порядке $(wc -l < "$out")"
