#!/bin/bash
# md5 блока сырых данных сенсора RAF (FujiIFD StripOffsets/StripByteCounts, абсолютное смещение) — не меняется от записи метаданных.
f="$1"; read -r off len < <(exiftool -q -n -s3 -FujiIFD:StripOffsets -FujiIFD:StripByteCounts "$f" | tr '\n' ' ')
tail -c +$((off+1)) "$f" | head -c "$len" | md5sum | cut -d' ' -f1
