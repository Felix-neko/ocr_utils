#!/bin/bash
# md5 пикселей встроенного превью (из RAF или из уже извлечённого jpg): метаданные JPEG не учитываются.
case "$1" in *.jpg) convert "$1" rgb:- | md5sum | cut -d' ' -f1;; *) exiftool -q -b -PreviewImage "$1" | convert - rgb:- | md5sum | cut -d' ' -f1;; esac
