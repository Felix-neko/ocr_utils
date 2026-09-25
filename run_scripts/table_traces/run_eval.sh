#!/usr/bin/env bash
#
# Трассы линеек и сетка по кривым на самых перекошенных таблицах пака-1: сравнение с сеткой по осям.
#
# Что делает: select — берёт из базы разметки (только чтение) таблицы с наибольшим
#   max(|skew_deg|, angle_spread) из detector_info (TOP штук) плюс известные трудные случаи
#   (1968/03 с.4, 1968/05 с.70, 1976/08 IMG_0067_1L — боковая, 1967/10 IMG_0033_1L);
#   run — вырезает каждую таблицу из TIFF 600 dpi (только чтение), приводит к 300 dpi, строит
#   сетку по осям (find_lines + grid_from_lines) и сетку по кривым (curved_table), пишет CSV по
#   таблицам и по линейкам и оверлеи «по осям | по кривым».
# Читает: $DB, $PACK_DIR (из common.sh пака).
# Пишет: $OUT_DIR/кандидаты.csv, таблицы.csv, линейки.csv, оверлеи/ (оверлеи вне git).
# Сколько идёт: ~1 с счёта на таблицу плюс чтение TIFF с NTFS-3G; 44 таблицы — около минуты.
# Числа: в паке |skew| > 2° только у 2 таблиц, > 1° — у 9, angle_spread > 1° — у 32 (замер по
#   detector_info 903 таблиц, 2026-09-25); TOP=40 накрывает всех с разбросом > 1°.
# JOBS=8, а не 16: задача упирается в чтение сканов с /mnt/dump3 (NTFS-3G).

set -euo pipefail
set -m
trap 'kill -- -$$ 2>/dev/null' EXIT INT TERM

source "$(dirname "${BASH_SOURCE[0]}")/../scan_markup/pack1/common.sh"

TOP=40
JOBS=8
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$REPO/reports/curved_table_grid"

cd "$REPO"
uv run python -m research.table_traces select --db "$DB" --top "$TOP" --out "$OUT_DIR/кандидаты.csv"
uv run python -m research.table_traces run --candidates "$OUT_DIR/кандидаты.csv" --pack-dir "$PACK_DIR" \
    --out-dir "$OUT_DIR" --jobs "$JOBS"
