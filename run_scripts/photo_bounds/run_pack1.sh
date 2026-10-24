#!/usr/bin/env bash
#
# Стенд research/photo_bounds на паке-1 МТС: точная рамка и чёткий край растровых фото против ручной разметки CVAT.
#
# Читает: DB_REVIEWED (pack1_reviewed.sqlite, ТОЛЬКО ЧТЕНИЕ — sqlite URI mode=ro, без open_db), оригиналы TIFF пака-1
# (Я.Диск, только чтение, один раз на полосу), кэш surya LAYOUT_CACHE_DIR (вариант scan, только чтение).
# Пишет: $PHOTO_BOUNDS_ROOT (SSD hotstore).
#
# Набор: 404 фото на 378 полосах (виды grayscale/color без full_page; областей ≥ 0.80 кадра нет — максимум 0.73).
# Время (2026-10-24): cache ~2 мин (6 воркеров, упор в NTFS-3G), run ~20 с, sheets ~1 мин, text-count ~13 мин
# (разбор строк 2×30 с на полосу).
# Числа метода — в research/photo_bounds/bounds.py (мм бумаги): размытие растра σ 0.35 мм, ядро снимка — тон ниже
# бумаги на 30 или разброс > 12, рост по светлому растру (местный порог бумаги, полоска 0.25 мм с покрытием 0.7),
# профиль края 5 мм внутрь / 3 мм наружу через 1 мм, сторона — многочлен 2-й степени (Хьюбер 0.15 мм, отсев 0.6 мм);
# сторона чёткая: покрытие ≥ 0.6, остаток p90 ≤ 0.3 мм, ширина перехода ≤ 1.4 мм (пороги — по разметке глазами
# sets/photo_classes.csv, отчёт reports/photo_bounds.md).
#
set -euo pipefail

set -m
trap 'rc=$?; trap - EXIT; trap "" INT TERM; kill -- -$$ 2>/dev/null || true; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$(dirname "$0")/../.."
source run_scripts/photo_bounds/common.sh

# 12 по умолчанию: CPU-задачи стенда (16 ядер, часть — под прогоны рядом); кэш — 6 (упор в чтение NTFS-3G).
JOBS="${JOBS:-12}"
CACHE_JOBS="${CACHE_JOBS:-6}"
RUN="${RUN:-run_v14}"

uv run python -m research.photo_bounds select --db "$DB_REVIEWED" --root "$PHOTO_BOUNDS_ROOT"
uv run python -m research.photo_bounds cache --root "$PHOTO_BOUNDS_ROOT" --surya-cache "$LAYOUT_CACHE_DIR" --jobs "$CACHE_JOBS"
uv run python -m research.photo_bounds run --root "$PHOTO_BOUNDS_ROOT" --out "$PHOTO_BOUNDS_ROOT/$RUN" --jobs "$JOBS"
uv run python -m research.photo_bounds sheets --root "$PHOTO_BOUNDS_ROOT" --run-dir "$PHOTO_BOUNDS_ROOT/$RUN" \
    --kind all --per-sheet 16 --cols 4 --no-zoom --max-width 2400
uv run python -m research.photo_bounds sheets --root "$PHOTO_BOUNDS_ROOT" --run-dir "$PHOTO_BOUNDS_ROOT/$RUN" \
    --kind class --kind disagree --kind new_cut --kind new_spare --per-sheet 6 --cols 3
uv run python -m research.photo_bounds verify --root "$PHOTO_BOUNDS_ROOT" --run-dir "$PHOTO_BOUNDS_ROOT/$RUN" --photos 30
uv run python -m research.photo_bounds text-count --root "$PHOTO_BOUNDS_ROOT" --run-dir "$PHOTO_BOUNDS_ROOT/$RUN" --jobs "$JOBS"
uv run python -m research.photo_bounds text-sheets --root "$PHOTO_BOUNDS_ROOT" --run-dir "$PHOTO_BOUNDS_ROOT/$RUN" --top 8
