#!/usr/bin/env bash
#
# Зона копирования по ВСЕМ кадрам двух папок ЭГ (1970/07-12 — 625 RAF, 1969/04-06 — 315 RAF): боевой вариант стенда.
# Крышки и форзацы (1_0010, DSCF0001, DSCF0002, DSCF0342) — skip=1 (frames.NOT_PAGES).
#
# Шаги и ожидаемое время на RTX 5060 Ti (по выборке 57 кадров):
#   превью уже в кэше; пальцы YOLO-World → SAM-B → LaMa ~2.1 с/кадр (GPU, последовательно, готовые пропускаются);
#   SAM2.1 Hiera-L бокс+точки ~0.43 с/кадр (GPU); отрезка за складкой + оверлеи — пул 14; выход под ScanTailor — пул 12.
# Выход: /mnt/hotstore/scan_processing/newspaper_crop/runs/{sam2_box_nofinger_all,spine_cut_all}.
set -euo pipefail
cd "$(dirname "$0")/../.."

run() { uv run python -m research.newspaper_crop --set all "$@"; }

run select-all
run previews
run fingers
run run --engine sam2_box --source nofinger --tag nofinger_all
run spine-cut --run sam2_box_nofinger_all --name spine_cut_all
# Зазор у своей печати: 10 px рабочего масштаба ≈ 2 мм (медианная высота полосы 2153 px при ~430 мм).
run guard-eval --run spine_cut_all --guard-px 10
run copy-out --run spine_cut_all --guard
# Сырьё для эталона наклона сторон (CVAT: tilt-export / tilt-import).
run tilt-candidates --run spine_cut_all
run sheets --run spine_cut_all --per-sheet 24 --cols 8 --height 420
