#!/usr/bin/env bash
#
# Стенд «зона копирования на фотосканах газет» (research/newspaper_crop) на выборке ЭГ 1969/04-06 и 1970/07-12.
#
# Числа прогона 2026-10-07 (RTX 5060 Ti, отчёт reports/newspaper_crop_zone.md):
#   выборка 60 кадров (по 25 равномерно + 13 трудных), skip=1 у крышки и форзацев → 57 в работе;
#   превью RAF 2944×4416 (встроенный JPEG), рабочий масштаб маски 0.5 (~120 dpi);
#   пальцы (YOLO-World → SAM-B → LaMa) 2.1 с/кадр; SAM2.1 Hiera-L бокс+точки 0.43 с/кадр, 1.45 ГБ VRAM;
#   лучший — sam2_box на nofinger: 0 провалов, дорисовка медиана 0.021, отсмотр глазами 55 ок + 2 «торец нижних листов».
# Выход: /mnt/hotstore/scan_processing/newspaper_crop (никогда не Я.Диск — исходники там только читаются).
set -euo pipefail
cd "$(dirname "$0")/../.."

run() { uv run python -m research.newspaper_crop "$@"; }

# Превью всех RAF двух папок (для отбора трудных кадров глазами); 8 воркеров — упор в NTFS-3G, не в счёт.
run previews --all-frames --jobs 8
for g in 1970_07-12 1969_04-06; do run contact-sheet --group "$g"; done

# Выборка: равномерно плюс трудные кадры (пометки skip/note в sets/sample.csv сохраняются).
EXTRA=()
for e in 1969_04-06/DSCF0018 1969_04-06/DSCF0289 1969_04-06/DSCF0329 1969_04-06/DSCF0213 1969_04-06/DSCF0016 \
         1969_04-06/DSCF0202_1 1970_07-12/2_2880 1970_07-12/2_2890 1970_07-12/2_3350 1970_07-12/1_2610 \
         1970_07-12/2_3190 1970_07-12/1_0960 1970_07-12/2_3310; do
    EXTRA+=(--extra "$e")
done
run select --per-group 25 "${EXTRA[@]}"
run previews

# Пальцы до сегментации.
run fingers

# Движки по превью как есть и по кадрам без пальцев (SAM 3 — веса finger_models/sam3/sam3.pt).
for e in sam2_points sam2_box sam3_text birefnet color dino_kmeans dino_sam2 scan_cropping; do
    run run --engine "$e"
done
for e in sam2_box sam3_text; do
    run run --engine "$e" --source nofinger --tag nofinger
done
run run --engine ink_hull --source nofinger
run vote --runs sam2_box_nofinger --runs sam3_text_nofinger --runs scan_cropping --name vote3

# Листы для отсмотра и выход под ScanTailor лучшего варианта.
for r in sam2_points sam2_box sam2_box_nofinger sam3_text sam3_text_nofinger birefnet color dino_kmeans dino_sam2 \
         scan_cropping ink_hull vote3; do
    run sheets --run "$r" || true
done
run corner-sheets --run sam2_box_nofinger
run spine-cut --run sam2_box_nofinger --name spine_cut
# Итоговые прямоугольники — после отрезки за складкой (runs/spine_cut/copy; в sam2_box_nofinger/copy — ссылка на них).
run copy-out --run spine_cut
run zone-overlays --run sam2_box_nofinger
