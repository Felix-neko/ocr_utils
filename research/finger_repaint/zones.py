"""Оверлеи зон закраса пальца: старая дыра LaMa (асимметричная дилатация), новая дыра (круговая, варианты H, J) и окна LaMa новой дыры — на исходном кадре, целиком и с увеличением по окнам."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.inpainting.roi import roi_bounds_list
from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed, header_strip
from ocr_utils.scan_cropping.finger_removal.removal import FINGER_PADDING, FINGER_ROI_SCALE
from ocr_utils.scan_cropping.v2.scale import Scale
from research.border_fill.frames import key_of
from research.finger_repaint.cli import PACK6_PX_PER_MM, read_set
from research.finger_repaint.repaint import SYM_RADIUS_RATIO, Variant, load_input, symmetric_hole

log = logging.getLogger(__name__)

# Цвета BGR: старая дыра — красный контур; новая дыра — синяя полупрозрачная заливка с контуром; окно LaMa новой
# дыры — зелёная рамка.
COLOR_OLD_HOLE = (0, 0, 220)
COLOR_NEW_HOLE = (220, 90, 20)
COLOR_NEW_ROI = (0, 150, 0)
NEW_HOLE_ALPHA = 0.3
# Высота миниатюры кадра и вырезок окон на листе, px.
THUMB_H = 900
ZOOM_H = 900


def draw_zones(
    img: np.ndarray, old: np.ndarray, new: np.ndarray, rois: list[tuple[int, int, int, int]], k: float
) -> np.ndarray:
    """Кадр в масштабе ``k`` с зонами: новая дыра (заливка и контур), старая дыра (контур), окна (рамки).

    Args:
        img: Кадр BGR (полный размер).
        old: Старая дыра LaMa (bool, размер кадра).
        new: Новая дыра LaMa (bool, размер кадра).
        rois: Окна LaMa новой дыры (x1, y1, x2, y2) в px кадра.
        k: Масштаб вывода.

    Returns:
        Картинка BGR с разметкой.
    """
    size = (max(1, round(img.shape[1] * k)), max(1, round(img.shape[0] * k)))
    canvas = cv2.resize(img, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    small_new = cv2.resize(new.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    small_old = cv2.resize(old.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    # Заливка новой дыры — в слой и смешение, буквы под ней читаются.
    layer = canvas.copy()
    layer[small_new > 0] = COLOR_NEW_HOLE
    canvas = cv2.addWeighted(layer, NEW_HOLE_ALPHA, canvas, 1 - NEW_HOLE_ALPHA, 0)
    width = max(2, round(min(size) / 300))
    for mask, color in ((small_new, COLOR_NEW_HOLE), (small_old, COLOR_OLD_HOLE)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, color, width)
    for x1, y1, x2, y2 in rois:
        cv2.rectangle(
            canvas, (round(x1 * k), round(y1 * k)), (round(x2 * k) - 1, round(y2 * k) - 1), COLOR_NEW_ROI, width
        )
    return canvas


def frame_sheet(key: str, img: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Лист кадра: миниатюра целиком и по вырезке на каждое окно новой дыры, легенда и шапка в полях."""
    rois = roi_bounds_list(new.astype(np.uint8), FINGER_PADDING, FINGER_ROI_SCALE)
    parts = [draw_zones(img, old, new, rois, THUMB_H / img.shape[0])]
    for x1, y1, x2, y2 in rois:
        # Вырезка — окно с полем 10 % по сторонам, чтобы рамка окна была видна.
        mx, my = (x2 - x1) // 10, (y2 - y1) // 10
        bx1, by1 = max(0, x1 - mx), max(0, y1 - my)
        bx2, by2 = min(img.shape[1], x2 + mx), min(img.shape[0], y2 + my)
        shifted = [(a - bx1, b - by1, c - bx1, d - by1) for a, b, c, d in rois]
        crop = draw_zones(
            img[by1:by2, bx1:bx2], old[by1:by2, bx1:bx2], new[by1:by2, bx1:bx2], shifted, ZOOM_H / (by2 - by1)
        )
        parts += [np.full((ZOOM_H, 16, 3), 255, np.uint8), crop]
    height = max(p.shape[0] for p in parts)
    parts = [
        cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in parts
    ]
    old_mm2 = old.sum() / PACK6_PX_PER_MM**2
    new_mm2 = new.sum() / PACK6_PX_PER_MM**2
    header = [
        f"{key} — зоны закраса пальца на исходном кадре",
        f"старая дыра {old_mm2:.0f} мм², новая {new_mm2:.0f} мм², окон LaMa {len(rois)}; слева кадр целиком, "
        "справа окна с увеличением",
    ]
    legend = [
        LegendEntry("старая дыра LaMa (асимметричная дилатация, как сейчас; в H — зона вклейки)", COLOR_OLD_HOLE),
        LegendEntry(
            f"новая дыра LaMa (круг {SYM_RADIUS_RATIO[Variant.SYM_HOLE.value]:.0f}× припуска; в J — и зона вклейки)",
            COLOR_NEW_HOLE,
            alpha=NEW_HOLE_ALPHA,
            style=SampleStyle.BOX,
        ),
        LegendEntry(f"окно LaMa новой дыры (+{FINGER_PADDING} px, ×{FINGER_ROI_SCALE})", COLOR_NEW_ROI),
    ]
    return framed(np.hstack(parts), header, legend)


@click.command()
@click.option("--base", "base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Прогон base стенда (кэш с fingers.npz).")  # fmt: skip
@click.option("--src-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень входа (только чтение).")  # fmt: skip
@click.option("--set", "set_paths", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Файлы множеств.")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Куда листы.")
def main(base_dir: Path, src_root: Path, set_paths: tuple[Path, ...], out_dir: Path) -> None:
    """Оверлеи зон по кадрам множеств: ``<out-dir>/<множество>/NN_<ключ>.jpg`` (только CPU)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scale = Scale(PACK6_PX_PER_MM)
    for set_path in set_paths:
        out = out_dir / set_path.stem
        out.mkdir(parents=True, exist_ok=True)
        for i, rel in enumerate(read_set(set_path), 1):
            key = key_of(rel)
            inp = load_input(src_root / rel, base_dir / "cache" / key)
            if not inp.paint.any():
                continue
            new = symmetric_hole(inp, SYM_RADIUS_RATIO[Variant.SYM_HOLE.value], scale)
            sheet = frame_sheet(key, inp.original, inp.paint, new)
            cv2.imwrite(str(out / f"{i:02d}_{key}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            log.info("%s %d %s", set_path.stem, i, key)


if __name__ == "__main__":
    main()
