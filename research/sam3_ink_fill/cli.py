"""CLI стенда: по кадрам из кэша scan_cropping v2 — SAM 3 по кадру без крупной краски, маски бумаги «было | стало», листы."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.scan_cropping.v2.glue_refine import REFINE_DEPTH
from ocr_utils.scan_cropping.v2.paper import load_sam3, sam3_pages
from ocr_utils.scan_cropping.v2.pipeline import paper_masks
from ocr_utils.scan_cropping.v2.ink_fill import fill_bold_ink, touching_ink

log = logging.getLogger(__name__)

# Цвета (BGR, палитра ``/draw-overlay``): бумага прежняя — серый справочный, новая — зелёный, закрашенная краска —
# оранжевый полупрозрачно.
COLOUR_OLD = (170, 170, 170)
COLOUR_NEW = (0, 150, 0)
COLOUR_INK = (0, 165, 255)
INK_ALPHA = 0.45
# Ширина листа, px.
SHEET_WIDTH = 2600


def contour(canvas: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], width: int) -> None:
    """Внешние контуры маски на локальном холсте.

    Args:
        canvas: Холст BGR (меняется на месте).
        mask: Булева маска.
        colour: Цвет.
        width: Толщина, px.
    """
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, cs, -1, colour, width)


def sheet(img: np.ndarray, filled: np.ndarray, ink: np.ndarray, old: np.ndarray, new: np.ndarray, title: str):
    """Лист: кадр с прежней и новой бумагой | кадр, поданный в SAM 3, с закрашенной краской.

    Args:
        img: Кадр сегментации.
        filled: Кадр без крупной краски.
        ink: Закрашенное.
        old: Итоговая бумага прежде.
        new: Итоговая бумага по SAM 3 без краски.
        title: Шапка.

    Returns:
        BGR.
    """
    left = img.copy()
    contour(left, old, COLOUR_OLD, 5)
    contour(left, new, COLOUR_NEW, 3)
    right = filled.copy()
    layer = right.copy()
    layer[ink] = COLOUR_INK
    cv2.addWeighted(layer, INK_ALPHA, right, 1 - INK_ALPHA, 0, right)
    pair = np.hstack([left, np.full((img.shape[0], 20, 3), 255, np.uint8), right])
    s = SHEET_WIDTH / pair.shape[1]
    pair = cv2.resize(pair, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    legend = [
        LegendEntry("итоговая бумага прежде (SAM 3 по кадру)", COLOUR_OLD),
        LegendEntry("итоговая бумага: прежняя ∪ SAM 3 по кадру без крупной краски", COLOUR_NEW),
        LegendEntry("закрашено Telea перед SAM 3 (крупная краска)", COLOUR_INK, alpha=INK_ALPHA),
    ]
    return framed(pair, [title, "слева — кадр с бумагой было | стало, справа — кадр, поданный в SAM 3"], legend)


@click.group()
def main() -> None:
    """Стенд «SAM 3 по кадру без крупной краски»."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--cache", "frames", type=click.Path(exists=True, file_okay=False, path_type=Path), multiple=True,
              required=True, help="Папка кадра в кэше scan_cropping v2 (seg.png, book.npz, sam3.npz); можно несколько.")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка выхода: листы, маски.")  # fmt: skip
def run(frames: tuple[Path, ...], out_dir: Path) -> None:
    """SAM 3 по кадру без крупной краски на кадрах из кэша v2 и листы «было | стало»."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = load_sam3()
    for work in frames:
        img = cv2.imread(str(work / "seg.png"))
        book = dict(np.load(work / "book.npz"))
        filled, ink = fill_bold_ink(img, book["book"], book["bg"], book["hulls"])
        raw, inst = sam3_pages(predictor, filled, book["hulls"])
        old = paper_masks(img, book, dict(np.load(work / "sam3.npz")), REFINE_DEPTH)
        alone = paper_masks(img, book, {"raw": raw, "inst": inst}, REFINE_DEPTH).paper
        # Итог — объединение: где SAM 3 и так взял лист, ничего не теряется; где его останавливала напечатанная
        # рамка, добавляется поле до края листа.
        new = old.paper | touching_ink(alone & ~old.paper, ink)
        np.savez_compressed(out_dir / f"{work.name}.npz", ink=ink, raw=raw, inst=inst, paper=new, alone=alone)
        gained = int((new & ~old.paper).sum())
        lost_alone = int((old.paper & ~alone).sum())
        title = (f"{work.name}: бумага ∪ SAM 3 без крупной краски — +{gained} px сегментации "
                 f"(сам по себе кадр без краски потерял бы {lost_alone} px)")  # fmt: skip
        cv2.imwrite(str(out_dir / f"{work.name}.jpg"), sheet(img, filled, ink, old.paper, new, title),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])  # fmt: skip
        log.info("%s: закрашено %d px, бумаги +%d (без объединения −%d)", work.name, int(ink.sum()), gained, lost_alone)
