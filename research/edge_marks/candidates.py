"""Кандидаты тестового множества: краска за выровненной стороной у размеченного конца строки — рамка, вырезки трёх масштабов (компонент, строка, полоса) из бинаризованного PDF."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
import pandas as pd

from ocr_utils.geometry_regression.render import render_gray
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from research.text_block_specks.pages import NOGEO_DIR
from research.text_block_specks.sheets import side_x

# Во сколько раз рендер крупнее рабочей копии (координаты ``ends.csv`` — в рабочей копии).
K = RENDER_DPI / WORK_DPI
PX_PER_MM = RENDER_DPI / 25.4
# Полоса строки по высоте: середина строки ± столько x-высот.
BAND_XH = 0.9
# Полоса поиска краски по ширине: от стороны на столько мм внутрь (захват края буквы у дефиса) и
# на столько наружу за конец строки.
INSIDE_MM, OUTSIDE_MM = 0.3, 1.0
# Вырезка (a): рамка кандидата с полем в столько x-высот, увеличение и белое поле.
CROP_A_XH = 1.5
UPSCALE = 2
MARGIN_PX = 16
# Вырезка (b): строка — от конца на столько мм внутрь текста и на столько наружу.
LINE_IN_MM, LINE_OUT_MM = 60.0, 8.0
# Поле закраски кандидата, пиксели рендера.
ERASE_PAD = 2
# Класс разметки → истина для судей: сор или пометка — ``junk``, настоящий знак или вёрстка — ``sign``.
JUNK = {"S", "M", "T"}
SIGN = {"L", "P"}


@dataclass
class Candidate:
    """Кандидат: краска за стороной у конца строки.

    Attributes:
        id: Номер конца строки в разметке (``e…``, ``n…``).
        key: Полоса.
        pdf, page: Бинаризованный PDF и страница (с единицы).
        side: ``left`` или ``right``.
        label: Класс разметки (S, M, T, L, P).
        truth: ``junk`` или ``sign``.
        box: Рамка краски за стороной ``(x0, y0, x1, y1)``, пиксели рендера ``RENDER_DPI``.
        line_box: Рамка строки для вырезки (b), те же пиксели.
        end_x, row_y, side_x: Конец строки, середина строки и сторона блока, пиксели рендера.
        x_h: x-высота строки, пиксели рендера.
    """

    id: str
    key: str
    pdf: str
    page: int
    side: str
    label: str
    truth: str
    box: tuple[int, int, int, int]
    line_box: tuple[int, int, int, int]
    end_x: float
    row_y: float
    side_x: float
    x_h: float


def outermost_ink(
    binary: np.ndarray, end_x: float, row_y: float, sx: float, x_h: float, right: bool
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Крайняя наружу компонента краски строки за стороной блока — то, что и сделало выступ конца строки.

    Берётся не вся краска за стороной, а одна крайняя компонента (с правой стороны — с самым правым краем) и
    компоненты, лежащие над или под ней (перекрытие по x: точки «ё», части «:» и «;»). Иначе висячий дефис и
    соринка за ним попадали в одну рамку, и судьи видели в ней знак.

    Args:
        binary: Краска рендера (``True`` — краска).
        end_x, row_y, sx: Конец строки, середина строки, сторона (пиксели рендера).
        x_h: x-высота.
        right: Правая сторона.

    Returns:
        ``(рамка (x0, y0, x1, y1), маска краски кандидата в рамке)`` или ``None`` — краски за стороной нет.
    """
    # Окно поиска: полоса строки по высоте с запасом (компонента, задевшая полосу, берётся целиком) и
    # от стороны с краем ``INSIDE_MM`` внутрь до конца строки и ``OUTSIDE_MM`` дальше.
    wy0, wy1 = max(0, int(row_y - 2 * BAND_XH * x_h)), int(row_y + 2 * BAND_XH * x_h) + 1
    if right:
        wx0, wx1 = int(sx - INSIDE_MM * PX_PER_MM), int(max(end_x, sx) + OUTSIDE_MM * PX_PER_MM) + 1
    else:
        wx0, wx1 = int(min(end_x, sx) - OUTSIDE_MM * PX_PER_MM), int(sx + INSIDE_MM * PX_PER_MM) + 1
    wx0 = max(0, wx0)
    window = binary[wy0:wy1, wx0:wx1].astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(window, 8)
    # Полоса строки в координатах окна: компонента должна её задевать.
    by0, by1 = int(row_y - BAND_XH * x_h) - wy0, int(row_y + BAND_XH * x_h) + 1 - wy0
    usable = [i for i in range(1, count) if stats[i, 1] < by1 and stats[i, 1] + stats[i, 3] > by0]
    if not usable:
        return None
    # Крайняя наружу: у правой стороны — с самым правым краем, у левой — с самым левым.
    if right:
        outer = max(usable, key=lambda i: stats[i, 0] + stats[i, 2])
    else:
        outer = min(usable, key=lambda i: stats[i, 0])
    ox0, ox1 = stats[outer, 0], stats[outer, 0] + stats[outer, 2]
    # Добавить компоненты над и под крайней (перекрытие по x).
    chosen = [i for i in usable if stats[i, 0] < ox1 and stats[i, 0] + stats[i, 2] > ox0]
    mask = np.isin(labels, chosen)
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    return (wx0 + x0, wy0 + y0, wx0 + x1, wy0 + y1), mask[y0:y1, x0:x1]


def crop(gray: np.ndarray, box: tuple[int, int, int, int], pad: int, upscale: int = 1) -> np.ndarray:
    """Вырезка рамки с полем ``pad`` (вне страницы — белое), увеличенная в ``upscale`` раз и с белой каймой."""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    out = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(gray.shape[1], x1), min(gray.shape[0], y1)
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    if upscale > 1:
        out = cv2.resize(out, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(out, MARGIN_PX, MARGIN_PX, MARGIN_PX, MARGIN_PX, cv2.BORDER_CONSTANT, value=255)


def build(labels_csv: Path, ends_csv: Path, out_dir: Path, pdf_dir: Path = NOGEO_DIR) -> list[Candidate]:
    """Кандидаты по размеченным концам строк: рамки, вырезки ``a/``, ``b/``, рендеры полос ``pages/`` и ``candidates.json``.

    Args:
        labels_csv: Разметка концов (``sets/ends_labelled.csv``).
        ends_csv: ``ends.csv`` прошлого стенда (x-высота строки).
        out_dir: Куда писать.
        pdf_dir: Бинаризованные PDF.

    Returns:
        Кандидаты (концы с классом X и без краски за стороной пропускаются).
    """
    labels = pd.read_csv(labels_csv)
    labels = labels[labels.label.isin(JUNK | SIGN)]
    ends = pd.read_csv(ends_csv, usecols=["key", "block", "side", "row", "x_h_mm"])
    labels = labels.merge(ends, on=["key", "block", "side", "row"], how="left")
    for folder in ("a", "b", "b_erased", "pages"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    out: list[Candidate] = []
    for (pdf, page), group in labels.groupby(["pdf", "page"]):
        with fitz.open(pdf_dir / pdf) as document:
            gray = render_gray(document, int(page) - 1)
        key = group.key.iloc[0]
        cv2.imwrite(str(out_dir / "pages" / f"{key}.png"), gray)
        binary = gray < 128
        for record in group.to_dict("records"):
            right = record["side"] == "right"
            x_h = float(record["x_h_mm"]) * PX_PER_MM
            end_x, row_y = float(record["x"]) * K, float(record["y"]) * K
            sx = side_x(float(record["x"]), float(record["resid_mm"]), record["side"]) * K
            found = outermost_ink(binary, end_x, row_y, sx, x_h, right)
            if found is None:
                continue
            box, mask = found
            line_in, line_out = LINE_IN_MM * PX_PER_MM, LINE_OUT_MM * PX_PER_MM
            lx0 = int(end_x - line_in) if right else int(end_x - line_out)
            lx1 = int(end_x + line_out) if right else int(end_x + line_in)
            line_box = (max(0, lx0), int(row_y - 1.3 * x_h), min(gray.shape[1], lx1), int(row_y + 1.3 * x_h))
            candidate = Candidate(
                id=record["id"],
                key=key,
                pdf=pdf,
                page=int(page),
                side=record["side"],
                label=record["label"],
                truth="junk" if record["label"] in JUNK else "sign",
                box=box,
                line_box=line_box,
                end_x=end_x,
                row_y=row_y,
                side_x=sx,
                x_h=x_h,
            )
            cv2.imwrite(str(out_dir / "a" / f"{candidate.id}.png"), crop(gray, box, int(CROP_A_XH * x_h), UPSCALE))
            cv2.imwrite(str(out_dir / "b" / f"{candidate.id}.png"), crop(gray, line_box, 0))
            # Та же строка с закрашенной краской кандидата (для разностных судей: что кандидат добавил к тексту):
            # закрашиваются только пиксели кандидата с каймой ``ERASE_PAD``, соседняя буква остаётся целой.
            patch = np.zeros(gray.shape, dtype=np.uint8)
            patch[box[1] : box[3], box[0] : box[2]] = mask
            patch = cv2.dilate(patch, np.ones((2 * ERASE_PAD + 1, 2 * ERASE_PAD + 1), np.uint8))
            erased = gray.copy()
            erased[patch > 0] = 255
            cv2.imwrite(str(out_dir / "b_erased" / f"{candidate.id}.png"), crop(erased, line_box, 0))
            out.append(candidate)
    (out_dir / "candidates.json").write_text(json.dumps([asdict(c) for c in out], ensure_ascii=False, indent=0))
    return out


def load(out_dir: Path) -> list[Candidate]:
    """Кандидаты из ``candidates.json``."""
    return [
        Candidate(**{**item, "box": tuple(item["box"]), "line_box": tuple(item["line_box"])})
        for item in json.loads((out_dir / "candidates.json").read_text())
    ]


__all__ = ["Candidate", "build", "crop", "load", "outermost_ink"]
