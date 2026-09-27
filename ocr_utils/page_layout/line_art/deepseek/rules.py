"""Общие правила работы с выводом DeepSeek-OCR-2: вырезка кандидата, класс блока, отношение блока к области, выдумки, пятна краски.

Все рамки DeepSeek — в пикселях вырезки кандидата при ``CROP_DPI`` (с полем ``CROP_PAD_MM``);
в пиксели полосы их переводит :func:`crop_to_page`. Правила и пороги выведены на стенде
``research/line_art_titles`` (reports/line_art_titles.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.classes import ObjectClass

# Разрешение вырезок, которые видит DeepSeek: крупный заголовок при 600 dpi только замедляет модель.
CROP_DPI = 300

# Поле вокруг кандидата при вырезке, мм, со всех сторон. Широкое боковое поле (до 20 мм) модель
# читала соседней колонкой и зацикливалась на ней (1968/03 с.134).
CROP_PAD_MM = 2.0

# Класс объекта по метке блока DeepSeek.
CLASS_BY_LABEL = {"image": ObjectClass.DRAWING, "table": ObjectClass.TABLE, "equation": ObjectClass.FORMULA}

# Разметка формулы в тексте DeepSeek: LaTeX (``\(``, ``\[``, ``\frac`` …) или ``$…$``. Треть настоящих
# формул модель отдаёт блоком ``text`` с такой разметкой внутри (просмотр 1039 формул surya).
MATH = re.compile(r"\\\(|\\\[|\$\$|\$[^$\n]+\$|\\frac|\\sum|\^\{|_\{")

# Иероглифы CJK. Одиночный иероглиф бывает и в настоящей формуле (модель так читает «тыс.»,
# «руб.»: «2485,7 此 c. py6.»), выдуманный же текст — связная китайская фраза. Порог — от
# HALLUCINATION_MIN иероглифов в блоке (у выдумок на паке-1 — от 5, у настоящих формул — 1–2).
HALLUCINATION = re.compile(r"[一-鿿]")
HALLUCINATION_MIN = 3

# Блок относится к области, если их пересечение не меньше этой доли МЕНЬШЕЙ из двух рамок.
BLOCK_ON_REGION_SHARE = 0.5

# Буквоподобное пятно: высота в этих пределах, мм, стороны не дальше LETTER_MAX_ASPECT от 1:1.
LETTER_HEIGHT_MM = (1.5, 30.0)
LETTER_MAX_ASPECT = 4.0

# Надпись не бывает выше этого, мм: у рубрик-вензелей пака-1 95-й перцентиль 30 мм, у схем
# 5-й — 31 мм (reports/line_art_titles.md).
TITLE_MAX_HEIGHT_MM = 40.0


@dataclass(frozen=True)
class Crop:
    """Где вырезка кандидата лежит на полосе.

    Attributes:
        box: Рамка кандидата в пикселях полосы.
        dpi: Разрешение полосы.
        inner: Рамка кандидата в пикселях вырезки ``(x0, y0, x1, y1)``.
        size: Размер вырезки ``(ширина, высота)``.
    """

    box: Box
    dpi: int
    inner: tuple[int, int, int, int]
    size: tuple[int, int]

    def to_json(self) -> dict:
        """Словарь для JSON."""
        return {"box": list(self.box.as_tuple()), "dpi": self.dpi, "inner": list(self.inner), "size": list(self.size)}

    @staticmethod
    def from_json(payload: dict) -> "Crop":
        """Обратно из :meth:`to_json`."""
        return Crop(Box(*payload["box"]), int(payload["dpi"]), tuple(payload["inner"]), tuple(payload["size"]))


def crop_region(gray: np.ndarray, box: Box, dpi: int, pad_mm: float = CROP_PAD_MM) -> tuple[np.ndarray, Crop]:
    """Вырезать кандидата с полем и ужать до ``CROP_DPI``.

    Args:
        gray: Серая полоса в родном разрешении.
        box: Рамка кандидата в пикселях полосы.
        dpi: Разрешение полосы.
        pad_mm: Поле со всех сторон, мм.

    Returns:
        Пара (серая вырезка при ``CROP_DPI``, :class:`Crop` — где она на полосе).
    """
    height, width = gray.shape[:2]
    pad = int(round(pad_mm / 25.4 * dpi))
    padded = box.padded(pad).clipped(width, height)
    scale = CROP_DPI / dpi
    crop = cv2.resize(gray[padded.slice], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    inner = (
        int(round((box.x0 - padded.x0) * scale)),
        int(round((box.y0 - padded.y0) * scale)),
        int(round((box.x1 - padded.x0) * scale)),
        int(round((box.y1 - padded.y0) * scale)),
    )
    return crop, Crop(box, dpi, inner, (crop.shape[1], crop.shape[0]))


def crop_to_page(element: dict, crop: Crop) -> Box:
    """Рамка в пикселях вырезки → пиксели полосы.

    Args:
        element: Словарь с ``x0 … y1`` в пикселях вырезки.
        crop: Где вырезка лежит на полосе.

    Returns:
        Рамка в пикселях полосы.
    """
    scale = crop.dpi / CROP_DPI
    ox = crop.box.x0 - crop.inner[0] * scale
    oy = crop.box.y0 - crop.inner[1] * scale
    return Box(
        int(round(ox + element["x0"] * scale)),
        int(round(oy + element["y0"] * scale)),
        int(round(ox + element["x1"] * scale)),
        int(round(oy + element["y1"] * scale)),
    )


def is_hallucinated(text: str) -> bool:
    """Выдуман ли текст моделью: в нём не меньше ``HALLUCINATION_MIN`` китайских иероглифов."""
    return len(HALLUCINATION.findall(text)) >= HALLUCINATION_MIN


def block_class(block: dict) -> ObjectClass | None:
    """Класс объекта по блоку DeepSeek: рисунок, таблица, формула (``equation`` или текст с разметкой формулы) или ``None``.

    Args:
        block: Блок DeepSeek (``label``, ``text``).

    Returns:
        :class:`ObjectClass` или ``None`` для текстового блока без формулы.
    """
    if block["label"] in CLASS_BY_LABEL:
        return CLASS_BY_LABEL[block["label"]]
    text = block.get("text", "")
    # Китайский текст на полосе советского журнала — выдумка модели на нечитаемом заголовке
    # («已知 \( f(x) = x^{2} … \)», 1966/05 с.20): LaTeX в нём формулой не считается.
    if MATH.search(text) and not is_hallucinated(text):
        return ObjectClass.FORMULA
    return None


def on_region(block: dict, inner) -> bool:
    """Относится ли блок к области: пересечение не меньше ``BLOCK_ON_REGION_SHARE`` МЕНЬШЕЙ из двух рамок.

    Симметрично, как согласие рамок в детекторе: мелкий блок в поле вокруг области не относится к
    ней, а блок больше области, накрывший её целиком, относится (1966/02 с.93).

    Args:
        block: Рамка ``x0, y0, x1, y1`` в пикселях вырезки.
        inner: Рамка области в вырезке ``(x0, y0, x1, y1)``.

    Returns:
        ``True``, если блок относится к области.
    """
    width = min(block["x1"], inner[2]) - max(block["x0"], inner[0])
    height = min(block["y1"], inner[3]) - max(block["y0"], inner[1])
    if width <= 0 or height <= 0:
        return False
    block_area = max(1, (block["x1"] - block["x0"]) * (block["y1"] - block["y0"]))
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return width * height >= BLOCK_ON_REGION_SHARE * min(block_area, inner_area)


@dataclass(frozen=True)
class Blobs:
    """Краска области и её связные пятна.

    Attributes:
        ink: Маска краски вырезки (1 — краска), обнулённая вне рамки области.
        areas: Площади пятен, px; индекс ``i`` — пятно с меткой ``i + 1``.
        boxes: Рамки пятен ``(x0, y0, x1, y1)`` в пикселях вырезки.
        letter_like: Буквоподобно ли пятно.
        largest: Индекс самого крупного пятна или ``-1``.
    """

    ink: np.ndarray
    areas: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    letter_like: np.ndarray
    largest: int

    @property
    def total(self) -> int:
        """Площадь краски области, px (не меньше 1)."""
        return max(1, int(self.ink.sum()))


def ink_blobs(gray: np.ndarray, inner, dpi: int = CROP_DPI) -> Blobs:
    """Краска внутри рамки области (порог Оцу по вырезке) и её связные пятна с признаком буквоподобия.

    Args:
        gray: Серая вырезка (с полем).
        inner: Рамка области в вырезке.
        dpi: Разрешение вырезки.

    Returns:
        :class:`Blobs`.
    """
    mm = dpi / 25.4
    x0, y0, x1, y1 = inner
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = np.zeros_like(mask)
    ink[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    body = stats[1:] if count > 1 else np.zeros((0, 5), np.int32)
    areas = body[:, cv2.CC_STAT_AREA]
    widths, heights = body[:, cv2.CC_STAT_WIDTH], body[:, cv2.CC_STAT_HEIGHT]
    letter_like = (
        (heights >= LETTER_HEIGHT_MM[0] * mm)
        & (heights <= LETTER_HEIGHT_MM[1] * mm)
        & (np.maximum(widths, heights) <= LETTER_MAX_ASPECT * np.maximum(1, np.minimum(widths, heights)))
    )
    boxes = [(int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]), int(s[0] + s[2]), int(s[1] + s[3])) for s in body]
    largest = int(np.argmax(areas)) if areas.size else -1
    return Blobs(ink, areas, boxes, letter_like, largest)


def text_block_ink_share(blobs: Blobs, blocks: list[dict]) -> float:
    """Доля краски области под текстовыми блоками DeepSeek (не рисунок, не таблица, не формула).

    Args:
        blobs: Краска области.
        blocks: Блоки DeepSeek ``markdown``.

    Returns:
        Доля 0..1.
    """
    covered = np.zeros_like(blobs.ink)
    for block in blocks:
        if block_class(block) is None:
            covered[max(0, block["y0"]) : block["y1"], max(0, block["x0"]) : block["x1"]] = 1
    return float((blobs.ink * covered).sum()) / blobs.total


def is_title_like(blobs: Blobs, crop: Crop, blocks: list[dict]) -> bool:
    """Правило «надпись»: только текстовые блоки над краской и невысокая область.

    Args:
        blobs: Краска области.
        crop: Где вырезка на полосе (высота области — из рамки кандидата).
        blocks: Блоки DeepSeek ``markdown`` области.

    Returns:
        ``True``, если по правилу это надпись.
    """
    height_mm = crop.box.height / crop.dpi * 25.4
    no_objects = not any(block_class(b) is not None and on_region(b, crop.inner) for b in blocks)
    return no_objects and text_block_ink_share(blobs, blocks) > 0 and height_mm <= TITLE_MAX_HEIGHT_MM


__all__ = [
    "BLOCK_ON_REGION_SHARE",
    "Blobs",
    "CLASS_BY_LABEL",
    "CROP_DPI",
    "CROP_PAD_MM",
    "Crop",
    "MATH",
    "TITLE_MAX_HEIGHT_MM",
    "block_class",
    "crop_region",
    "crop_to_page",
    "ink_blobs",
    "is_hallucinated",
    "is_title_like",
    "on_region",
    "text_block_ink_share",
]
