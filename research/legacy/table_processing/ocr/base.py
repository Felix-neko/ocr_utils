"""Общий контракт распознавателей: выпрямленная ячейка → строки текста.

ВХОД У ВСЕХ ОДИН И ТОТ ЖЕ — уже выпрямленная вырезка ячейки в 300 dpi. Поворот делается
снаружи и одинаково для всех движков: иначе сравнение мерило бы не качество распознавания,
а удачность чужой предобработки.

ВЫХОД — строки, а не строка. Боковой заголовок почти всегда набран в две-три строки
(«количество / в сутко-комп- / лекте»), и рендеру нужно знать, где были переносы: он их
всё равно переложит заново, но перенос по дефису склеивается без пробела, а обычный — с
пробелом.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

# Мусор по краям строки: одиночный знак без букв и цифр. Берётся он от обрубка линейки,
# который не убрали ни отступ, ни чистка вырезки: движок читает его как «|», «—», «©», «‹».
# Убирается только ПО КРАЯМ и только длиной до двух знаков — внутри строки такой знак может
# оказаться настоящим (тире в «1966—1970»).
JUNK_EDGE = re.compile(r"^[^\w]{1,2}(?=\s|$)|(?<=\s)[^\w]{1,2}$", re.UNICODE)

# Поля вокруг вырезки перед распознаванием: у всех движков точность на прижатом к краю
# тексте заметно ниже.
PAD_PX = 16

# Ниже этой стороны вырезка увеличивается вдвое: у tesseract точность падает, когда знак
# ниже 20 px, а петит шапки при 300 dpi даёт как раз около 20.
MIN_SIDE_PX = 300
UPSCALE = 2


@dataclass
class OcrResult:
    """Что прочитал движок в одной ячейке."""

    lines: list[str] = field(default_factory=list)
    confidence: float = 0.0
    seconds: float = 0.0
    note: str = ""

    @property
    def text(self) -> str:
        """Строки, склеенные в одну: перенос по дефису — без пробела, мусор по краям убран."""
        joined = ""
        for line in self.lines:
            piece = strip_junk(line)
            if not piece:
                continue
            if joined.endswith(("-", "‐", "‑")):
                joined = joined[:-1] + piece
            elif joined:
                joined += " " + piece
            else:
                joined = piece
        return joined


class Recognizer(Protocol):
    def __call__(self, images: Sequence[np.ndarray]) -> list[OcrResult]: ...


def _always() -> bool:
    return True


@dataclass(frozen=True)
class Engine:
    name: str
    summary: str
    stage: str  # "cpu" | "gpu"
    make: Callable[[], Recognizer]
    available: Callable[[], bool] = _always


def prepare(gray: np.ndarray) -> np.ndarray:
    """Общая для всех движков подготовка: поля цветом бумаги и увеличение мелкой вырезки."""
    image = gray
    if min(image.shape[:2]) < MIN_SIDE_PX:
        image = cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(image, PAD_PX, PAD_PX, PAD_PX, PAD_PX, cv2.BORDER_CONSTANT, value=255)


def strip_junk(line: str) -> str:
    """Убрать одиночные знаки без букв и цифр по краям строки."""
    previous = None
    current = line.strip()
    while current != previous:
        previous = current
        current = JUNK_EDGE.sub("", current).strip()
    return current
