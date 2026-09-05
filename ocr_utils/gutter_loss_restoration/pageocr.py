"""Чтение полос разворота через surya: текст строк и боксы слов.

ЗАЧЕМ ЭТО ЗДЕСЬ. Восстановить срезанный корешком хвост строки можно только зная, какое
слово там стояло. Слово вычисляется из переноса — «начало на этой строке, конец на
следующей», — а для этого нужен текст обеих строк. Читает его surya локально на GPU:
ни одного токена модели это не стоит, а качество на бумаге 1926 года достаточное.

ПОЧЕМУ ПОЛОСЫ ЧИТАЮТСЯ ПОРОЗНЬ. Если отдать распознавателю разворот целиком, строки
левой и правой полос сливаются в одну: у тугого переплёта между ними нет и десятка
пикселей пробела. Поэтому кадр режется по сгибу, и каждая полоса читается сама по себе.

ПОЧЕМУ БОКСЫ СЛОВ ВАЖНЕЕ ТЕКСТА. Распознаватель на обрезанном слове ошибается на
символ туда-сюда: «развернут» он читает как «разверну». Опираться на это при вклейке
букв нельзя. Зато НАЧАЛО последнего слова он даёт точно — и дальше слово переверстывается
целиком от этой точки, а не дописывается с угаданного места.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Поля вокруг полосы при вырезке под распознавание, в долях ширины полосы.
CROP_PAD = 0.01

# Сколько строк распознавать за раз. Без ограничения плотная табличная полоса (три
# сотни строк) укладывает видеопамять: замерено, 15 ГБ кончаются на середине кадра.
BATCH = 48
BATCH_MIN = 8


@dataclass(frozen=True)
class Word:
    """Слово строки.

    Attributes:
        text: Текст слова.
        x0: Левый край в координатах исходного кадра.
        x1: Правый край.
    """

    text: str
    x0: int
    x1: int


@dataclass(frozen=True)
class Line:
    """Распознанная строка полосы.

    Attributes:
        text: Текст строки.
        top: Верх строки в координатах исходного кадра.
        bottom: Низ строки.
        x0: Левый край.
        x1: Правый край.
        words: Слова строки слева направо.
    """

    text: str
    top: int
    bottom: int
    x0: int
    x1: int
    words: tuple[Word, ...]

    @property
    def head(self) -> str:
        """Первое слово строки без знаков препинания по краям."""
        return self.words[0].text if self.words else ""

    @property
    def tail(self) -> str:
        """Последнее слово строки."""
        return self.words[-1].text if self.words else ""


@lru_cache(maxsize=1)
def _predictor():
    """Готовит распознаватель surya (один раз на процесс)."""
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation = FoundationPredictor()
    return RecognitionPredictor(foundation), DetectionPredictor()


def _recognize(recognizer, detector, crops, batch: int = BATCH):
    """Распознаёт полосы, уменьшая партию, если не хватило видеопамяти."""
    import torch

    while True:
        try:
            return recognizer(
                crops, det_predictor=detector, math_mode=False, return_words=True, recognition_batch_size=batch
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= BATCH_MIN:
                raise
            batch = max(BATCH_MIN, batch // 2)


def read_halves(path: Path, fold: int) -> dict[str, list[Line]]:
    """Читает обе полосы разворота.

    Args:
        path: Путь к кадру.
        fold: Столбец сгиба в координатах исходного кадра.

    Returns:
        Словарь {"L": строки левой полосы, "R": строки правой}; координаты — в кадре.
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size
    crops, offsets, sides = [], [], []
    for side, (x0, x1) in (("L", (0, fold)), ("R", (fold, width))):
        pad = int((x1 - x0) * CROP_PAD)
        a, b = max(0, x0 - pad if side == "R" else x0), min(width, x1 + pad if side == "L" else x1)
        if b - a < 200:
            continue
        crops.append(image.crop((a, 0, b, height)))
        offsets.append(a)
        sides.append(side)
    if not crops:
        return {}

    recognizer, detector = _predictor()
    results = _recognize(recognizer, detector, crops)

    out: dict[str, list[Line]] = {}
    for side, offset, result in zip(sides, offsets, results):
        lines = []
        for text_line in result.text_lines:
            box = [int(v) for v in text_line.bbox]
            words = tuple(
                Word(text=w.text.strip(), x0=int(w.bbox[0]) + offset, x1=int(w.bbox[2]) + offset)
                for w in (text_line.words or [])
                if w.text and w.text.strip()
            )
            lines.append(
                Line(
                    text=text_line.text.strip(),
                    top=box[1],
                    bottom=box[3],
                    x0=box[0] + offset,
                    x1=box[2] + offset,
                    words=words,
                )
            )
        out[side] = sorted(lines, key=lambda line: line.top)
    return out
