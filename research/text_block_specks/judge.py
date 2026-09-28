"""Судьи «буква или мусор» по распознаванию: пара вырезок конца строки — с компонентом и с закрашенным компонентом.

Разностный судья: если текст вырезки не меняется, когда спорный компонент закрашен белым, компонент —
мусор; если пропал знак (точка, дефис, буква) — компонент настоящий. Распознают DeepSeek-OCR-2 (vLLM,
промпт ``free`` — :mod:`deepseek_run`) и tesseract (``--psm 7``, одна строка) как дешёвое сравнение.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI

K = RENDER_DPI / WORK_DPI
# Поле вырезки в высотах строчной: вглубь текста от компонента, наружу за него, над верхом строчных и под базовой линией.
CROP_IN_XH, CROP_OUT_XH, CROP_UP_XH, CROP_DOWN_XH = 12.0, 1.5, 1.0, 0.8
# Закраска компонента: бокс раздувается на столько пикселей рендера (антиалиас края).
ERASE_PAD = 2
# Увеличение вырезки перед распознаванием (у мелкой вырезки на глобальном виде 1024 буквы слишком мелкие).
UPSCALE = 2
# Поле белого вокруг вырезки, доля её высоты.
MARGIN = 0.5


class Variant(str, Enum):
    """Вариант вырезки."""

    WITH = "with"  # как на странице
    ERASED = "erased"  # спорный компонент закрашен белым


@dataclass(frozen=True)
class Candidate:
    """Спорный компонент для судьи.

    Attributes:
        id: Идентификатор (подпись и имя файла).
        pdf: Имя PDF.
        page: Номер страницы PDF с единицы.
        side: ``left`` или ``right`` — у какого конца строки компонент.
        box: Бокс компонента ``x0, y0, x1, y1`` (пиксели рабочей копии).
        x_h: Высота строчной строки, пиксели рабочей копии.
        bottom: Низ компонента относительно базовой линии в ``x_h`` (из признаков) — по нему восстанавливается базовая линия.
        span: Края строки по x (пиксели рабочей копии): вырезка берёт строку целиком — у полной строки
            распознавание устойчивее, чем у обрывка с разрезанным словом. ``None`` — ``CROP_IN_XH`` от компонента.
    """

    id: str
    pdf: str
    page: int
    side: str
    box: tuple[float, float, float, float]
    x_h: float
    bottom: float
    span: tuple[float, float] | None = None


def crop_pair(gray: np.ndarray, candidate: Candidate) -> dict[Variant, np.ndarray]:
    """Две вырезки конца строки ``RENDER_DPI`` (с компонентом и без), увеличенные и с белым полем.

    Args:
        gray: Серый рендер страницы ``RENDER_DPI``.
        candidate: Спорный компонент.

    Returns:
        ``вариант → серая вырезка``.
    """
    x0, y0, x1, y1 = (v * K for v in candidate.box)
    x_h = candidate.x_h * K
    # Базовая линия строки: низ компонента минус его смещение от неё.
    base = y1 - candidate.bottom * x_h
    top, bottom = base - x_h - CROP_UP_XH * x_h, base + CROP_DOWN_XH * x_h
    # Край строки годится, только если лежит по нужную сторону от компонента (иначе ряд чужой).
    span = candidate.span
    if span is not None and (span[0] * K >= x0 if candidate.side == "right" else span[1] * K <= x1):
        span = None
    if candidate.side == "right":
        inner = span[0] * K - x_h if span else x0 - CROP_IN_XH * x_h
        left, right = inner, x1 + CROP_OUT_XH * x_h
    else:
        inner = span[1] * K + x_h if span else x1 + CROP_IN_XH * x_h
        left, right = x0 - CROP_OUT_XH * x_h, inner
    # Компонент может торчать за полосу строки (соринка ниже базовой линии) — полоса расширяется до него.
    top, bottom = min(top, y0 - 2), max(bottom, y1 + 2)
    box = [int(round(v)) for v in (left, top, right, bottom)]
    box = [max(0, box[0]), max(0, box[1]), min(gray.shape[1], box[2]), min(gray.shape[0], box[3])]
    crop = gray[box[1] : box[3], box[0] : box[2]].copy()
    # Закрашивается всё от компонента наружу по всей высоте вырезки: соринка на бинарном PDF часто
    # распадается на несколько кусков, и закраска одного бокса оставляла соседний кусок «точкой».
    erased = crop.copy()
    if candidate.side == "right":
        erased[:, max(0, int(x0) - ERASE_PAD - box[0]) :] = 255
    else:
        erased[:, : max(0, int(np.ceil(x1)) + ERASE_PAD - box[0])] = 255
    return {Variant.WITH: _framed(crop), Variant.ERASED: _framed(erased)}


def _framed(crop: np.ndarray) -> np.ndarray:
    """Вырезка, увеличенная в ``UPSCALE`` раз и обведённая белым полем."""
    big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    pad = int(MARGIN * big.shape[0])
    return cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def tesseract_text(image_path: Path) -> str:
    """Текст одной строки по tesseract (русский, ``--psm 7``)."""
    result = subprocess.run(
        ["tesseract", str(image_path), "-", "-l", "rus", "--psm", "7"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


# Сколько символов у края должны совпасть, чтобы расхождения дальше от края не считались.
TAIL = 3
# Всё, кроме букв, цифр и знаков препинания, при сравнении не учитывается (пробелы, переводы строк, разметка).
NORMALIZE = re.compile(r"[\s*_`#|<>\\]+")


def normalized(text: str) -> str:
    """Текст для сравнения: без пробелов и служебной разметки markdown."""
    return NORMALIZE.sub("", text)


class Verdict(str, Enum):
    """Вердикт разностного судьи."""

    NOISE = "noise"  # текст не изменился — компонент мусор
    SIGN = "sign"  # без компонента пропал знак — компонент настоящий
    UNSURE = "unsure"  # текст изменился не только на краю (распознавание неустойчиво)
    EMPTY = "empty"  # пустой ответ


def diff_verdict(with_text: str, erased_text: str, side: str) -> tuple[Verdict, str]:
    """Вердикт по паре распознанных текстов: сравниваются только края текста у компонента.

    Распознавание всей строки от вырезки к вырезке немного плавает в середине (одна-две буквы), и
    требовать полного совпадения — значит почти всегда получать «неясно». Поэтому тексты сверяются по
    выравниванию (``difflib``): если все расхождения лежат дальше ``TAIL`` символов от края у компонента,
    а у края тексты совпадают — мусор; если у края в вырезке с компонентом есть лишние символы, а в
    остальном края совпадают — знак.

    Args:
        with_text: Текст вырезки с компонентом.
        erased_text: Текст вырезки с закрашенным компонентом.
        side: У какого конца строки компонент.

    Returns:
        Пара: вердикт и «лишние» символы вырезки с компонентом (что компонент добавил к тексту).
    """
    a, b = normalized(with_text), normalized(erased_text)
    if not a:
        return Verdict.EMPTY, ""
    if a == b:
        return Verdict.NOISE, ""
    # Левый конец сводится к правому переворотом строк: край у компонента — всегда конец строки.
    if side == "left":
        a, b = a[::-1], b[::-1]
    # Лишнее в конце ``a``: хвост, которому в ``b`` ничего не соответствует.
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return Verdict.UNSURE, a
    last = blocks[-1]
    extra_a, extra_b = a[last.a + last.size :], b[last.b + last.size :]
    # Совпадающий кусок должен доходить до края: иначе расхождение у самого края — неясно.
    if last.size < TAIL:
        return Verdict.UNSURE, a[-TAIL:]
    if not extra_a and not extra_b:
        return Verdict.NOISE, ""
    if extra_a and not extra_b:
        return Verdict.SIGN, extra_a[::-1] if side == "left" else extra_a
    return Verdict.UNSURE, extra_a


def write_jobs(entries: list[tuple[str, Path]], path: Path) -> None:
    """Файл заданий воркера DeepSeek: строки ``{"id", "crop"}``."""
    path.write_text("".join(json.dumps({"id": i, "crop": str(p)}) + "\n" for i, p in entries))


__all__ = ["Candidate", "Variant", "Verdict", "crop_pair", "diff_verdict", "normalized", "tesseract_text", "write_jobs"]
