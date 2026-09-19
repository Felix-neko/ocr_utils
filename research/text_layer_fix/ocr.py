"""Чтение зон повёрнутого текста: tesseract по выпрямленной вырезке, второе мнение surya по флагу.

Движок и пороги — из ``ocr_utils.rotated_text.tables`` (там замерено: tesseract на боковых
ячейках CER 0.034 против 2.36 у surya; подмена только при уверенности от 0.6 и слове от
трёх букв). Зона без стороны читается под 90 и 270, побеждает чтение с большим числом
букв, при равенстве — с большей уверенностью.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.rotated_text.tables.ocr import LANGUAGES, CellText, read_cell
from ocr_utils.rotated_text.tables.orientation import LETTER, has_word, looks_like_text
from ocr_utils.rotated_text.tables.pipeline import CONFIDENCE_REPLACE, JUNK_TOKEN_SHARE, NUMERIC_CONFIDENCE, junk_share
from ocr_utils.rotated_text.tables.structure import strip_stubs
from ocr_utils.scan_markup.table_detection.geometry import Box

from research.text_layer_fix.zones import RotatedZone, ZoneKind, rotate_crop

TABLE_KINDS = (ZoneKind.TABLE_CELL, ZoneKind.TABLE_CELL_MIXED, ZoneKind.TABLE_CELL_UPRIGHT)


# Насколько прямое чтение должно быть увереннее повёрнутого, чтобы снять зону с ячейки.
UPRIGHT_MARGIN = 0.1


@dataclass
class ZoneText:
    """Что прочитано в зоне и годится ли это для вставки."""

    text: str
    confidence: float
    engine: str
    rotate_cw: "int | None"
    lines: int = 0
    accepted: bool = False
    reason: str = ""
    letters: int = 0

    def to_json(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "engine": self.engine,
            "rotate_cw": self.rotate_cw,
            "lines": self.lines,
            "accepted": self.accepted,
            "reason": self.reason,
            "letters": self.letters,
        }


def acceptable(text: str, confidence: float, min_letters: int = 3) -> tuple[bool, str]:
    """Годится ли прочитанное для вставки в слой (пороги ``rotated_text.tables.pipeline``).

    Args:
        text: Прочитанный текст.
        confidence: Уверенность движка 0..1.
        min_letters: Слово хотя бы из стольких букв — признак текста.

    Returns:
        Пара «принято, причина отказа».
    """
    if not looks_like_text(text):
        return False, "не похоже на текст"
    if confidence < CONFIDENCE_REPLACE:
        return False, f"уверенность {confidence:.2f} ниже {CONFIDENCE_REPLACE}"
    if junk_share(text) > JUNK_TOKEN_SHARE:
        return False, "россыпь одиночных знаков"
    letters = len(LETTER.findall(text))
    if letters >= min_letters and has_word(text, min_letters):
        return True, ""
    if confidence >= NUMERIC_CONFIDENCE:
        return True, ""
    return False, f"коротко ({letters} букв) и неуверенно ({confidence:.2f})"


def _read(gray: np.ndarray, box: Box, rotate: int, dpi: int, lang: str, table_cell: bool = False) -> CellText:
    """Чтение вырезки под углом. Ячейка таблицы режется ровно по внутренности (без поля,
    иначе в вырезку попадают линейки, и tesseract молчит) и чистится от обрубков линеек."""
    if table_cell:
        crop = rotate_crop(gray, box, 0, pad_mm=0.0, dpi=dpi)
        crop = strip_stubs(crop, dpi)
        from ocr_utils.scan_markup.rotation import rotate_cw as rotate_image

        return read_cell(rotate_image(crop, rotate) if rotate else crop, lang)
    return read_cell(rotate_crop(gray, box, rotate, dpi=dpi), lang)


def _score(reading: CellText) -> tuple[int, float]:
    """Чем сравнивать чтения под разными углами: буквы, затем уверенность."""
    return (len(LETTER.findall(reading.text)), reading.confidence)


def read_zone(
    gray600: np.ndarray, zone: RotatedZone, dpi: int = 600, lang: str = LANGUAGES, compare_upright: bool = False
) -> ZoneText:
    """Прочитать зону; сторону без вердикта выбрать по чтению.

    Args:
        gray600: Растр страницы.
        zone: Зона в пикселях растра.
        dpi: Разрешение растра.
        lang: Языки tesseract.
        compare_upright: Прочитать и без поворота: если прямое чтение даёт не меньше букв и
            выше уверенность, зона на самом деле прямая (широкие «ШТ.» по форме «лежат»), и
            ответ приходит с ``rotate_cw == 0`` — такую зону трогать нельзя.

    Returns:
        Текст зоны с уверенностью, стороной и решением о пригодности.
    """
    candidates = [zone.rotate_cw] if zone.rotate_cw is not None else [90, 270]
    table_cell = zone.kind in TABLE_KINDS
    readings: list[tuple[int, CellText]] = [
        (angle, _read(gray600, zone.box, angle, dpi, lang, table_cell)) for angle in candidates
    ]
    angle, best = max(readings, key=lambda item: _score(item[1]))
    if compare_upright and zone.rotate_cw != 0:
        upright = _read(gray600, zone.box, 0, dpi, lang, table_cell)
        letters_up, conf_up = _score(upright)
        letters_best, conf_best = _score(best)
        # Прямое чтение побеждает, если оно заметно увереннее (мусор от боковых букв tesseract
        # читает с уверенностью 0.5–0.8, «шт.» прямо — 0.95) при хотя бы двух буквах, либо
        # не уступает по буквам при большей уверенности; одни цифры — только при высокой.
        wins_by_margin = letters_up >= 2 and conf_up >= conf_best + UPRIGHT_MARGIN
        wins_by_letters = (
            letters_up >= letters_best and conf_up > conf_best and (letters_up > 0 or conf_up >= NUMERIC_CONFIDENCE)
        )
        if wins_by_margin or wins_by_letters:
            text = upright.text
            return ZoneText(
                text,
                upright.confidence,
                upright.engine or "tesseract",
                0,
                len(upright.lines),
                False,
                "прямой текст читается лучше повёрнутого",
                letters_up,
            )
    text = best.text
    accepted, reason = acceptable(text, best.confidence)
    return ZoneText(
        text,
        best.confidence,
        best.engine or "tesseract",
        angle,
        len(best.lines),
        accepted,
        reason,
        len(LETTER.findall(text)),
    )
