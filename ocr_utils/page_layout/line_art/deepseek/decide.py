"""Решение по кандидату line art из двух проходов DeepSeek-OCR-2: объекты (класс от DeepSeek, рамка по возможности детектора), надпись или неясно.

Правила (reports/line_art_titles.md, решения пользователя 2026-09-27):

* **Первый проход** — блоки ``markdown`` на рамке кандидата (:func:`rules.on_region`).
  * Есть ``image`` / ``table`` / формула и нет текста — объекты сразу, второй проход не нужен.
    Один класс — **рамка детектора** (у формулы — достроенная по пятнам); несколько — по объекту
    на блок, рамка DeepSeek, достроенная (:func:`expand.grow_to_components`).
  * Только текст — второй проход (рядом с надписью мог спрятаться рисунок или буквица).
  * ``image`` и текст — тоже второй проход: уточнить рамку рисунка без надписи.
* **Второй проход** (слова залиты): нашлись рисунок/таблица — рамка DeepSeek второго прохода
  (плотная, без линеек, объединённая с согласной классикой, :func:`pass2.verdict_pass2`); они
  заменяют рисунки первого прохода, таблицы и формулы первого прохода остаются. Не нашлись —
  итог первого прохода; без объектов — правило надписи: «надпись» или «неясно».
* **Пометка на полях** (:mod:`marks`, 2026-09-30): кандидат-«штрих» по одним пикселям, чей итог — только рисунки или
  «неясно», а остаток залитой вырезки — немного тонких волнистых штрихов без углов, становится «пометкой»: не рисунок
  (не запрет для текстовых блоков, не line art для детектора порчи геометрии).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.classes import ObjectClass
from ocr_utils.page_layout.line_art.deepseek.marks import is_stroke_candidate, looks_like_mark, residue
from ocr_utils.page_layout.line_art.deepseek.pass2 import classic_boxes, fill_words, verdict_pass2
from ocr_utils.page_layout.line_art.deepseek.rules import (
    Crop,
    block_class,
    crop_to_page,
    ink_blobs,
    is_title_like,
    on_region,
)
from ocr_utils.page_layout.line_art.expand import FIGURE_GROW_MM, FORMULA_GROW_MM, grow_to_components


# Версия решения по кандидату (правила этого модуля); пишется в JSON полосы (``pack_analysis.final.versions``).
# Отдельно от ``LINE_ART_VERSION``: классический line art не менялся, его пересчёт в базе разметки не нужен.
# v2 — пометка на полях (:mod:`marks`).
DECISION_VERSION = 2


class Outcome(str, Enum):
    """Итог кандидата."""

    OBJECTS = "объекты"
    TITLE = "надпись"
    UNCLEAR = "неясно"
    MARK = "пометка"  # карандашная пометка на полях: не рисунок (:mod:`marks`)


class BoxSource(str, Enum):
    """Чья рамка у объекта."""

    DETECTOR = "детектор"
    DEEPSEEK = "DeepSeek"
    PASS2 = "DeepSeek, второй проход"


@dataclass
class Decision:
    """Решение по кандидату.

    Attributes:
        outcome: Объекты, надпись, неясно или пометка.
        objects: Объекты ``{"class", "box" (пиксели полосы), "box_source"}``.
        pass2_used: Дал ли второй проход объекты.
    """

    outcome: Outcome
    objects: list[dict] = field(default_factory=list)
    pass2_used: bool = False

    def to_json(self) -> dict:
        """Словарь для JSON."""
        return {"outcome": self.outcome.value, "objects": self.objects, "pass2_used": self.pass2_used}


def regional_classes(crop: Crop, markdown: list[dict]) -> tuple[list[dict], set[ObjectClass], bool]:
    """Нетекстовые блоки на рамке кандидата, их классы и есть ли на рамке текст.

    Args:
        crop: Где вырезка на полосе.
        markdown: Блоки DeepSeek ``markdown`` первого прохода.

    Returns:
        (нетекстовые блоки на рамке, их классы, есть ли на рамке текстовые блоки).
    """
    on = [b for b in markdown if on_region(b, crop.inner)]
    objects = [b for b in on if block_class(b) is not None]
    has_text = any(block_class(b) is None for b in on)
    return objects, {block_class(b) for b in objects}, has_text


def needs_pass2(crop: Crop, markdown: list[dict]) -> bool:
    """Нужен ли второй проход: только текст, либо рисунок вместе с текстом.

    Args:
        crop: Где вырезка на полосе.
        markdown: Блоки первого прохода.

    Returns:
        ``True``, если вырезку надо залить и показать модели снова.
    """
    _, classes, has_text = regional_classes(crop, markdown)
    return not classes or (ObjectClass.DRAWING in classes and has_text)


def _pass1_objects(crop: Crop, blocks: list[dict], classes: set[ObjectClass], ink: np.ndarray, barriers) -> list[dict]:
    """Объекты первого прохода: один класс — рамка детектора, несколько — рамки DeepSeek, достроенные."""
    if len(classes) == 1:
        cls = next(iter(classes))
        box = crop.box
        if cls is ObjectClass.FORMULA:
            # Формула: рамка детектора уточняется, чтобы не резать пятна (индексы, дробная черта).
            box = grow_to_components(box, ink, crop.dpi, FORMULA_GROW_MM, barriers).box
        return [{"class": cls.value, "box": list(box.as_tuple()), "box_source": BoxSource.DETECTOR.value}]
    height, width = ink.shape[:2]
    objects = []
    for block in blocks:
        cls = block_class(block)
        limit = FORMULA_GROW_MM if cls is ObjectClass.FORMULA else FIGURE_GROW_MM
        box = crop_to_page(block, crop).clipped(width, height)
        grown = grow_to_components(box, ink, crop.dpi, limit, barriers).box
        objects.append({"class": cls.value, "box": list(grown.as_tuple()), "box_source": BoxSource.DEEPSEEK.value})
    return objects


def decide(
    crop: Crop,
    gray_crop: np.ndarray,
    markdown: list[dict],
    ink: np.ndarray,
    barriers: list[Box],
    pass2_blocks: list[dict] | None = None,
    pass2_binary: np.ndarray | None = None,
    info: dict | None = None,
    words: list[dict] | None = None,
) -> Decision:
    """Итог по кандидату из обоих проходов с проверкой «пометка на полях».

    Args:
        crop: Где вырезка на полосе.
        gray_crop: Серая вырезка первого прохода.
        markdown: Блоки ``markdown`` первого прохода.
        ink: Маска краски полосы в родном разрешении (для достройки рамок).
        barriers: Растр и таблицы полосы — за них рамки не растут.
        pass2_blocks: Блоки ``markdown`` второго прохода или ``None`` (не было).
        pass2_binary: Залитая вырезка второго прохода (краска 0) или ``None``.
        info: Сведения классического детектора о кандидате (``kind``, ``sources``); ``None`` — проверка пометки не делается.
        words: Слова первого прохода (промпт ``ocr``) — залить вырезку заново, если второго прохода не было;
            ``None`` — тогда пометка проверяется только по вырезке второго прохода.

    Returns:
        :class:`Decision`.
    """
    decision = _decide_objects(crop, gray_crop, markdown, ink, barriers, pass2_blocks, pass2_binary)
    if is_mark(decision, crop, gray_crop, pass2_binary, info, words):
        return Decision(Outcome.MARK)
    return decision


def is_mark(
    decision: Decision,
    crop: Crop,
    gray_crop: np.ndarray,
    pass2_binary: np.ndarray | None,
    info: dict | None,
    words: list[dict] | None,
) -> bool:
    """Пометка ли кандидат: «штрих» по пикселям, итог — только рисунки или «неясно», остаток — ручные штрихи.

    Args:
        decision: Итог по кандидату без проверки пометки.
        crop: Где вырезка на полосе.
        gray_crop: Серая вырезка первого прохода.
        pass2_binary: Залитая вырезка второго прохода или ``None``.
        info: Сведения классического детектора о кандидате.
        words: Слова первого прохода (для заливки заново).

    Returns:
        ``True`` — пометка (:func:`marks.looks_like_mark`).
    """
    drawings_only = decision.outcome is Outcome.OBJECTS and {o["class"] for o in decision.objects} == {
        ObjectClass.DRAWING.value
    }
    if not (drawings_only or decision.outcome is Outcome.UNCLEAR) or not is_stroke_candidate(info):
        return False
    binary = pass2_binary
    if binary is None:
        if words is None:
            return False
        binary, _ = fill_words(gray_crop, crop.inner, words)
    return looks_like_mark(residue(binary, crop.inner))


def _decide_objects(
    crop: Crop,
    gray_crop: np.ndarray,
    markdown: list[dict],
    ink: np.ndarray,
    barriers: list[Box],
    pass2_blocks: list[dict] | None,
    pass2_binary: np.ndarray | None,
) -> Decision:
    """Итог по кандидату из обоих проходов без проверки пометки: объекты, надпись или неясно (аргументы — как у :func:`decide`)."""
    blocks, classes, _ = regional_classes(crop, markdown)
    pass1 = _pass1_objects(crop, blocks, classes, ink, barriers) if classes else []
    if pass2_blocks is not None and pass2_binary is not None:
        found = verdict_pass2(pass2_binary, crop.inner, pass2_blocks, classic_boxes(pass2_binary, crop.inner))
        if found:
            # Рисунки второго прохода заменяют рисунки первого; таблицы и формулы первого остаются.
            kept = [o for o in pass1 if o["class"] != ObjectClass.DRAWING.value]
            second = [
                {
                    "class": o["class"],
                    "box": list(crop_to_page(_as_dict(o["box"]), crop).as_tuple()),
                    "box_source": BoxSource.PASS2.value,
                }
                for o in found
            ]
            return Decision(Outcome.OBJECTS, kept + second, pass2_used=True)
    if pass1:
        return Decision(Outcome.OBJECTS, pass1)
    blobs = ink_blobs(gray_crop, crop.inner)
    return Decision(Outcome.TITLE if is_title_like(blobs, crop, markdown) else Outcome.UNCLEAR)


def _as_dict(box: list[int]) -> dict:
    """Рамка списком → словарь ``x0 … y1`` (формат блоков DeepSeek)."""
    return {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]}


__all__ = ["BoxSource", "DECISION_VERSION", "Decision", "Outcome", "decide", "is_mark", "needs_pass2", "regional_classes"]
