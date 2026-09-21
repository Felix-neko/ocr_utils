"""Вердикт по каждому слову текстового слоя: оставить, удалить, усечь, показать как подозрительное.

Решение геометрическое: рамка слова FineReader против зон повёрнутого текста в пикселях
растра. Слово, чья рамка лежит в зоне на ``DELETE_OVERLAP`` и больше, — удалить; задетое
зоной частично — усечь до глифов вне зоны (по рамке каждого глифа), а не править текст
языковой моделью; слово с повёрнутой матрицей FineReader — оставить как есть (он сам
прочитал его правильно). Короткое или бессмысленное слово рядом с зоной, но вне её —
SUSPECT: не удаляется, только показывается на оверлее, чтобы оценить пропуски детектора.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

import fitz

from ocr_utils.page_layout.geometry import Box

from ocr_utils.text_layer_fix import px_to_mm
from ocr_utils.text_layer_fix.text_layer import SpanShape, TextLayer
from ocr_utils.text_layer_fix.zones import RotatedZone, ZoneKind

# Доля площади слова внутри зоны, от которой слово удаляется целиком.
DELETE_OVERLAP = 0.6
# Доля, от которой слово считается задетым и усекается; ниже — не трогается.
SANITIZE_LOW = 0.15
# Доля площади глифа в зоне, от которой глиф вырезается при усечении.
GLYPH_DROP_OVERLAP = 0.5
# Подозрительное слово вне зон учитывается, если до ближайшей зоны не дальше стольких мм.
SUSPECT_REACH_MM = 4.0

LETTERS = re.compile(r"[а-яёА-ЯЁa-zA-Z]")
DIGITS = re.compile(r"[0-9]")
CYRILLIC_WORD = re.compile(r"[а-яёА-ЯЁ]{3,}")
NUMBER = re.compile(r"^[\d\s.,;:%—–-]+$")


def looks_genuine(text: str, known: "Callable[[str], bool] | None") -> bool:
    """Похоже ли слово на настоящее чтение (число или словарное слово от трёх букв).

    Args:
        text: Текст слова.
        known: Проверка «слово есть в словаре» (``hyphen_join.Morph.known``); None — только числа.

    Returns:
        True для «Наименование», «1966», «12,5 %»; False для «СЪСЧчГ», «Е», «goo».
    """
    stripped = text.strip()
    if not stripped:
        return False
    if NUMBER.match(stripped) and DIGITS.search(stripped):
        return True
    if known is None:
        return False
    return any(known(piece.lower()) for piece in CYRILLIC_WORD.findall(stripped))


class Verdict(StrEnum):
    """Что делать со словом слоя."""

    KEEP = "keep"
    KEEP_ROTATED = "keep_rotated"  # FineReader сам написал его повёрнутым — верно
    DELETE = "delete"
    SANITIZE = "sanitize"  # усечь до глифов вне зоны
    SUSPECT = "suspect"  # похоже на мусор рядом с зоной, но зоной не накрыто; не трогаем


@dataclass
class WordVerdict:
    """Решение по одному слову с тем, по чему оно принято."""

    mcid: "int | None"
    verdict: Verdict
    text: str
    bbox_px: tuple[int, int, int, int]
    zone_index: "int | None" = None
    overlap: float = 0.0
    keep_glyphs: tuple[int, ...] = ()
    dist_mm: float = 0.0
    reason: str = ""
    struct_line: "str | None" = None
    shape: str = ""
    stretch: float = 0.0

    def to_json(self) -> dict:
        return {
            "mcid": self.mcid,
            "verdict": str(self.verdict),
            "text": self.text,
            "bbox_px": list(self.bbox_px),
            "zone_index": self.zone_index,
            "overlap": round(self.overlap, 3),
            "keep_glyphs": list(self.keep_glyphs),
            "dist_mm": round(self.dist_mm, 2),
            "reason": self.reason,
            "struct_line": self.struct_line,
            "shape": self.shape,
            "stretch": round(self.stretch, 3),
        }


def looks_like_junk(text: str) -> bool:
    """Похоже ли слово на россыпь от повёрнутого текста: коротко и без цифр, либо без букв и цифр вовсе.

    Args:
        text: Текст слова.

    Returns:
        True для «X», «^», «i"o=», «goo»-подобных обрывков; числа и слова от трёх букв — False.
    """
    stripped = text.strip()
    if not stripped:
        return False
    letters = len(LETTERS.findall(stripped))
    digits = len(DIGITS.findall(stripped))
    if letters == 0 and digits == 0:
        return True
    if digits and letters == 0:
        return False
    return letters <= 2 and digits == 0


def _overlap(rect: fitz.Rect, box: Box) -> float:
    """Доля площади ``rect`` внутри ``box``."""
    area = rect.get_area()
    if area <= 0:
        return 0.0
    inter = rect & fitz.Rect(box.x0, box.y0, box.x1, box.y1)
    return inter.get_area() / area if not inter.is_empty else 0.0


def _distance_px(rect: fitz.Rect, box: Box) -> float:
    """Расстояние между рамками в пикселях (0, если пересекаются)."""
    dx = max(box.x0 - rect.x1, rect.x0 - box.x1, 0.0)
    dy = max(box.y0 - rect.y1, rect.y0 - box.y1, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _active(zone: RotatedZone) -> bool:
    """Зона, в которой слой подлежит удалению: боковая ячейка (даже без стороны) и всё, где сторона найдена.

    Зона с ``rotate_cw == 0`` снята чтением (прямой текст читается лучше) — не активна."""
    if zone.rotate_cw == 0:
        return False
    if zone.kind == ZoneKind.TABLE_CELL:
        return True
    if zone.kind == ZoneKind.TABLE_CELL_MIXED:
        return zone.rotate_cw is not None
    return zone.rotate_cw is not None


def classify_words(
    layer: TextLayer,
    zones: list[RotatedZone],
    px: fitz.Matrix,
    dpi: float,
    readable: "set[int] | None" = None,
    known: "Callable[[str], bool] | None" = None,
) -> list[WordVerdict]:
    """Вердикты по всем словам страницы.

    Args:
        layer: Разобранный слой страницы.
        zones: Зоны повёрнутого текста в пикселях растра.
        px: Матрица «pt → пиксели растра» (``PageRaster.to_px``).
        dpi: Разрешение растра (для расстояний в мм).
        readable: Индексы зон, чьё чтение принято (есть чем заменить). В зоне без принятого
            чтения удаляется только то, что похоже на мусор: настоящее слово или число
            FineReader мог прочитать верно (замер по выборке: «Наименование материалов» в
            ячейке, которую tesseract прочёл как «в»), и терять его ради пустоты незачем —
            такое слово остаётся с вердиктом SUSPECT. None — считать все зоны прочитанными.
        known: Словарная проверка для ``readable``; None — настоящими считаются только числа.

    Returns:
        Список вердиктов в порядке слов слоя (пустые спаны пропущены).
    """
    active = [(i, z) for i, z in enumerate(zones) if _active(z)]
    # Пассивные зоны — смешанные ячейки без стороны: слова в них только показываются. Зоны,
    # снятые чтением (``rotate_cw == 0``), не считаются вовсе.
    passive = [(i, z) for i, z in enumerate(zones) if not _active(z) and z.rotate_cw != 0]
    result: list[WordVerdict] = []
    for word in layer.words:
        text = word.text.strip()
        if not word.glyphs or not text:
            continue
        rect = word.bbox * px
        record = WordVerdict(
            word.mcid,
            Verdict.KEEP,
            text,
            tuple(int(round(v)) for v in rect),
            struct_line=word.struct_line,
            shape=str(word.shape),
            stretch=word.stretch,
        )
        if word.shape == SpanShape.ROTATED:
            record.verdict = Verdict.KEEP_ROTATED
            record.reason = "FineReader написал слово повёрнутым"
            best = max(((i, _overlap(rect, z.box)) for i, z in active), key=lambda t: t[1], default=(None, 0.0))
            record.zone_index, record.overlap = best
            result.append(record)
            continue
        best_index, best_overlap = None, 0.0
        for i, zone in active:
            overlap = _overlap(rect, zone.box)
            if overlap > best_overlap:
                best_index, best_overlap = i, overlap
        record.zone_index, record.overlap = best_index, best_overlap
        if best_overlap >= DELETE_OVERLAP:
            unreadable = readable is not None and best_index not in readable
            if unreadable and not looks_like_junk(text) and looks_genuine(text, known):
                record.verdict = Verdict.SUSPECT
                record.reason = "зона не прочитана, слово похоже на настоящее"
            else:
                record.verdict = Verdict.DELETE
                record.reason = f"в зоне на {best_overlap:.2f}" + (" (зона не прочитана, мусор)" if unreadable else "")
        elif best_overlap >= SANITIZE_LOW:
            zone = zones[best_index]
            keep = tuple(
                g.index for g in word.glyphs if g.bbox is None or _overlap(g.bbox * px, zone.box) < GLYPH_DROP_OVERLAP
            )
            if not keep:
                record.verdict, record.reason = Verdict.DELETE, "все глифы в зоне"
            elif len(keep) == len(word.glyphs):
                record.verdict, record.reason = Verdict.KEEP, "рамка задета, глифы вне зоны"
            else:
                record.verdict, record.keep_glyphs = Verdict.SANITIZE, keep
                record.reason = f"глифов вне зоны {len(keep)} из {len(word.glyphs)}"
        else:
            nearest = min((_distance_px(rect, z.box) for _, z in active + passive), default=float("inf"))
            record.dist_mm = px_to_mm(nearest, dpi) if nearest != float("inf") else 999.0
            inside_passive = any(_overlap(rect, z.box) >= DELETE_OVERLAP for _, z in passive)
            if inside_passive:
                record.verdict, record.reason = Verdict.SUSPECT, "в смешанной ячейке без стороны"
            elif looks_like_junk(text) and record.dist_mm <= SUSPECT_REACH_MM:
                record.verdict, record.reason = Verdict.SUSPECT, "россыпь рядом с зоной"
        result.append(record)
    return result


@dataclass
class VerdictCounts:
    """Сколько слов каждого вердикта на странице."""

    counts: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def of(verdicts: list[WordVerdict]) -> "VerdictCounts":
        counts: dict[str, int] = {v.value: 0 for v in Verdict}
        for record in verdicts:
            counts[record.verdict.value] += 1
        return VerdictCounts(counts)
