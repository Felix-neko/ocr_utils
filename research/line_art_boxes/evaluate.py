"""Сверка вариантов с ручной разметкой рамок боевого кода (``sets/*.csv``): сколько испорченных рамок осталось, сколько настоящих рисунков потеряно, сколько малых рамок добрано."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from research.line_art_boxes.detect import PageArt, load_page_art


class BoxKind(StrEnum):
    """Что на самом деле в рамке боевого кода (разметка глазами по листам ``sheets --variant base``)."""

    DRAWING = "drawing"  # рисунок, схема — рамка верная
    LOGO = "logo"  # логотип рубрики
    ORNAMENT = "ornament"  # орнамент или фигурная линейка рубрики
    MASTHEAD = "masthead"  # шапка журнала «ХОЗЯЙСТВО И ПРАВО» (часто с рукописным номером поверх)
    OVERGROWN = "overgrown"  # рисунок плюс заголовок и текст (рост через полосу у корешка)
    HEADING = "heading"  # слипшиеся буквы заголовка
    BODY_TEXT = "body_text"  # слипшиеся буквы двух строк основного текста
    TABLE = "table"  # кусок таблицы, не найденной детектором таблиц
    STAMP_TEXT = "stamp_text"  # штамп или рукописная пометка поверх строк текста (рамка накрывает строки)
    HANDWRITING = "handwriting"  # рукописные пометки на полях, обложке, пустой странице
    EDGE_STRIP = "edge_strip"  # полоса тени у края вырезки (пустое поле, строки не задеты)
    BLANK_PAGE = "blank_page"  # пустая страница целиком


# Рамки, выкидывающие строки текста из опор поля (вредные для rectify).
HARMFUL = (BoxKind.OVERGROWN, BoxKind.HEADING, BoxKind.BODY_TEXT, BoxKind.TABLE, BoxKind.STAMP_TEXT)
# Ложные, но безвредные: на полях, на пустых страницах.
HARMLESS = (BoxKind.HANDWRITING, BoxKind.EDGE_STRIP, BoxKind.BLANK_PAGE)
# Верные рамки: их терять нельзя.
CORRECT = (BoxKind.DRAWING, BoxKind.LOGO, BoxKind.ORNAMENT, BoxKind.MASTHEAD)

# Разметка ``sets/hp_1991_base.csv`` сделана по рамкам прежнего кода (до правок ядра 2026-10-25): номера рамок — в
# разборе этого варианта.
LABELS_VARIANT = "legacy"

# Ложная рамка осталась, если рамки варианта накрывают её не меньше чем на эту долю.
REMAIN_COVER = 0.5
# Верная рамка потеряна, если рамки варианта накрывают её меньше чем на эту долю.
KEEP_COVER = 0.8
# Завышенная рамка исправлена, если лучшая рамка варианта не больше этой доли её площади.
SHRUNK_SHARE = 0.7
# Малая рамка добрана, если лучшая рамка варианта больше её хотя бы на эту долю площади.
GROWN_SHARE = 1.03


@dataclass(frozen=True)
class Label:
    """Строка разметки: страница, номер рамки в разборе ``LABELS_VARIANT`` (с 1), что в ней, мала ли рамка."""

    page: str
    box: int
    kind: BoxKind
    undersized: bool


@dataclass(frozen=True)
class Tally:
    """Итог варианта по разметке.

    Поля:
        variant: Вариант.
        harmful_left: Вредных рамок осталось (по видам).
        harmless_left: Безвредных ложных осталось.
        overgrown_fixed: Завышенных исправлено из всех завышенных.
        correct_lost: Потерянные верные рамки (``страница#номер``).
        undersized_grown: Малых рамок добрано из всех малых.
        boxes: Всего рамок варианта на размеченных страницах.
    """

    variant: str
    harmful_left: dict[str, int]
    harmless_left: int
    overgrown_fixed: tuple[int, int]
    correct_lost: tuple[str, ...]
    undersized_grown: tuple[int, int]
    boxes: int


def read_labels(path: Path) -> list[Label]:
    """Прочитать разметку ``page,box,kind,undersized``.

    Args:
        path: CSV.

    Returns:
        Строки разметки.
    """
    with path.open() as f:
        return [Label(r["page"], int(r["box"]), BoxKind(r["kind"]), r["undersized"] == "1") for r in csv.DictReader(f)]


def _area(box) -> int:
    """Площадь рамки ``(x0, y0, x1, y1)``."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _common(a, b) -> int:
    """Площадь пересечения двух рамок."""
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def cover(box, others) -> float:
    """Доля площади ``box``, накрытая рамками ``others`` (сумма пересечений, не больше 1)."""
    return min(1.0, sum(_common(box, o) for o in others) / max(1, _area(box)))


def best_match(box, others):
    """Рамка из ``others`` с наибольшим пересечением с ``box`` или None."""
    scored = [(o, _common(box, o)) for o in others]
    scored = [s for s in scored if s[1] > 0]
    return max(scored, key=lambda s: s[1])[0] if scored else None


def page_key(path: Path) -> str:
    """Ключ страницы разметки по JSON разбора: ``<2 буквы выпуска>/<вырезка>``."""
    return f"{path.parent.name[:2]}/{path.stem}"


def tally(labels: list[Label], pages: dict[str, dict[str, PageArt]], variant: str) -> Tally:
    """Итог варианта по разметке.

    Args:
        labels: Разметка рамок прежнего кода (``LABELS_VARIANT``).
        pages: Ключ страницы → разборы вариантов.
        variant: Вариант.

    Returns:
        :class:`Tally`.
    """
    harmful = {k.value: 0 for k in HARMFUL}
    harmless, fixed, overgrown, grown, small = 0, 0, 0, 0, 0
    lost: list[str] = []
    for label in labels:
        arts = pages.get(label.page)
        if arts is None or variant not in arts or LABELS_VARIANT not in arts:
            continue
        box = arts[LABELS_VARIANT].boxes[label.box - 1].box
        others = [b.box for b in arts[variant].boxes]
        covered = cover(box, others)
        if label.kind is BoxKind.OVERGROWN:
            overgrown += 1
            match = best_match(box, others)
            fixed += int(match is None or _area(match) <= SHRUNK_SHARE * _area(box))
            if match is not None and _area(match) > SHRUNK_SHARE * _area(box):
                harmful[label.kind.value] += 1
        elif label.kind in HARMFUL:
            harmful[label.kind.value] += int(covered >= REMAIN_COVER)
        elif label.kind in HARMLESS:
            harmless += int(covered >= REMAIN_COVER)
        elif covered < KEEP_COVER:
            lost.append(f"{label.page}#{label.box} ({label.kind.value})")
        if label.undersized:
            small += 1
            match = best_match(box, others)
            grown += int(match is not None and _area(match) >= GROWN_SHARE * _area(box))
    count = sum(len(arts[variant].boxes) for arts in pages.values() if variant in arts)
    return Tally(variant, harmful, harmless, (fixed, overgrown), tuple(lost), (grown, small), count)


def load_pages(detect_dir: Path, rectify_only: bool = True) -> dict[str, dict[str, PageArt]]:
    """Разборы всех страниц папки ``detect``.

    Args:
        detect_dir: Папка JSON разборов.
        rectify_only: Только страницы, где ``rectify`` зовёт детектор (не цветные, флаги разрешают поле).

    Returns:
        Ключ страницы → вариант → разбор.
    """
    pages = {}
    for path in sorted(detect_dir.glob("*/*.json")):
        meta, arts = load_page_art(path)
        if rectify_only and not (meta.get("plain", True) and meta.get("field", True)):
            continue
        # Страницы PDF (``regress``) — по полному имени PDF: префикс выпуска у них общий («fu»).
        pages[f"{path.parent.name}/{path.stem}" if "pdf" in meta else page_key(path)] = arts
    return pages


def changes(
    pages: dict[str, dict[str, PageArt]], variant: str, reference: str = LABELS_VARIANT
) -> tuple[int, int, int, int]:
    """Сколько рамок варианта отличается от варианта ``reference`` без разметки.

    Args:
        pages: Разборы.
        variant: Вариант.
        reference: С чем сравнивать (по умолчанию — прежний код).

    Returns:
        ``(страниц со сменой, снято рамок, новых рамок, рамок со сменой границ)``: снята — рамку ``reference`` вариант
        накрывает меньше чем на половину; новая — рамку варианта ``reference`` накрывает меньше чем на половину.
    """
    changed = removed = added = moved = 0
    for arts in pages.values():
        if variant not in arts or reference not in arts:
            continue
        before = [b.box for b in arts[reference].boxes]
        after = [b.box for b in arts[variant].boxes]
        if before == after:
            continue
        changed += 1
        removed += sum(cover(b, after) < REMAIN_COVER for b in before)
        added += sum(cover(a, before) < REMAIN_COVER for a in after)
        moved += sum(cover(b, after) >= REMAIN_COVER and b not in after for b in before)
    return changed, removed, added, moved
