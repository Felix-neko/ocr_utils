"""Разметка пака из sqlite — в плоские, пиклуемые структуры.

ЗАЧЕМ ПЛОСКО. Пиксельная часть прогона идёт в ``ProcessPoolExecutor``, а воркер не
должен видеть ни sqlite, ни ORM, ни ``cvat_sdk``: соединение SQLAlchemy через
pickle не проходит, ленивая подгрузка связей в чужом процессе тем более, а
открывать базу в каждом воркере значило бы держать 8 соединений к файлу, который
всё равно только читают. Поэтому база читается ОДИН раз, в родителе, и дальше по
процессам едут обычные датаклассы.

Стоит это дёшево: вся разметка пака-1 — 620 прямоугольников и 431 маска, весь RLE
вместе занимает сотни килобайт.

Формат хранения маски (RLE от CVAT) по-прежнему знает ровно один модуль —
``scan_markup.cvat.export``; здесь только вызов его :func:`mask_from_row`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_utils.scan_markup.cvat.export import mask_from_row
from ocr_utils.scan_markup.db.models import COLOR_PICTURE_KINDS, MASK_KINDS, MaskAnnotation
from ocr_utils.scan_markup.db.repo import iter_pages, require_pack
from ocr_utils.scan_markup.db.session import open_db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rect:
    """Растровая область полосы в координатах оригинала."""

    x1: int
    y1: int
    x2: int
    y2: int
    kind: str
    full_page: bool

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class MaskRow:
    """Строка маски под удаление: охватывающий прямоугольник и RLE."""

    kind: str
    left: int
    top: int
    width: int
    height: int
    rle: str


@dataclass(frozen=True)
class PageMarkup:
    """Всё, что нужно знать о полосе, чтобы её обработать, — без обращения к базе."""

    rel_path: str
    width: int
    height: int
    dpi: "int | None"
    divisor: "int | None"
    regions: "tuple[Rect, ...]"
    masks: "tuple[MaskRow, ...]"
    file_name: str = ""
    # sha256 исходника с шага ``detect``. Едет сюда затем, что из него собирается имя
    # выходного файла (см. ``scan_cleanup.naming``), а воркер базы не видит.
    file_hash: "str | None" = None
    # На сколько повернуть полосу ПО ЧАСОВОЙ при записи. Едет сюда по той же причине, что и
    # отпечаток: решение принято по базе, а воркер базы не видит.
    rotate_cw: int = 0

    def masks_of(self, kind: str) -> "tuple[MaskRow, ...]":
        return tuple(m for m in self.masks if m.kind == kind)

    def regions_of(self, kinds: "tuple[str, ...]") -> "tuple[Rect, ...]":
        return tuple(r for r in self.regions if r.kind in kinds)

    @property
    def needs_inpaint(self) -> bool:
        """Нужен ли этой полосе GPU. По нему прогон делится на два этапа."""
        return bool(self.masks)

    def source_path(self, pack_dir: Path) -> Path:
        return pack_dir / self.rel_path

    @property
    def has_colour(self) -> bool:
        """Есть ли на полосе то, ради чего её стоит хранить В ЦВЕТЕ.

        Это цветной растр и цветной типографский набор — список берётся из
        ``COLOR_PICTURE_KINDS``, а не сравнением с ``KIND_COLOR``: потребитель, написавший
        «цвет, если kind == color», молча обесцветил бы цветной текст. Серый растр цвета
        не требует: он и в оригинале серый.
        """
        return any(r.kind in COLOR_PICTURE_KINDS for r in self.regions)


def load_markup(
    db_path: Path,
    pack_name: str,
    *,
    only_year: "str | None" = None,
    only_issue: "str | None" = None,
    only_rel: "set[str] | None" = None,
    limit: "int | None" = None,
) -> "list[PageMarkup]":
    """Полосы пака с их разметкой, в порядке год → выпуск → номер полосы.

    ``only_rel`` — точный набор относительных путей (для сравнений и повторов по
    списку); он применяется поверх ``only_year`` / ``only_issue``.

    Полосы без размеров (``width``/``height`` пусты — ``detect`` по ним не ходил)
    пропускаются с предупреждением: без размеров кадра ни маску не развернуть, ни
    прямоугольник не проверить.
    """
    session_factory = open_db(db_path, create=False)
    pages: "list[PageMarkup]" = []
    skipped = 0
    with session_factory() as session:
        pack = require_pack(session, pack_name)
        for _year, _issue, page in iter_pages(pack, only_year, only_issue):
            if only_rel is not None and page.source_rel_path not in only_rel:
                continue
            if not page.width or not page.height:
                skipped += 1
                continue
            pages.append(
                PageMarkup(
                    rel_path=page.source_rel_path,
                    file_name=page.source_file_name,
                    file_hash=page.file_hash,
                    width=int(page.width),
                    height=int(page.height),
                    dpi=int(page.dpi) if page.dpi else None,
                    divisor=int(page.divisor) if page.divisor else None,
                    rotate_cw=int(page.rotate_cw or 0),
                    regions=tuple(
                        Rect(int(r.x1), int(r.y1), int(r.x2), int(r.y2), r.kind, bool(r.full_page))
                        for r in page.rect_regions
                    ),
                    masks=tuple(
                        MaskRow(m.kind, int(m.left), int(m.top), int(m.width), int(m.height), m.rle)
                        for m in page.masks
                        if m.kind in MASK_KINDS
                    ),
                )
            )
            if limit is not None and len(pages) >= limit:
                break

    if skipped:
        logger.warning("Пропущено полос без размеров кадра (не проходила детекция): %d", skipped)
    return pages


def decode_mask_rows(rows: "tuple[MaskRow, ...]", width: int, height: int) -> np.ndarray:
    """Объединённая маска строк — bool во весь кадр оригинала.

    Строк одного вида на полосе бывает несколько (разметчик обвёл печать и подпись
    двумя объектами), и для группировки их надо сперва слить в одну карту: связность
    считается по ней, а не по отдельным строкам.
    """
    out = np.zeros((height, width), dtype=bool)
    for row in rows:
        out |= mask_from_row(
            MaskAnnotation(kind=row.kind, left=row.left, top=row.top, width=row.width, height=row.height, rle=row.rle),
            width,
            height,
        )
    return out
