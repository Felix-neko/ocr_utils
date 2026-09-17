"""Растровые области из базы разметки — в координатах страницы бинаризованного PDF.

ЗАЧЕМ. Полутоновую фотографию распрямление строк заметно не портит, и в список на
просмотр она попадать не должна. Отличить её от штриха ПИКСЕЛЯМИ на этом материале не
вышло (замер — в докстринге ``features``), зато для пака-1 уже есть выверенная глазами
разметка в CVAT, сложенная в SQLite. Её и берём.

ПЕРЕСЧЁТ КООРДИНАТ ТОЧЕН, И ЭТО ПРОВЕРЕНО. Промежуточный PDF собран из скана с полями
(``pdf_utils.intermediate_pdfs``), без масштабирования, поэтому

    страница PDF в пикселях = скан + поля с четырёх сторон.

Проверка на ``full_1966_01.pdf``, страница 5: скан 3589x6445 при 600 dpi, поля выпуска
12.192 и 6.096 мм, то есть 288 и 144 px; страница рендерится ровно в 4165x6733 =
3589 + 2*288 на 6445 + 2*144. Сходится до пикселя, никаких подгонок не требуется.

Ключ разметки — пара ``(имя PDF, номер страницы в нём)``: ``issues.full_intermediate_pdf_name``
и ``pages.full_pdf_page_idx``. Именно эту пару и знает детектор, обходящий папку с PDF;
относительный путь исходного скана ему неизвестен и не нужен, поэтому ``scan_cleanup.source.load_markup``
здесь не подходит — она отдаёт разметку по ``rel_path`` и без полей выпуска.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from ocr_utils.db.models import RASTER_KINDS, Issue, Page, RectRegion, YearPackage
from ocr_utils.db.repo import require_pack
from ocr_utils.db.session import open_db

logger = logging.getLogger(__name__)

# Исключаются области всех РАСТРОВЫХ видов (``RASTER_KINDS`` из моделей, а не свой список:
# новый вид растра иначе прошёл бы мимо молча). ``stamp_suspect`` — библиотечная печать:
# это цветной ШТРИХ, но и не иллюстрация, смотреть её после FineReader незачем. Таблицы и
# схемы (``TABLE_KINDS``) НЕ исключаются: они как раз штрих, и их надо смотреть.

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class PdfPageMarkup:
    """Разметка одной страницы PDF: прямоугольники растра в пикселях СКАНА плюс поля."""

    scan_width: int
    scan_height: int
    scan_dpi: int
    margin_left_px: float
    margin_top_px: float
    regions: tuple[tuple[int, int, int, int], ...]
    full_page: bool

    def boxes_at(self, dpi: int) -> list[tuple[int, int, int, int]]:
        """Прямоугольники растра в пикселях страницы, отрендеренной в ``dpi``."""
        k = dpi / self.scan_dpi
        left = self.margin_left_px * k
        top = self.margin_top_px * k
        return [
            (int(x1 * k + left), int(y1 * k + top), int(x2 * k + left), int(y2 * k + top))
            for x1, y1, x2, y2 in self.regions
        ]


def load_pdf_markup(db_path: Path, pack_name: str) -> dict[tuple[str, int], PdfPageMarkup]:
    """Разметка пака, разложенная по парам ``(имя PDF, номер страницы в нём)``.

    Полосы без ``full_pdf_page_idx`` (не попавшие в промежуточный PDF) и выпуски без
    имени PDF пропускаются молча: их в обходимой папке всё равно нет.

    Args:
        db_path: Файл базы разметки.
        pack_name: Имя пака в базе.

    Returns:
        Словарь; ключ — ``(имя файла PDF, 0-based номер страницы)``.
    """
    session_factory = open_db(db_path, create=False)
    markup: dict[tuple[str, int], PdfPageMarkup] = {}
    with session_factory() as session:
        pack = require_pack(session, pack_name)
        query = (
            select(Page, Issue)
            .join(Issue, Page.issue_id == Issue.id)
            .join(YearPackage, Issue.year_package_id == YearPackage.id)
            .where(YearPackage.pack_id == pack.id)
        )
        for page, issue in session.execute(query):
            if page.full_pdf_page_idx is None or not issue.full_intermediate_pdf_name:
                continue
            if not page.width or not page.height:
                continue
            dpi = int(page.dpi or 600)
            scale = dpi / MM_PER_INCH
            regions = tuple(
                (int(r.x1), int(r.y1), int(r.x2), int(r.y2)) for r in page.rect_regions if r.kind in RASTER_KINDS
            )
            markup[(issue.full_intermediate_pdf_name, int(page.full_pdf_page_idx))] = PdfPageMarkup(
                scan_width=int(page.width),
                scan_height=int(page.height),
                scan_dpi=dpi,
                margin_left_px=float(issue.full_intermediate_pdf_margin_left_mm or 0.0) * scale,
                margin_top_px=float(issue.full_intermediate_pdf_margin_top_mm or 0.0) * scale,
                regions=regions,
                full_page=any(r.full_page for r in page.rect_regions if r.kind in RASTER_KINDS),
            )
    logger.info("Разметка: %d страниц пака «%s»", len(markup), pack_name)
    return markup
