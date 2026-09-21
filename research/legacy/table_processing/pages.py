"""Связь «таблица в DOCX → страница PDF → файл скана».

ТРИ КООРДИНАТЫ ОДНОГО И ТОГО ЖЕ. Выгрузка FineReader — это DOCX, вход FineReader — это
промежуточный PDF, а исходник — JPEG скана. Чтобы вырезать таблицу из скана, надо пройти
всю цепочку.

НОМЕР СТРАНИЦЫ DOCX НЕ ГОДИТСЯ — проверено на всех 98 выгруженных выпусках. FineReader не
пишет ``lastRenderedPageBreak``, и счётчик по разрывам страниц и несплошным секциям
расходится с числом страниц PDF на ±1 у большинства выпусков (например, ``full_1966_03``:
98 против 97). Опорные метки ``w:pgNumType/@w:start`` в тех же файлах не совпадают с этим
счётчиком уже на второй странице. Поэтому счётчик используется ТОЛЬКО как окно поиска,
а страницу называет текстовый слой распознанного PDF (см. ``mining/page_match``).

СТРАНИЦА PDF → СКАН — из базы разметки, ровно как в
``ocr_utils.page_layout.orientation.pdf_pages``: страница N (с единицы) это ``order_index``
N-1. Проверено запросом по всем 12135 полосам: ``full_pdf_page_idx == order_index`` везде.
База открывается ТОЛЬКО НА ЧТЕНИЕ.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanPage:
    """Полоса выпуска: чем она была в PDF и где лежит её заострённая копия."""

    rel_path: str  # {год}/{выпуск}/IMG_xxxx_1L.jpg — относительно SHARPENED_DIR
    page_number: int  # с единицы, как считает человек и как считает FineReader
    rotate_cw: int  # поворот полосы из базы; ненулевой означает «полоса напечатана боком»
    margin_mm: tuple[float, float, float, float]  # left, right, top, bottom полей промежуточного PDF


def load_issue_pages(db_path: Path, pack_name: str, issue: str) -> dict[int, ScanPage]:
    """Страницы одного выпуска: ``{номер страницы PDF: полоса}``.

    ``issue`` — как в имени файла DOCX, «1966_01».
    """
    year, number = issue.split("_", 1)
    query = """
        SELECT p.full_pdf_page_idx,
               p.sharpened_text_pic_rel_path,
               p.source_rel_path,
               COALESCE(p.rotate_cw, 0),
               COALESCE(i.full_intermediate_pdf_margin_left_mm, 0),
               COALESCE(i.full_intermediate_pdf_margin_right_mm, 0),
               COALESCE(i.full_intermediate_pdf_margin_top_mm, 0),
               COALESCE(i.full_intermediate_pdf_margin_bottom_mm, 0)
          FROM pages p
          JOIN issues i ON p.issue_id = i.id
          JOIN year_packages y ON i.year_package_id = y.id
          JOIN packs k ON y.pack_id = k.id
         WHERE k.name = ? AND y.name = ? AND i.name = ? AND p.full_pdf_page_idx IS NOT NULL
    """
    pages: dict[int, ScanPage] = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        for index, sharpened, source, rotate, left, right, top, bottom in connection.execute(
            query, (pack_name, year, number)
        ):
            rel = sharpened or str(Path(source).with_suffix(".jpg"))
            pages[int(index) + 1] = ScanPage(rel, int(index) + 1, int(rotate), (left, right, top, bottom))
    return pages


def issue_from_name(name: str) -> str:
    """«full_1966_01.docx» или «full_1966_01.pdf» → «1966_01»."""
    match = re.search(r"(\d{4})_(\d{2})", Path(name).stem)
    if not match:
        raise ValueError(f"не разобрать выпуск из имени {name!r}")
    return f"{match.group(1)}_{match.group(2)}"


def scan_files(sharpened_dir: Path, issue: str, ext: str = "jpg") -> list[Path]:
    """Файлы выпуска в том порядке, в каком они лежат в PDF (лексикографическом).

    Запасной путь на случай, когда базы нет: порядок страниц в промежуточном PDF задаётся
    ``order_index``, а он и есть позиция в отсортированном листинге папки — проверено на
    всех 123 выпусках пака.

    ``ext`` — расширение полос в каталоге: у заострённых копий ``jpg``, у исходных сканов
    «Готовое/пак-1» ``tif``; структура папок и основы имён у них одинаковые.
    """
    year, number = issue.split("_", 1)
    return sorted((sharpened_dir / year / number).glob(f"*.{ext}"))


def resolve_scan(
    sharpened_dir: Path, issue: str, page_number: int, pages: dict[int, ScanPage] | None, ext: str = "jpg"
) -> Path | None:
    """Файл скана по номеру страницы PDF: сперва из базы, иначе по позиции в листинге."""
    if pages and page_number in pages:
        candidate = (sharpened_dir / pages[page_number].rel_path).with_suffix(f".{ext}")
        if candidate.is_file():
            return candidate
    files = scan_files(sharpened_dir, issue, ext)
    if 1 <= page_number <= len(files):
        return files[page_number - 1]
    return None
