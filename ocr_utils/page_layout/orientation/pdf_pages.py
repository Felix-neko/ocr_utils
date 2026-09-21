"""Номер полосы в промежуточном PDF — из базы разметки.

ЗАЧЕМ. Боль, с которой всё началось, звучит как «FineReader падает на странице 80 файла
full_1967_01.pdf». Чтобы отчёт отвечал на тот же вопрос, в котором задан, находка должна
называться номером страницы PDF, а не только именем файла скана.

База открывается ТОЛЬКО НА ЧТЕНИЕ (``mode=ro``): в ней лежит ручная разметка, которой на
диске больше нигде нет, и детектор ориентации не имеет к ней никакого отношения.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfPage:
    pdf_name: str
    page_number: int  # с единицы, как считает человек и как считает FineReader


def load_pdf_pages(db_path: Path, pack_name: str) -> dict[str, PdfPage]:
    """Отображение «путь скана без расширения» → страница промежуточного PDF.

    Ключ без расширения намеренно: в базе записаны исходные ``.tif``, а детектор обычно
    работает по заострённым ``.jpg`` в такой же раскладке.
    """
    query = """
        SELECT p.source_rel_path, i.full_intermediate_pdf_name, p.full_pdf_page_idx
          FROM pages p
          JOIN issues i ON p.issue_id = i.id
          JOIN year_packages y ON i.year_package_id = y.id
          JOIN packs k ON y.pack_id = k.id
         WHERE k.name = ? AND p.full_pdf_page_idx IS NOT NULL
    """
    mapping: dict[str, PdfPage] = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        for rel_path, pdf_name, page_index in connection.execute(query, (pack_name,)):
            key = str(Path(rel_path).with_suffix(""))
            mapping[key] = PdfPage(pdf_name or "", int(page_index) + 1)
    logger.info("Из базы взято %d соответствий полоса → страница PDF", len(mapping))
    return mapping


def lookup(mapping: dict[str, PdfPage], rel_path: str) -> PdfPage | None:
    return mapping.get(str(Path(rel_path).with_suffix("")))
