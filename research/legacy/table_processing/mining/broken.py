"""Таблицы, которые FineReader не увидел: есть на скане, нет в DOCX.

ЗАЧЕМ. Вторая болезнь после боковых шапок. FineReader иногда вовсе не понимает, что перед
ним таблица, и раскладывает её в абзацы. Замер по шести выпускам 1966 года: детектор нашёл
таблицу на 74 страницах, на 25 из них таблицы в DOCX нет, и 15 из этих 25 находок —
настоящие таблицы (остальные были шумом детектора, который с тех пор починен).

МУСОР В ТЕКСТЕ — КОЛОНКА ДЛЯ РАНЖИРОВАНИЯ, А НЕ УСЛОВИЕ ОТБОРА. Это замерено и важно:
доля мусорных токенов на этих 25 страницах 0.00-0.14 при медиане по паку 0.026 и p99 0.158.
Не увидев таблицу, FineReader обычно раскладывает её в аккуратные на вид абзацы, а не в
мешанину. Отбор по мусору потерял бы как раз то, что ищем.

ЧТО ИСКЛЮЧАЕТСЯ И СЧИТАЕТСЯ ОТДЕЛЬНО. Полосы, напечатанные боком (``pages.rotate_cw``),
и таблицы с боковым текстом в шапке: это отдельная задача, уже решённая в другом месте
пакета, и мешать её сюда незачем.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_type_hints

from research.legacy.table_processing import paths as default_paths
from research.legacy.table_processing.detection import ruling, ruling_v4
from research.legacy.table_processing.geometry import Box
from research.legacy.table_processing.imaging import load_gray
from research.legacy.table_processing.mining.docx_tables import iter_tables
from research.legacy.table_processing.mining.garbage import TOKEN, token_is_garbage
from research.legacy.table_processing.mining.page_match import match, read_pages
from research.legacy.table_processing.pages import load_issue_pages, resolve_scan, scan_files

logger = logging.getLogger(__name__)

# Разрешение поиска таблиц на полосе — то же, что у детектора.
DETECT_DPI = ruling.WORK_DPI

# Меньше этого числа токенов на странице — доля мусора считается по шуму.
MIN_TOKENS = 30

# Доля ячеек шапки с боковым текстом, начиная с которой таблица считается «с поворотом» и
# уходит в отдельную категорию.
ROTATED_SHARE = 0.3


@dataclass
class BrokenTable:
    """Находка на странице, где в DOCX таблицы нет."""

    issue: str
    page_number: int
    scan_rel_path: str
    x0: int
    y0: int
    x1: int
    y1: int
    dpi: int
    garbage_share: float
    tokens: int
    rotated_cells: int
    page_rotate_cw: int
    category: str  # "пропущена" | "боковая страница" | "боковой текст"

    @property
    def box(self) -> Box:
        return Box(self.x0, self.y0, self.x1, self.y1)


FIELDS = list(BrokenTable.__dataclass_fields__)


def page_garbage(text: str) -> tuple[float, int]:
    """Доля мусорных токенов на странице и сколько их всего."""
    tokens = TOKEN.findall(text)
    if len(tokens) < MIN_TOKENS:
        return 0.0, len(tokens)
    return sum(1 for token in tokens if token_is_garbage(token)) / len(tokens), len(tokens)


def rotated_cells(gray, box: Box, dpi: int) -> int:
    """Сколько ячеек находки набрано боком — по самой дешёвой мере (форма буквы)."""
    from research.legacy.table_processing.rotation.base import CellCrop
    from research.legacy.table_processing.rotation.glyph_aspect import detect as glyph_detect
    from research.legacy.table_processing.structure.ruling_grid import cell_image, extract

    crop = gray[box.clipped(gray.shape[1], gray.shape[0]).slice]
    if crop.size == 0:
        return 0
    grid = extract(crop, dpi)
    rotated = 0
    for cell in grid.cells:
        verdict = glyph_detect(CellCrop("", cell, cell_image(crop, cell, dpi), dpi))
        rotated += int(verdict.rotate_cw != 0 and verdict.confidence > 0.0)
    return rotated


def find_issue(
    issue: str,
    docx_dir: Path = default_paths.DOCX_DIR,
    pdf_dir: Path = default_paths.RECOGNIZED_PDF_DIR,
    sharpened_dir: Path = default_paths.SHARPENED_DIR,
    db_path: Path = default_paths.MARKUP_DB,
    pack_name: str = default_paths.PACK_NAME,
    source_dpi: int = default_paths.SOURCE_DPI,
) -> list[BrokenTable]:
    """Все находки выпуска на страницах, где в DOCX таблицы нет."""
    import fitz

    docx_path = docx_dir / f"full_{issue}.docx"
    pdf_path = pdf_dir / f"full_{issue}.pdf"
    if not docx_path.is_file() or not pdf_path.is_file():
        return []

    page_texts = read_pages(pdf_path)
    with_table: set[int] = set()
    for table in iter_tables(docx_path, issue):
        found = match(table, page_texts)
        if found.found:
            with_table.add(found.page_number)

    try:
        db_pages = load_issue_pages(db_path, pack_name, issue)
    except Exception as error:
        logger.warning("База недоступна (%s), поворот полос неизвестен", error)
        db_pages = {}

    with fitz.open(pdf_path) as document:
        texts = [document[index].get_text() for index in range(document.page_count)]

    scans = scan_files(sharpened_dir, issue)
    found: list[BrokenTable] = []
    for page_number in range(1, len(scans) + 1):
        if page_number in with_table:
            continue
        scan = resolve_scan(sharpened_dir, issue, page_number, db_pages)
        if scan is None:
            continue
        gray = load_gray(scan, DETECT_DPI, source_dpi)
        tables = [table for table in ruling_v4.detect(gray, DETECT_DPI) if table.is_table]
        if not tables:
            continue
        share, tokens = page_garbage(texts[page_number - 1] if page_number <= len(texts) else "")
        rotate_cw = db_pages[page_number].rotate_cw if page_number in db_pages else 0
        for table in tables:
            rotated = rotated_cells(gray, table.box, DETECT_DPI)
            cells_total = max(1, int(table.metrics.get("cells", 1)))
            if rotate_cw:
                category = "боковая страница"
            elif rotated >= ROTATED_SHARE * cells_total:
                category = "боковой текст"
            else:
                category = "пропущена"
            found.append(
                BrokenTable(
                    issue=issue,
                    page_number=page_number,
                    scan_rel_path=str(scan.relative_to(sharpened_dir)),
                    x0=table.box.x0,
                    y0=table.box.y0,
                    x1=table.box.x1,
                    y1=table.box.y1,
                    dpi=DETECT_DPI,
                    garbage_share=round(share, 4),
                    tokens=tokens,
                    rotated_cells=rotated,
                    page_rotate_cw=rotate_cw,
                    category=category,
                )
            )
    return found


def write_csv(records: list[BrokenTable], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (-item.garbage_share, item.issue, item.page_number)):
            writer.writerow(asdict(record))


def read_csv(path: Path) -> list[BrokenTable]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Типы берутся через ``get_type_hints``, а не из ``field.type``: с
    # ``from __future__ import annotations`` там лежит СТРОКА «int», и проверка на тип
    # молча не срабатывала — все поля оставались строками, а деление на них падало.
    hints = get_type_hints(BrokenTable)
    records: list[BrokenTable] = []
    for row in rows:
        typed = {}
        for name in BrokenTable.__dataclass_fields__:
            value = row.get(name, "")
            kind = hints.get(name, str)
            typed[name] = kind(value) if kind in (int, float) else value
        records.append(BrokenTable(**typed))
    return records
