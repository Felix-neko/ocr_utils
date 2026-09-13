"""Добыча примеров: от DOCX до вырезанной таблицы на диске.

ЦЕПОЧКА. Таблица в DOCX → счёт мешанины → страница распознанного PDF по текстовому слою →
файл скана по базе разметки → рамка от детектора линеек, выбранная по словарю таблицы →
вырезка в 600 dpi. Каждое звено проверяемо по отдельности, и в манифест пишутся все
промежуточные числа: когда вырезка окажется не той таблицей, будет видно, какое звено врёт.

ЧТО ДОРОГО. Разжатие скана: 0.15 с на полосу в 150 dpi и около секунды в 600 dpi. Поэтому
сканы читаются только для таблиц выше порога счёта, и по одному выпуску на воркер —
вперемешку с чтением DOCX и текстового слоя, которые стоят копейки.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_type_hints

import cv2

from research.legacy.table_processing import paths as default_paths
from research.legacy.table_processing.detection import quality, ruling, ruling_v4
from research.legacy.table_processing.detection.refine import push_edges
from research.legacy.table_processing.geometry import Box
from research.legacy.table_processing.imaging import load_gray
from research.legacy.table_processing.mining import anchor as anchor_module
from research.legacy.table_processing.mining.docx_tables import iter_tables
from research.legacy.table_processing.mining.garbage import score as garbage_score
from research.legacy.table_processing.mining.page_match import match, read_pages
from research.legacy.table_processing.pages import issue_from_name, load_issue_pages, resolve_scan
from research.legacy.table_processing.structure.ruling_grid import text_ink

logger = logging.getLogger(__name__)

# Поля вокруг вырезанной таблицы: 3 мм. Меньше — и обрезается внешняя линейка, которой
# у этих таблиц слева и справа часто нет вовсе; больше — и в кадр лезет соседний абзац.
CROP_PAD_MM = 3.0

# Разрешение, в котором ищутся линейки на полосе (координаты потом умножаются).
DETECT_DPI = ruling.WORK_DPI


@dataclass
class MinedTable:
    """Строка манифеста: одна таблица со всеми числами, по которым её нашли."""

    crop_id: str
    issue: str
    table_index: int
    rank: float
    garbage_share: float
    header_share: float
    symbol_share: float
    garbage_cells: int
    n_rows: int
    n_cols: int
    page_number: int
    match_score: float
    match_margin: float
    match_source: str
    scan_rel_path: str
    coverage: float
    x0: int
    y0: int
    x1: int
    y1: int
    skew_deg: float
    curved: int
    crop_file: str

    @property
    def box(self) -> Box:
        return Box(self.x0, self.y0, self.x1, self.y1)


FIELDS = list(MinedTable.__dataclass_fields__)


def mine_issue(
    issue: str,
    min_rank: float,
    docx_dir: Path = default_paths.DOCX_DIR,
    pdf_dir: Path = default_paths.RECOGNIZED_PDF_DIR,
    sharpened_dir: Path = default_paths.SHARPENED_DIR,
    db_path: Path = default_paths.MARKUP_DB,
    pack_name: str = default_paths.PACK_NAME,
    source_dpi: int = default_paths.SOURCE_DPI,
) -> list[MinedTable]:
    """Все таблицы выпуска, у которых счёт мешанины не ниже порога, уже с рамками."""
    docx_path = docx_dir / f"full_{issue}.docx"
    pdf_path = pdf_dir / f"full_{issue}.pdf"
    if not docx_path.is_file() or not pdf_path.is_file():
        logger.warning("Выпуск %s пропущен: нет %s или %s", issue, docx_path.name, pdf_path.name)
        return []

    tables = iter_tables(docx_path, issue)
    interesting = [(table, garbage_score(table)) for table in tables]
    interesting = [(table, score) for table, score in interesting if score.rank >= min_rank]
    if not interesting:
        return []

    page_texts = read_pages(pdf_path)
    try:
        db_pages = load_issue_pages(db_path, pack_name, issue)
    except Exception as error:  # база может быть занята или отсутствовать — не повод падать
        logger.warning("База разметки недоступна (%s), номера страниц из листинга", error)
        db_pages = {}

    scale = source_dpi / DETECT_DPI
    found: list[MinedTable] = []
    cached_page: dict[int, tuple[list, list]] = {}
    for table, score in interesting:
        matched = match(table, page_texts)
        scan = resolve_scan(sharpened_dir, issue, matched.page_number, db_pages) if matched.found else None
        if scan is None:
            logger.warning("%s таблица %d: страница не нашлась", issue, table.index)
            continue

        if matched.page_number not in cached_page:
            gray = load_gray(scan, DETECT_DPI, source_dpi)
            margin = db_pages[matched.page_number].margin_mm if matched.page_number in db_pages else (0, 0, 0, 0)
            words = anchor_module.page_words(pdf_path, matched.page_number, margin, DETECT_DPI)
            # Имя не ``found``: так зовётся накопитель манифеста выше, и его перезапись роняла
            # весь выпуск на первой же странице со второй таблицей.
            page_tables = [table for table in ruling_v4.detect(gray, DETECT_DPI) if table.is_table]
            cached_page[matched.page_number] = (page_tables, words)
        boxes, words = cached_page[matched.page_number]

        picked, coverage = anchor_module.pick_table(boxes, table, words)
        if picked is None:
            logger.warning("%s таблица %d: рамка не выбрана (покрытие %.2f)", issue, table.index, coverage)
            continue

        box = picked.box.scaled(scale)
        found.append(
            MinedTable(
                crop_id=f"{issue}_t{table.index:02d}",
                issue=issue,
                table_index=table.index,
                rank=round(score.rank, 4),
                garbage_share=round(score.garbage_share, 4),
                header_share=round(score.header_share, 4),
                symbol_share=round(score.symbol_share, 4),
                garbage_cells=score.garbage_cells,
                n_rows=table.n_rows,
                n_cols=table.n_cols,
                page_number=matched.page_number,
                match_score=round(matched.score, 3),
                match_margin=round(matched.margin, 3),
                match_source=matched.source,
                scan_rel_path=str(scan.relative_to(sharpened_dir)),
                coverage=round(coverage, 3),
                x0=box.x0,
                y0=box.y0,
                x1=box.x1,
                y1=box.y1,
                skew_deg=round(picked.skew_deg, 3),
                curved=int(picked.metrics.get("curved", 0.0)),
                crop_file=f"{issue}_t{table.index:02d}.png",
            )
        )
    return found


def export_crop(record: MinedTable, sharpened_dir: Path, out_dir: Path, source_dpi: int) -> str:
    """Вырезать таблицу в исходном разрешении. Возвращает ошибку словами или пустую строку."""
    scan = sharpened_dir / record.scan_rel_path
    try:
        gray = load_gray(scan, source_dpi, source_dpi)
    except OSError as error:
        return f"{record.crop_id}: {error}"
    # Поле — МАКСИМУМ, а не обязательство. Слепое ``padded(pad)`` сводило на нет всю доводку
    # рамки: детектор аккуратно ставит границу по чистому просвету, а экспорт снова уезжает
    # на три миллиметра и разрезает буквы, только уже в другом месте. Поэтому после полей
    # граница ещё раз обходит компоненты краски (``push_edges`` четвёртой версии).
    #
    # Обход считается на уменьшенной копии 150 dpi, а не на исходных 600: морфология по
    # линейкам стоит шестнадцатикратно дороже на полном разрешении, а граница таблицы не
    # нужна точнее полумиллиметра (см. докстринг ``detection.ruling``).
    pad = ruling.mm_to_px(CROP_PAD_MM, source_dpi)
    padded = record.box.padded(pad).clipped(gray.shape[1], gray.shape[0])
    scale = DETECT_DPI / source_dpi
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = ruling.find_lines(small, DETECT_DPI)
    ink = text_ink(small, lines)
    pushed = push_edges(
        padded.scaled(scale).clipped(small.shape[1], small.shape[0]),
        quality.glyph_components(ink),
        lines,
        DETECT_DPI,
        small.shape[:2],
        ink,
    )
    box = pushed.box.scaled(1 / scale).clipped(gray.shape[1], gray.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_dir / record.crop_file), gray[box.slice]):
        return f"{record.crop_id}: не записалась вырезка"
    return ""


def write_manifest(records: list[MinedTable], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (-item.rank, item.crop_id)):
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> list[MinedTable]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Типы берутся через ``get_type_hints``, а не из ``field.type``: с
    # ``from __future__ import annotations`` там лежит СТРОКА «int», и проверка на тип
    # молча не срабатывала — все поля оставались строками, а деление на них падало.
    hints = get_type_hints(MinedTable)
    records: list[MinedTable] = []
    for row in rows:
        typed = {}
        for name in MinedTable.__dataclass_fields__:
            value = row.get(name, "")
            kind = hints.get(name, str)
            typed[name] = kind(value) if kind in (int, float) else value
        records.append(MinedTable(**typed))
    return records


def issues_in(docx_dir: Path) -> list[str]:
    """Выпуски, для которых есть выгрузка DOCX. Временные файлы Word (``~$``) пропускаются."""
    names = [path.name for path in sorted(docx_dir.glob("full_*.docx")) if not path.name.startswith("~$")]
    return [issue_from_name(name) for name in names]
