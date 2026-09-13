"""Сверка детектора таблиц: старая версия, новая и то, что нашёл сам FineReader.

ЗАЧЕМ. Вторая ступень детектора (``detection.verify``) калибровалась на 61 находке и
проверялась на выборке в 396 полос. Этого мало, чтобы спокойно на неё опереться: выборка —
не пак. Здесь всё считается по всем 12 135 полосам и сразу тремя способами, чтобы числа
можно было сложить в одну таблицу и не путать между собой.

ТРИ МНОЖЕСТВА, КОТОРЫЕ ЛЕГКО ПЕРЕПУТАТЬ и которые считаются здесь одновременно:

* таблицы, которые создал в DOCX сам FineReader (их 934) и страницы, где они стоят;
* находки СТАРОГО детектора — всё, что даёт морфология по линейкам без проверки;
* находки НОВОГО — то же самое, прошедшее проверку.

Новый — это старый плюс фильтр, поэтому его находки ВСЕГДА подмножество старых, и сравнение
укладывается в таблицу два на три по полосам. Это же свойство проверяется тестом.

ОДИН ПРОХОД. Выпуск целиком отдаётся воркеру: он читает DOCX, привязывает таблицы к
страницам по текстовому слою, обходит сканы выпуска, считает обе версии детектора и тут же
пишет debug-картинки. Так полосы читаются с диска по разу, а не трижды.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from research.legacy.table_processing import paths as default_paths
from research.legacy.table_processing.detection import ruling, ruling_v3, ruling_v4, verify
from research.legacy.table_processing.geometry import KIND_DIAGRAM, KIND_DRAWING, KIND_TABLE, Box
from research.legacy.table_processing.layout import surya as layout_module
from research.legacy.table_processing.imaging import load_gray, to_rgb
from research.legacy.table_processing.mining.docx_tables import iter_tables
from research.legacy.table_processing.mining.page_match import match, read_pages
from research.legacy.table_processing.pages import load_issue_pages, resolve_scan, scan_files
from research.legacy.table_processing.report import downscale

logger = logging.getLogger(__name__)

# Разрешение поиска — то же, в котором детектор работает в конвейере.
DETECT_DPI = ruling.WORK_DPI

# Debug-картинки: JPEG с длинной стороной 1100 px. В PNG три набора весили бы под гигабайт,
# а разглядывать рамку таблицы на 1100 px можно без труда.
LONG_SIDE = 1100
JPEG_QUALITY = 78

# Цвета рамок. У старого детектора вердиктов не было, поэтому там все находки одним цветом.
OLD_COLOUR = (0, 0, 220)
NEW_COLOUR = (0, 150, 0)

# Третья версия — своим цветом: её рамки сравнивают с зелёными рамками второй.
V3_COLOUR = (200, 90, 0)

# Четвёртая версия красит рамку по ВИДУ находки: таблица, схема, рисунок (BGR).
KIND_COLOURS = {KIND_TABLE: (0, 150, 0), KIND_DIAGRAM: (200, 60, 0), KIND_DRAWING: (110, 110, 110)}

CATEGORY_AGREE = "согласие"
CATEGORY_LOST = "новый потерял"
CATEGORY_BLIND = "оба не увидели"
CATEGORY_FR_MISS = "пропуск FineReader"
CATEGORY_NOISE = "убранный шум"


@dataclass
class DocxTableRow:
    """Одна таблица из выгрузки DOCX и страница, на которую она легла."""

    issue: str
    index: int
    n_rows: int
    n_cols: int
    page_number: int
    match_score: float
    match_margin: float
    match_source: str
    scan_rel_path: str


@dataclass
class FindingRow:
    """Одна находка детектора со всеми признаками и вердиктом новой ступени."""

    issue: str
    page_number: int
    scan_rel_path: str
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    dpi: int
    verdict: str
    reason: str
    features: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        row = {name: value for name, value in asdict(self).items() if name != "features"}
        row.update(self.features)
        return row


@dataclass
class PageRow:
    """Полоса, на которой есть хоть что-то: таблица в DOCX или находка детектора."""

    issue: str
    page_number: int
    scan_rel_path: str
    docx_tables: int
    old_findings: int
    new_findings: int
    v3_findings: int
    v4_findings: int  # только таблицы четвёртой версии
    v4_kinds: str  # все находки четвёртой версии по видам: «таблица:2,схема:1»
    category: str


DOCX_FIELDS = list(DocxTableRow.__dataclass_fields__)
PAGE_FIELDS = list(PageRow.__dataclass_fields__)
FINDING_FIELDS = [name for name in FindingRow.__dataclass_fields__ if name != "features"] + list(verify.HEADER)


def image_name(issue: str, page_number: int, scan_rel_path: str) -> str:
    """Имя debug-картинки, ОДИНАКОВОЕ во всех трёх наборах.

    Одинаковое намеренно: так одну и ту же полосу можно открыть в трёх папках рядом и
    сравнить. Число найденных таблиц в имя не выносится — оно есть в ``страницы.csv``, а
    из имени сделало бы файлы несопоставимыми.
    """
    return f"{issue}_с{page_number:03d}_{Path(scan_rel_path).stem}.jpg"


def _save(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), downscale(image, LONG_SIDE), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


OVERLAY_DIRS = ("оверлеи_старый", "оверлеи_новый", "оверлеи_v3", "оверлеи_v4", "страницы_finereader")


def forget_issue_overlays(out_dir: Path, issue: str) -> int:
    """Удалить картинки выпуска от прошлого прогона. Возвращает, сколько удалено.

    Без этого папка оверлеев — не результат прогона, а ОБЪЕДИНЕНИЕ всех прогонов: полоса, на
    которой детектор перестал срабатывать, остаётся лежать со старой рамкой. Так и случилось:
    после правила про баннер рубрики 15 полос с баннерами исчезли из ``страницы.csv``, но их
    картинки остались, и человек разложил их по папкам как ошибки действующего детектора.
    """
    removed = 0
    for name in OVERLAY_DIRS:
        folder = out_dir / name
        if not folder.is_dir():
            continue
        for path in folder.glob(f"{issue}_с*.jpg"):
            path.unlink()
            removed += 1
    return removed


def categorize(docx_tables: int, old_findings: int, new_findings: int) -> str:
    """Клетка таблицы сравнения для одной полосы.

    Категория считается по ВТОРОЙ версии: она подмножество первой, и только поэтому
    сравнение укладывается в таблицу два на три. Третья версия это вложение нарушает
    (она находит то, чего не находила вторая), поэтому её счёт живёт отдельной колонкой
    ``v3_findings``, а не новой категорией — иначе таблица перестала бы быть таблицей.
    """
    if docx_tables:
        if new_findings:
            return CATEGORY_AGREE
        return CATEGORY_LOST if old_findings else CATEGORY_BLIND
    if new_findings:
        return CATEGORY_FR_MISS
    return CATEGORY_NOISE


@dataclass(frozen=True)
class Finding:
    """Одна находка кластеризации с признаками и вердиктом проверки."""

    index: int
    box: Box
    signs: verify.Features
    verdict: bool
    reason: str


def findings_of(gray: np.ndarray, dpi: int, rule=verify.is_table) -> list[Finding]:
    """Все находки кластеризации на полосе, каждая с признаками и вердиктом.

    Один и тот же цикл «детектор без проверки → признаки → правило» жил в трёх местах:
    здесь, в ``cli._audit_one`` и в разборе порогов. Три копии одного цикла — это три места,
    где придётся вспомнить про правку, и одно, где про неё забудут.

    ``rule`` подставляется снаружи: у второй и третьей версии правила разные, а признаки
    одни и те же.
    """
    height, width = gray.shape[:2]
    found: list[Finding] = []
    for index, table in enumerate(ruling.detect(gray, dpi, verify_findings=False)):
        box = table.box.clipped(width, height)
        if box.width < 8 or box.height < 8:
            continue
        crop = gray[box.slice]
        signs = verify.features(crop, ruling.find_lines(crop, dpi), dpi)
        good, reason = rule(signs)
        found.append(Finding(index=index, box=box, signs=signs, verdict=good, reason=reason))
    return found


def docx_pages(issue: str, docx_dir: Path, pdf_dir: Path) -> list[DocxTableRow]:
    """Таблицы выпуска из DOCX, привязанные к страницам по текстовому слою.

    Счётчик страниц самого DOCX не используется: FineReader не пишет
    ``lastRenderedPageBreak``, и счётчик расходится с числом страниц PDF на ±1 у
    большинства выпусков. Привязка идёт по редким токенам таблицы.
    """
    docx_path = docx_dir / f"full_{issue}.docx"
    pdf_path = pdf_dir / f"full_{issue}.pdf"
    if not docx_path.is_file() or not pdf_path.is_file():
        logger.warning("Выпуск %s: нет %s или %s", issue, docx_path.name, pdf_path.name)
        return []
    page_texts = read_pages(pdf_path)
    found: list[DocxTableRow] = []
    for table in iter_tables(docx_path, issue):
        matched = match(table, page_texts)
        found.append(
            DocxTableRow(
                issue=issue,
                index=table.index,
                n_rows=table.n_rows,
                n_cols=table.n_cols,
                page_number=matched.page_number,
                match_score=round(matched.score, 3),
                match_margin=round(matched.margin, 3),
                match_source=matched.source,
                scan_rel_path="",
            )
        )
    return found


def check_issue(
    issue: str,
    out_dir: Path,
    docx_dir: Path = default_paths.DOCX_DIR,
    pdf_dir: Path = default_paths.RECOGNIZED_PDF_DIR,
    sharpened_dir: Path = default_paths.SHARPENED_DIR,
    db_path: Path = default_paths.MARKUP_DB,
    pack_name: str = default_paths.PACK_NAME,
    source_dpi: int = default_paths.SOURCE_DPI,
    draw: bool = True,
    layout_dir: "Path | None" = None,
    ext: str = "jpg",
) -> tuple[list[DocxTableRow], list[FindingRow], list[PageRow]]:
    """Весь разбор одного выпуска: таблицы DOCX, три версии детектора, debug-картинки.

    ``layout_dir`` — кэш разметки surya (``layout.surya``); если он есть, четвёртая версия
    получает разметку полосы, если нет — работает на CPU без неё. ``ext`` — расширение полос
    (``tif`` для исходных сканов «Готовое», где кэш surya считан по тем же TIFF).
    """
    try:
        db_pages = load_issue_pages(db_path, pack_name, issue)
    except Exception as error:
        logger.warning("Выпуск %s: база недоступна (%s)", issue, error)
        db_pages = {}

    tables = docx_pages(issue, docx_dir, pdf_dir)
    per_page: dict[int, int] = {}
    for table in tables:
        if table.page_number:
            per_page[table.page_number] = per_page.get(table.page_number, 0) + 1

    scans = scan_files(sharpened_dir, issue, ext)
    if draw:
        forget_issue_overlays(out_dir, issue)
    findings: list[FindingRow] = []
    pages: list[PageRow] = []
    for page_number in range(1, len(scans) + 1):
        scan = resolve_scan(sharpened_dir, issue, page_number, db_pages, ext)
        if scan is None:
            continue
        rel_path = str(scan.relative_to(sharpened_dir))
        for table in tables:
            if table.page_number == page_number and not table.scan_rel_path:
                table.scan_rel_path = rel_path

        gray = load_gray(scan, DETECT_DPI, source_dpi)
        raw = ruling.detect(gray, DETECT_DPI, verify_findings=False)
        third = ruling_v3.detect(gray, DETECT_DPI)
        fourth = ruling_v4.detect(gray, DETECT_DPI, layout=layout_module.load(layout_dir, rel_path))
        fourth_tables = [table for table in fourth if table.is_table]
        accepted: list = []
        for found in findings_of(gray, DETECT_DPI):
            if found.verdict:
                accepted.append(found.box)
            findings.append(
                FindingRow(
                    issue=issue,
                    page_number=page_number,
                    scan_rel_path=rel_path,
                    index=found.index,
                    x0=found.box.x0,
                    y0=found.box.y0,
                    x1=found.box.x1,
                    y1=found.box.y1,
                    dpi=DETECT_DPI,
                    verdict="таблица" if found.verdict else "нет",
                    reason=found.reason,
                    features={key: float(value) for key, value in found.signs.as_row().items()},
                )
            )

        docx_count = per_page.get(page_number, 0)
        if not (docx_count or raw or third or fourth):
            continue
        kinds: dict[str, int] = {}
        for table in fourth:
            kinds[table.kind] = kinds.get(table.kind, 0) + 1
        pages.append(
            PageRow(
                issue=issue,
                page_number=page_number,
                scan_rel_path=rel_path,
                docx_tables=docx_count,
                old_findings=len(raw),
                new_findings=len(accepted),
                v3_findings=len(third),
                v4_findings=len(fourth_tables),
                v4_kinds=",".join(f"{kind}:{count}" for kind, count in sorted(kinds.items())),
                category=categorize(docx_count, len(raw), len(accepted)),
            )
        )

        if not draw:
            continue
        name = image_name(issue, page_number, rel_path)
        if raw:
            canvas = to_rgb(gray)
            for table_box in raw:
                box = table_box.box.clipped(gray.shape[1], gray.shape[0])
                cv2.rectangle(canvas, (box.x0, box.y0), (box.x1, box.y1), OLD_COLOUR, 4)
            _save(canvas, out_dir / "оверлеи_старый" / name)
        if accepted:
            canvas = to_rgb(gray)
            for box in accepted:
                cv2.rectangle(canvas, (box.x0, box.y0), (box.x1, box.y1), NEW_COLOUR, 4)
            _save(canvas, out_dir / "оверлеи_новый" / name)
        if third:
            canvas = to_rgb(gray)
            for table_box in third:
                box = table_box.box.clipped(gray.shape[1], gray.shape[0])
                cv2.rectangle(canvas, (box.x0, box.y0), (box.x1, box.y1), V3_COLOUR, 4)
            _save(canvas, out_dir / "оверлеи_v3" / name)
        if fourth:
            canvas = to_rgb(gray)
            for table_box in fourth:
                box = table_box.box.clipped(gray.shape[1], gray.shape[0])
                cv2.rectangle(canvas, (box.x0, box.y0), (box.x1, box.y1), KIND_COLOURS[table_box.kind], 4)
            _save(canvas, out_dir / "оверлеи_v4" / name)
        if docx_count:
            _save(gray, out_dir / "страницы_finereader" / name)
    return tables, findings, pages


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
