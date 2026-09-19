"""Стадия C: сборка финального PDF выпуска по JSON анализа и сверка по сохранённому файлу.

Порядок правок страницы важен: (1) страница копируется из выбранного источника в новый
документ; (2) правится текстовый слой — это требует ОДНОГО потока содержимого, каким его
оставляет FineReader; (3) снимаются образы-фигуры FineReader под иллюстрациями; (4) поверх
кладутся JPEG иллюстраций (каждая — свой поток). Сверка идёт по сохранённому файлу: то, что
увидит просмотрщик, а не память MuPDF.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz
import pikepdf

from ocr_utils.final_pdfs.analysis import AnalysisParams, load_analysis, page_json_path
from ocr_utils.final_pdfs.pictures import (
    DEFAULT_DESCREEN_SIGMA_MM,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_PICTURE_DPI,
    PictureJpeg,
    crop_page,
    figures_under,
    insert_picture,
    placement_rect,
    remove_images,
    render_pictures,
)
from ocr_utils.final_pdfs.plan import IssuePlan, PageSource, final_pdf_path
from ocr_utils.final_pdfs.sources import IssuePair, Margins, verify_page_geometry
from ocr_utils.text_layer_fix.fixer import PageEdits, apply_page_edits, has_edits, plan_page_edits, verify_saved
from ocr_utils.text_layer_fix.rewrite import DEFAULT_FONT_PATH, InsertFont

logger = logging.getLogger(__name__)

# Допуск на место иллюстрации при сверке, pt (округления матрицы cm).
RECT_TOLERANCE_PT = 1.0


@dataclass(frozen=True)
class AssembleParams:
    """Параметры сборки выпуска (пиклуются в воркер)."""

    analysis: AnalysisParams
    out_dir: Path
    pictures_dir: Path  # очищенные полосы (blurred/), раскладка {год}/{выпуск}/полоса.tif
    margins: Margins
    picture_dpi: int = DEFAULT_PICTURE_DPI
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    descreen_sigma_mm: float = DEFAULT_DESCREEN_SIGMA_MM
    font_path: Path = DEFAULT_FONT_PATH
    skip_done: bool = True
    preview_pages: int = 0  # сколько собранных страниц с иллюстрациями отрендерить для глаз


@dataclass
class PageRecord:
    """Что сделано с одной страницей финального PDF."""

    pdf: str
    page: int  # с нуля
    source: str = ""
    reason: str = ""
    geometry_verdict: str = ""
    pictures: int = 0
    figures_removed: int = 0
    blanked: int = 0
    trimmed: int = 0
    inserted: int = 0
    refitted: int = 0  # слов слоя ужато в границу обрезанной страницы
    skipped_duplicates: int = 0
    cropped: bool = False  # страница обрезана до полосы (полностраничный растр)
    verify_ok: bool = True
    verify_notes: str = ""
    error: str = ""

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class IssueResult:
    """Итог сборки выпуска."""

    pdf: str
    out_path: str = ""
    status: str = "ok"  # ok | skipped | error
    reason: str = ""
    pages: int = 0
    pages_geo: int = 0
    pages_nogeo: int = 0
    pictures: int = 0
    figures_removed: int = 0
    cropped: int = 0
    blanked: int = 0
    inserted: int = 0
    refitted: int = 0
    verify_failures: int = 0
    page_errors: int = 0
    bytes_in_geo: int = 0
    bytes_out: int = 0
    seconds: float = 0.0
    records: list[PageRecord] = field(default_factory=list)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("records")
        return row


@dataclass
class _PageWork:
    """Промежуточное состояние страницы между правкой и сверкой."""

    index: int
    source: PageSource
    edits: PageEdits | None = None
    image_removed: bool = False  # основной образ снят (полностраничный растр)
    crop: fitz.Rect | None = None  # до чего обрезана страница (координаты исходной страницы)
    pictures: list[tuple[PictureJpeg, fitz.Rect]] = field(default_factory=list)


def assemble_issue(plan: IssuePlan, pair: IssuePair, params: AssembleParams) -> IssueResult:
    """Собрать финальный PDF выпуска.

    Args:
        plan: Выпуск с полосами и разметкой.
        pair: Пара PDF источников.
        params: Параметры.

    Returns:
        :class:`IssueResult` с записями по страницам; при ошибке выпуска — ``status='error'`` и
        ничего не записано.
    """
    started = time.time()
    out_path = final_pdf_path(params.out_dir, plan)
    result = IssueResult(plan.full_pdf_name, str(out_path), pages=pair.pages, bytes_in_geo=pair.geo.stat().st_size)
    if params.skip_done and out_path.is_file() and _page_count(out_path) == pair.pages:
        result.status, result.reason = "skipped", "уже собран"
        result.bytes_out = out_path.stat().st_size
        return result
    try:
        works = _build(plan, pair, params, out_path, result)
        _verify(plan, pair, params, out_path, works, result)
        if params.preview_pages:
            _preview(params, plan, out_path, works)
    except Exception as error:  # noqa: BLE001 — выпуск не должен валить прогон
        logger.exception("%s: сборка не удалась", plan.full_pdf_name)
        result.status, result.reason = "error", f"{type(error).__name__}: {error}"
        if out_path.is_file():
            out_path.unlink()
    result.seconds = round(time.time() - started, 1)
    return result


def _page_count(path: Path) -> int:
    with fitz.open(str(path)) as doc:
        return len(doc)


def _build(
    plan: IssuePlan, pair: IssuePair, params: AssembleParams, out_path: Path, result: IssueResult
) -> list[_PageWork]:
    """Собрать документ в памяти и сохранить; вернуть состояние страниц для сверки."""
    font = InsertFont(params.font_path)
    works: list[_PageWork] = []
    with (
        fitz.open(str(pair.geo)) as geo,
        fitz.open(str(pair.nogeo)) as nogeo,
        pikepdf.open(str(pair.geo)) as geo_pk,
        pikepdf.open(str(pair.nogeo)) as nogeo_pk,
    ):
        docs = {PageSource.GEO: (geo, geo_pk), PageSource.NOGEO: (nogeo, nogeo_pk)}
        out = fitz.open()
        for index in range(pair.pages):
            payload = load_analysis(page_json_path(params.analysis, plan, index))
            if payload is None:
                raise RuntimeError(f"с.{index + 1}: нет JSON анализа текущей версии — сначала стадия анализа")
            source = PageSource(payload["source"])
            record = PageRecord(
                plan.full_pdf_name,
                index,
                source.value,
                payload.get("reason", ""),
                payload.get("geometry_verdict", ""),
                pictures=len(plan.pages[index].pictures),
            )
            source_doc, source_pk = docs[source]
            out.insert_pdf(source_doc, from_page=index, to_page=index)
            out_page = out[index]
            work = _PageWork(index, source)
            page_plan = plan.pages[index]
            # 0. Иллюстрации готовятся заранее: полностраничная означает обрезку страницы до полосы,
            #    а обрезку должен знать план правок слоя (слова за границей ужимаются внутрь).
            pictures: list[PictureJpeg] = []
            rects: list[fitz.Rect] = []
            if page_plan.pictures:
                if source is not PageSource.NOGEO:
                    raise RuntimeError(f"с.{index + 1}: растр, а источник {source.value}")
                verify_page_geometry(nogeo[index], page_plan, params.margins)
                pictures = render_pictures(
                    params.pictures_dir / page_plan.original_rel_path,
                    page_plan,
                    params.picture_dpi,
                    params.jpeg_quality,
                    params.descreen_sigma_mm,
                )
                rects = [placement_rect(p.rect_px, params.margins, page_plan.dpi) for p in pictures]
                if any(p.full_page for p in pictures):
                    work.crop = placement_rect((0, 0, *page_plan.file_size), params.margins, page_plan.dpi)
            # 1. Текстовый слой — на нетронутой странице-источнике план, на копии применение.
            if not payload.get("error") and (has_edits(payload) or work.crop is not None):
                try:
                    work.edits = plan_page_edits(source_doc[index], source_pk, payload, work.crop)
                    fix = apply_page_edits(out, index, work.edits, font)
                    record.blanked, record.trimmed, record.refitted = fix.blanked, fix.trimmed, fix.refitted
                    record.inserted, record.skipped_duplicates = fix.inserted, fix.skipped_duplicates
                except Exception as error:  # noqa: BLE001 — слой не правим, страница остаётся как у FineReader
                    logger.warning("%s с.%d: слой не исправлен: %s", plan.full_pdf_name, index + 1, error)
                    record.error = f"слой: {type(error).__name__}: {error}"
                    work.edits = None
            elif payload.get("error"):
                record.error = f"анализ: {payload['error']}"
            # 2–4. Иллюстрации: снятие фигур FineReader, вставка JPEG, обрезка обложки до полосы.
            if pictures:
                full_page = work.crop is not None
                xrefs = figures_under(out_page, rects, full_page)
                record.figures_removed = remove_images(out, out_page, xrefs)
                work.image_removed = full_page
                for picture, rect in zip(pictures, rects):
                    insert_picture(out, out_page, picture, rect)
                    work.pictures.append((picture, rect))
                if work.crop is not None:
                    crop_page(out_page, work.crop)
                    record.cropped = True
            works.append(work)
            result.records.append(record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".part.pdf")
        out.save(str(tmp), garbage=4, deflate=True)
        out.close()
        tmp.replace(out_path)
    return works


def _verify(
    plan: IssuePlan,
    pair: IssuePair,
    params: AssembleParams,
    out_path: Path,
    works: list[_PageWork],
    result: IssueResult,
) -> None:
    """Сверить сохранённый файл: слой по плану, иллюстрации на месте, число страниц."""
    font = InsertFont(params.font_path)
    with fitz.open(str(out_path)) as saved, fitz.open(str(pair.geo)) as geo, fitz.open(str(pair.nogeo)) as nogeo:
        if len(saved) != pair.pages:
            raise RuntimeError(f"в сохранённом файле {len(saved)} страниц вместо {pair.pages}")
        docs = {PageSource.GEO: geo, PageSource.NOGEO: nogeo}
        for work, record in zip(works, result.records):
            saved_page = saved[work.index]
            notes: list[str] = []
            if work.edits is not None:
                main_xref = None if work.image_removed else _main_image_xref(saved_page)
                report = verify_saved(docs[work.source][work.index], saved_page, work.edits, main_xref, font, work.crop)
                if not report.ok:
                    notes.append(
                        f"слой: не на месте {report.kept_missing}, не удалено {report.deleted_remaining}, "
                        f"не найдено вставок {report.inserts_missing}, ужатых {report.refits_missing}, "
                        f"образ изменён {report.image_changed}"
                    )
                notes.extend(report.notes)
            notes.extend(_verify_pictures(saved, saved_page, work))
            record.verify_ok = not notes
            record.verify_notes = "; ".join(notes)
            if notes:
                result.verify_failures += 1
                logger.warning("%s с.%d: сверка: %s", plan.full_pdf_name, work.index + 1, record.verify_notes)
            if record.error:
                result.page_errors += 1
            result.pictures += len(work.pictures)
            result.figures_removed += record.figures_removed
            result.cropped += int(record.cropped)
            result.blanked += record.blanked
            result.inserted += record.inserted
            result.refitted += record.refitted
            if work.source is PageSource.GEO:
                result.pages_geo += 1
            else:
                result.pages_nogeo += 1
    result.bytes_out = out_path.stat().st_size


def _main_image_xref(page: fitz.Page) -> int | None:
    """xref самого большого образа страницы (основной JBIG2)."""
    infos = page.get_image_info(xrefs=True)
    if not infos:
        return None
    return int(max(infos, key=lambda info: info["width"] * info["height"])["xref"])


def _verify_pictures(doc: fitz.Document, page: fitz.Page, work: _PageWork) -> list[str]:
    """Каждая иллюстрация — DCT-образ нужной цветности на своём месте; снятые образы отсутствуют."""
    notes: list[str] = []
    infos = page.get_image_info(xrefs=True)
    shift = (-work.crop.x0, -work.crop.y0, -work.crop.x0, -work.crop.y0) if work.crop is not None else (0, 0, 0, 0)
    if work.crop is not None and (
        abs(page.rect.width - work.crop.width) > RECT_TOLERANCE_PT
        or abs(page.rect.height - work.crop.height) > RECT_TOLERANCE_PT
    ):
        notes.append(
            f"страница обрезана до {tuple(round(v) for v in page.rect)}, ожидалось {work.crop.width:.0f}×{work.crop.height:.0f}"
        )
    for picture, rect in work.pictures:
        rect = rect + shift
        found = False
        for info in infos:
            box = fitz.Rect(info["bbox"])
            if (info["width"], info["height"]) != (picture.width, picture.height):
                continue
            if abs(box.x0 - rect.x0) > RECT_TOLERANCE_PT or abs(box.y0 - rect.y0) > RECT_TOLERANCE_PT:
                continue
            if abs(box.x1 - rect.x1) > RECT_TOLERANCE_PT or abs(box.y1 - rect.y1) > RECT_TOLERANCE_PT:
                continue
            xref = int(info["xref"])
            filt = doc.xref_get_key(xref, "Filter")[1]
            expected_cs = "DeviceGray" if picture.gray else "DeviceRGB"
            if "DCTDecode" not in filt:
                notes.append(f"иллюстрация {picture.kind}: фильтр {filt}, не JPEG")
            elif info.get("cs-name") != expected_cs:
                notes.append(f"иллюстрация {picture.kind}: пространство {info.get('cs-name')} вместо {expected_cs}")
            found = True
            break
        if not found:
            notes.append(
                f"иллюстрация {picture.kind} {picture.width}×{picture.height} не найдена в {tuple(round(v) for v in rect)}"
            )
    if work.image_removed and any("JBIG2" in doc.xref_get_key(int(info["xref"]), "Filter")[1] for info in infos):
        notes.append("полностраничный растр, а бинарный образ остался")
    return notes


def _preview(params: AssembleParams, plan: IssuePlan, out_path: Path, works: list[_PageWork]) -> None:
    """Отрендерить первые N собранных страниц с иллюстрациями в ``<work_dir>/preview`` для глаз."""
    preview_dir = params.analysis.work_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(out_path)) as doc:
        done = 0
        for work in works:
            if not work.pictures:
                continue
            pix = doc[work.index].get_pixmap(dpi=100)
            pix.save(str(preview_dir / f"{out_path.stem}_p{work.index + 1:03d}.png"))
            done += 1
            if done >= params.preview_pages:
                break


PAGE_FIELDS = list(PageRecord.__dataclass_fields__)
ISSUE_FIELDS = [f for f in IssueResult.__dataclass_fields__ if f != "records"]


def merge_csv(path: Path, fields: list[str], rows: list[dict], key: tuple[str, ...]) -> None:
    """Записать CSV, сохранив строки прошлых запусков по другим выпускам: прогон по одному выпуску
    (``--only-issue``) не должен стирать сводку по паку. Строки с теми же ключами заменяются."""
    old: dict[tuple, dict] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                old[tuple(row.get(k, "") for k in key)] = row
    for row in rows:
        old[tuple(str(row[k]) for k in key)] = {k: row.get(k, "") for k in fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for _, row in sorted(old.items()):
            writer.writerow({k: row.get(k, "") for k in fields})


def write_pages_csv(path: Path, results: list[IssueResult]) -> None:
    """``pages.csv``: строка на страницу; выпуски прошлых запусков сохраняются, пропущенные (``skipped``) не трогаются."""
    rows = [record.to_row() for result in results for record in result.records]
    merge_csv(path, PAGE_FIELDS, rows, ("pdf", "page"))


def write_summary_csv(path: Path, results: list[IssueResult]) -> None:
    """``summary.csv``: строка на выпуск; выпуски прошлых запусков сохраняются."""
    merge_csv(path, ISSUE_FIELDS, [r.to_row() for r in results if r.status != "skipped"], ("pdf",))
