"""Синтетический пак под сборку финальных PDF: база, очищенные полосы, пара PDF «geo/nogeo».

Настоящие страницы FineReader в тестах не нужны: проверяется наша половина — выбор источника,
место иллюстраций, снятие образов, правка слоя. Полосы крошечные, страницы PDF строятся
«в манере FineReader»: один поток, картинка страницы + образ-фигура под врезкой.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image

from ocr_utils.db.models import KIND_COLOR, KIND_GRAYSCALE, SOURCE_CVAT, RectRegion
from ocr_utils.db.repo import upsert_pack
from ocr_utils.db.session import open_db
from ocr_utils.final_pdfs.plan import load_plans
from ocr_utils.final_pdfs.sources import IssuePair, Margins
from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear
from tests.ocr_utils.text_layer_fix.synthetic import finereader_like_pdf

PACK_NAME = "пак-тест"
W, H = 400, 600
DPI = 600
MARGINS = Margins(32, 16)

# Цвета, по которым узнаётся источник пикселя: полоса — красная, страница FineReader — серая.
PAGE_COLOR = (200, 30, 30)
INSET = (100, 200, 300, 400)

# Полосы выпуска: имя -> (иллюстрации, rotate_cw).
PAGES = {
    "0010.tif": ((), 0),
    "0020.tif": (((*INSET, KIND_GRAYSCALE),), 0),  # серая врезка
    "0030.tif": (((0, 0, W, H, KIND_COLOR),), 0),  # цветная во весь кадр
    "0040.tif": (((*INSET, KIND_GRAYSCALE),), 90),  # полоса повёрнута на 90° по часовой
}


def _write_tiff(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0], 3), np.uint8)
    array[:] = PAGE_COLOR
    # Внутри врезки — градиент, чтобы серый JPEG отличался от заливки.
    array[INSET[1] : INSET[3], INSET[0] : INSET[2], :] = np.linspace(0, 255, INSET[2] - INSET[0], dtype=np.uint8)[
        None, :, None
    ]
    Image.fromarray(array).save(path, dpi=(DPI, DPI))


def _fine_page(
    doc: fitz.Document, size_px: tuple[int, int], figure_px: tuple[int, int, int, int] | None, gray: int
) -> None:
    """Страница «как у FineReader»: образ во всю страницу и, если надо, образ-фигура; MediaBox по dpi."""
    scale = 72.0 / DPI
    page = doc.new_page(width=size_px[0] * scale, height=size_px[1] * scale)
    pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, size_px[0], size_px[1]), 0)
    pix.clear_with(gray)
    page.insert_image(page.rect, pixmap=pix)
    if figure_px:
        fig = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, figure_px[2] - figure_px[0], figure_px[3] - figure_px[1]), 0)
        fig.clear_with(0)
        page.insert_image(fitz.Rect(*[v * scale for v in figure_px]), pixmap=fig)
    # FineReader держит один поток на страницу — склеиваем то, что сделал insert_image.
    streams = page.get_contents()
    if len(streams) > 1:
        raw = b"\n".join(doc.xref_stream(x) for x in streams)
        doc.update_stream(streams[0], raw)
        page.set_contents(streams[0])


def build_pdfs(root: Path, plan) -> IssuePair:
    """Пара PDF выпуска: без коррекции — образ ровно «полоса + поля» и фигура под врезкой;
    с коррекцией — образ на 10 px уже (чтобы страницы различались)."""
    nogeo, geo = fitz.open(), fitz.open()
    for page_plan in plan.pages:
        width, height = page_plan.file_size
        size = (width + 2 * MARGINS.x_px, height + 2 * MARGINS.y_px)
        placed = page_plan.placed_pictures()
        figure = None
        if placed and not placed[0].covers(width, height):
            p = placed[0]
            figure = (
                p.x1 + MARGINS.x_px + 4,
                p.y1 + MARGINS.y_px + 4,
                p.x2 + MARGINS.x_px - 4,
                p.y2 + MARGINS.y_px - 4,
            )
        _fine_page(nogeo, size, figure, 230)
        _fine_page(geo, (size[0] - 10, size[1]), None, 250)
    geo_dir, nogeo_dir = root / "geo", root / "nogeo"
    geo_dir.mkdir(), nogeo_dir.mkdir()
    geo.save(str(geo_dir / plan.full_pdf_name))
    nogeo.save(str(nogeo_dir / plan.full_pdf_name))
    return IssuePair(geo_dir / plan.full_pdf_name, nogeo_dir / plan.full_pdf_name, len(plan.pages))


@pytest.fixture
def pack(tmp_path):
    """Пак из четырёх полос. Возвращает ``(db_path, blurred_dir, plan, pair, tmp_path)``."""
    blurred = tmp_path / "blurred"
    tree = [
        ScannedYear(
            name="1970",
            year=1970,
            rel_path="1970",
            issues=[
                ScannedIssue(
                    name="01",
                    number=1,
                    rel_path="1970/01",
                    pages=[
                        ScannedPage(
                            path=blurred / "1970/01" / name, file_name=name, rel_path=f"1970/01/{name}", order_index=i
                        )
                        for i, name in enumerate(PAGES)
                    ],
                )
            ],
        )
    ]
    db_path = tmp_path / "markup.sqlite"
    with open_db(db_path)() as session:
        pack_row = upsert_pack(session, PACK_NAME, blurred, tree)
        rows = {p.source_file_name: p for p in pack_row.year_packages[0].issues[0].pages}
        for name, page in rows.items():
            pictures, rotate = PAGES[name]
            page.width, page.height, page.dpi, page.divisor, page.rotate_cw = W, H, DPI, 8, rotate
            page.rect_regions = [
                RectRegion(x1=x1, y1=y1, x2=x2, y2=y2, kind=kind, full_page=False, source=SOURCE_CVAT)
                for x1, y1, x2, y2, kind in pictures
            ]
        session.commit()
    plan = load_plans(db_path, PACK_NAME)[0]
    for page_plan in plan.pages:
        _write_tiff(blurred / page_plan.original_rel_path, page_plan.file_size)
    pair = build_pdfs(tmp_path, plan)
    return db_path, blurred, plan, pair, tmp_path


@pytest.fixture
def layer_pdf(tmp_path):
    """Одностраничный PDF с потоком в манере FineReader (слова «Таблица», «и», «Резервы»)."""
    return finereader_like_pdf(tmp_path / "layer.pdf")
