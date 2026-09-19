"""Выбор источника страницы, пары PDF и сверка геометрии страницы с полосой."""

import fitz
import pytest

from pathlib import Path

from ocr_utils.final_pdfs.plan import PageSource, SourceReason, decide_source, final_pdf_name, final_pdf_path
from ocr_utils.final_pdfs.sources import Margins, margins_px, pair_issue_pdfs, verify_page_geometry
from tests.ocr_utils.final_pdfs.conftest import MARGINS


def test_decide_source_rules() -> None:
    assert decide_source(True, "ok") == (PageSource.NOGEO, SourceReason.PICTURES)
    assert decide_source(False, "bad") == (PageSource.NOGEO, SourceReason.GEOMETRY_BAD)
    assert decide_source(False, "mixed") == (PageSource.GEO, SourceReason.GEOMETRY_OK)
    assert decide_source(False, "ok") == (PageSource.GEO, SourceReason.GEOMETRY_OK)


def test_margins_px_rounds_to_pixels() -> None:
    # Поля пака-1: 12.192 / 6.096 мм при 600 dpi — ровно 288 / 144 px (MCU JPEG).
    assert margins_px(12.192, 6.096, 600) == Margins(288, 144)


def test_plan_and_pair(pack) -> None:
    _, _, plan, pair, _ = pack
    assert final_pdf_name(plan) == "1970_01.pdf"
    assert final_pdf_path(Path("/out"), plan) == Path("/out/1970/1970_01.pdf")
    assert [p.full_pdf_page_idx for p in plan.pages] == [0, 1, 2, 3]
    assert plan.pages[3].rotate_cw == 90 and plan.pages[3].file_size == (600, 400)
    found = pair_issue_pdfs(pair.geo.parent, pair.nogeo.parent, plan)
    assert found.pages == 4


def test_pair_refuses_page_count_mismatch(pack) -> None:
    _, _, plan, pair, tmp_path = pack
    with fitz.open(str(pair.geo)) as doc:
        doc.delete_page(0)
        doc.save(str(tmp_path / "short.pdf"))
    (tmp_path / "geo2").mkdir()
    (tmp_path / "short.pdf").rename(tmp_path / "geo2" / plan.full_pdf_name)
    with pytest.raises(ValueError, match="страниц"):
        pair_issue_pdfs(tmp_path / "geo2", pair.nogeo.parent, plan)
    with pytest.raises(FileNotFoundError):
        pair_issue_pdfs(tmp_path / "nowhere", pair.nogeo.parent, plan)


def test_verify_page_geometry(pack) -> None:
    _, _, plan, pair, _ = pack
    with fitz.open(str(pair.nogeo)) as nogeo, fitz.open(str(pair.geo)) as geo:
        for index in range(4):
            verify_page_geometry(nogeo[index], plan.pages[index], MARGINS)
        # Страница с коррекцией на 10 px уже — под полосу не подходит.
        with pytest.raises(ValueError, match="порядок страниц"):
            verify_page_geometry(geo[1], plan.pages[1], MARGINS)
        # Чужая полоса (повёрнутая) на месте прямой — тоже.
        with pytest.raises(ValueError):
            verify_page_geometry(nogeo[1], plan.pages[3], MARGINS)
