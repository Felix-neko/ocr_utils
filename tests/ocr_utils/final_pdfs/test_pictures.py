"""Иллюстрации: уменьшение, место на странице, JPEG на странице, снятие образов FineReader."""

import fitz
import numpy as np

from ocr_utils.final_pdfs.pictures import (
    figures_under,
    insert_picture,
    placement_rect,
    remove_images,
    render_pictures,
    resample,
)
from tests.ocr_utils.final_pdfs.conftest import DPI, INSET, MARGINS


def test_resample_halves_size_and_keeps_channels() -> None:
    gray = np.full((600, 400), 128, np.uint8)
    rgb = np.dstack([gray] * 3)
    assert resample(gray, 600, 300, 0.0).shape == (300, 200)
    assert resample(rgb, 600, 300, 0.05).shape == (300, 200, 3)
    # Целевое разрешение не ниже исходного — без изменений.
    assert resample(gray, 600, 600, 0.05) is gray


def test_render_pictures_gray_and_color(pack) -> None:
    _, blurred, plan, _, _ = pack
    inset = render_pictures(blurred / plan.pages[1].original_rel_path, plan.pages[1], 300, 75, 0.0)
    assert len(inset) == 1 and inset[0].gray and not inset[0].full_page
    assert (inset[0].width, inset[0].height) == (100, 100)
    assert inset[0].rect_px == INSET
    cover = render_pictures(blurred / plan.pages[2].original_rel_path, plan.pages[2], 300, 75, 0.0)
    assert cover[0].full_page and not cover[0].gray
    # Повёрнутая полоса: рамка врезки — в координатах повёрнутого файла.
    turned = render_pictures(blurred / plan.pages[3].original_rel_path, plan.pages[3], 300, 75, 0.0)
    assert turned[0].rect_px == (600 - INSET[3], INSET[0], 600 - INSET[1], INSET[2])


def test_placement_rect_adds_margins() -> None:
    rect = placement_rect(INSET, MARGINS, DPI)
    scale = 72.0 / DPI
    assert abs(rect.x0 - (INSET[0] + MARGINS.x_px) * scale) < 1e-6
    assert abs(rect.y1 - (INSET[3] + MARGINS.y_px) * scale) < 1e-6


def test_figures_under_and_removal(pack, tmp_path) -> None:
    _, blurred, plan, pair, _ = pack
    doc = fitz.open(str(pair.nogeo))
    page = doc[1]
    assert len(page.get_images()) == 2  # образ страницы + фигура под врезкой
    rect = placement_rect(INSET, MARGINS, DPI)
    xrefs = figures_under(page, [rect], full_page=False)
    assert len(xrefs) == 1  # только фигура: образ страницы лежит под врезкой меньше чем на 80 %
    assert remove_images(doc, page, xrefs) == 1
    picture = render_pictures(blurred / plan.pages[1].original_rel_path, plan.pages[1], 300, 75, 0.0)[0]
    insert_picture(doc, page, picture, rect)
    # Полностраничный растр снимает всё.
    cover_page = doc[2]
    all_xrefs = figures_under(cover_page, [cover_page.rect], full_page=True)
    assert remove_images(doc, cover_page, all_xrefs) == len(all_xrefs) == 1
    out = tmp_path / "out.pdf"
    doc.save(str(out), garbage=3)
    saved = fitz.open(str(out))
    infos = saved[1].get_image_info(xrefs=True)
    assert len(infos) == 2
    jpeg = [i for i in infos if i["width"] == 100][0]
    assert jpeg["cs-name"] == "DeviceGray"
    assert "DCTDecode" in saved.xref_get_key(jpeg["xref"], "Filter")[1]
    assert abs(fitz.Rect(jpeg["bbox"]).x0 - rect.x0) < 0.01 and abs(fitz.Rect(jpeg["bbox"]).y1 - rect.y1) < 0.01
    assert not saved[2].get_images()
