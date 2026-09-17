"""Сборка финальной PDF из распознанных промежуточных."""

import fitz
import pytest

from ocr_utils.pdf_utils.final_pdfs import AssembleParams, run_assemble
from ocr_utils.pdf_utils.intermediate_pdfs import BuildParams, run_build
from ocr_utils.db.repo import require_pack
from ocr_utils.db.session import open_db

from .conftest import DPI, H, ORIGINAL_COLOR, SHARPENED_COLOR, W

# Латиница и покороче: у встроенной Helvetica нет кириллицы, а страница синтетического
# пака шириной 48 pt — длинная строка ушла бы за край и обрезалась при извлечении.
# Тест проверял бы тогда шрифт и поля, а не сохранность текстового слоя.
TEXT = "OCR"


def _recognize(source_dir, dest_dir):
    """Изображает FineReader: та же геометрия страниц плюс текстовый слой."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.pdf")):
        document = fitz.open(path)
        for page in document:
            page.insert_text((10, 20), TEXT, fontsize=8, fontname="helv")
        document.save(dest_dir / path.name)
        document.close()


@pytest.fixture
def recognized(pack, tmp_path):
    """Пак, собранный в промежуточные PDF и «распознанный»."""
    db_path, originals, sharpened, name = pack
    run_build(
        db_path,
        name,
        BuildParams(
            originals_dir=originals,
            sharpened_dir=sharpened,
            full_pdf_dir=tmp_path / "full",
            pics_only_pdf_dir=tmp_path / "pics",
        ),
        jobs=1,
        progress=False,
    )
    _recognize(tmp_path / "full", tmp_path / "full_ocr")
    _recognize(tmp_path / "pics", tmp_path / "pics_ocr")
    return db_path, originals, name


def _params(tmp_path, originals, **kwargs):
    return AssembleParams(
        originals_dir=originals,
        full_pdf_dir=tmp_path / "full_ocr",
        pics_only_pdf_dir=tmp_path / "pics_ocr",
        final_dir=tmp_path / "final",
        **kwargs,
    )


def test_final_pdf_keeps_the_text_layer_and_restores_the_pictures(recognized, tmp_path):
    db_path, originals, name = recognized
    stats = run_assemble(db_path, name, _params(tmp_path, originals), jobs=1, progress=False)

    assert (stats.issues, stats.failed, stats.pages, stats.pictures) == (1, 0, 3, 2)
    document = fitz.open(tmp_path / "final" / "1970_01.pdf")
    assert document.page_count == 3

    # Текстовый слой на месте на всех страницах, включая те, куда врезаны иллюстрации.
    assert all(TEXT in page.get_text() for page in document)

    gray = round(0.299 * ORIGINAL_COLOR[0] + 0.587 * ORIGINAL_COLOR[1] + 0.114 * ORIGINAL_COLOR[2])
    inset = document[1].get_pixmap(dpi=DPI)
    assert inset.pixel(200, 300) == pytest.approx((gray, gray, gray), abs=10)
    assert inset.pixel(50, 50) == pytest.approx(SHARPENED_COLOR, abs=10)

    cover = document[2].get_pixmap(dpi=DPI)
    assert cover.pixel(W // 2, H // 2) == pytest.approx(ORIGINAL_COLOR, abs=10)


def test_pages_without_pictures_come_from_the_straightened_pdf(recognized, tmp_path):
    """Полосу без иллюстраций берём из полной PDF — той, где строки распрямлены."""
    db_path, originals, name = recognized
    # Метим первую страницу полной распознанной PDF, чтобы отличить её источник.
    path = tmp_path / "full_ocr" / "full_1970_01.pdf"
    document = fitz.open(path)
    document[0].insert_text((10, 40), "FULL", fontsize=8, fontname="helv")
    document.save(path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    document.close()

    run_assemble(db_path, name, _params(tmp_path, originals), jobs=1, progress=False)
    final = fitz.open(tmp_path / "final" / "1970_01.pdf")
    assert "FULL" in final[0].get_text()
    assert "FULL" not in final[1].get_text()


def test_page_count_mismatch_is_refused(recognized, tmp_path):
    """Если распознаватель выбросил страницу, номера в базе врут — собирать нельзя."""
    db_path, originals, name = recognized
    path = tmp_path / "full_ocr" / "full_1970_01.pdf"
    document = fitz.open(path)
    document.delete_page(0)
    document.save(tmp_path / "full_ocr" / "tmp.pdf")
    document.close()
    (tmp_path / "full_ocr" / "tmp.pdf").replace(path)

    stats = run_assemble(db_path, name, _params(tmp_path, originals), jobs=1, progress=False)
    assert stats.failed == 1
    assert "номера страниц" in stats.reports[0].reason


def test_by_year_lays_the_result_out_in_year_folders(recognized, tmp_path):
    db_path, originals, name = recognized
    run_assemble(db_path, name, _params(tmp_path, originals, by_year=True), jobs=1, progress=False)
    assert (tmp_path / "final" / "1970" / "1970_01.pdf").exists()

    with open_db(db_path)() as session:
        pack_row = require_pack(session, name)
        assert pack_row.final_pdfs_root == str(tmp_path / "final")
        assert pack_row.year_packages[0].issues[0].final_pdf_name == "1970_01.pdf"
