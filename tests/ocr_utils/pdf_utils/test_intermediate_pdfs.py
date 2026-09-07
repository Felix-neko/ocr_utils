"""Сборка промежуточных PDF: что попадает на страницу и что записывается в базу."""

import io

import fitz
import pytest

from ocr_utils.pdf_utils.intermediate_pdfs import BuildParams, load_plans, run_build
from ocr_utils.scan_markup.db.repo import require_pack
from ocr_utils.scan_markup.db.session import open_db

from .conftest import DPI, H, ORIGINAL_COLOR, PACK_NAME, SHARPENED_COLOR, W


def _params(tmp_path, originals, sharpened, **kwargs):
    """Параметры сборки. Поля по умолчанию НУЛЕВЫЕ — иначе поехали бы все координаты.

    Проверки ниже сверяют пиксели по номерам, а поля сдвигают содержимое страницы. Ставить
    их тут значило бы к каждому номеру прибавлять ширину поля и проверять заодно арифметику
    самого теста. Поля проверяются отдельно, в ``test_margins_*``.
    """
    kwargs.setdefault("margin_x_mm", 0.0)
    kwargs.setdefault("margin_y_mm", 0.0)
    return BuildParams(
        originals_dir=originals,
        sharpened_dir=sharpened,
        full_pdf_dir=tmp_path / "full",
        pics_only_pdf_dir=tmp_path / "pics",
        **kwargs,
    )


# Поля для проверок ниже: заметно больше пикселя и меньше самой полосы (400x600 при 600 dpi
# — это 17x25 мм). Точная ширина в пикселях округляется вверх до размера MCU, поэтому нигде
# не ожидается конкретное число: оно вычисляется из готового файла.
MARGIN_X_MM, MARGIN_Y_MM = 1.0, 0.5


def _used_margins(document):
    """Фактические поля страницы в пикселях — по тому, насколько она шире полосы."""
    rect = document[0].rect
    return round((rect.width * DPI / 72 - W) / 2), round((rect.height * DPI / 72 - H) / 2)


def test_page_indices_follow_the_order_and_skip_pages_without_pictures(pack):
    """Полоса без иллюстраций в PDF типа PAGES_WITH_PICS_ONLY не попадает вовсе."""
    db_path, originals, sharpened, name = pack
    (plan,) = load_plans(db_path, name)

    assert [p.full_pdf_page_idx for p in plan.pages] == [0, 1, 2]
    assert [p.pages_with_pics_only_pdf_page_idx for p in plan.pages] == [None, 0, 1]
    assert plan.full_pdf_name == "full_1970_01.pdf"
    assert plan.pics_pdf_name == "pages_with_pics_only_1970_01.pdf"


def test_issue_without_pictures_gets_no_pics_only_pdf(pack, tmp_path):
    """У полностью текстового выпуска второй PDF не из чего собрать — и его имя пусто."""
    db_path, originals, sharpened, name = pack
    with open_db(db_path)() as session:
        for page in require_pack(session, name).year_packages[0].issues[0].pages:
            page.raster_regions = []
        session.commit()

    (plan,) = load_plans(db_path, name)
    assert plan.pics_pdf_name is None

    stats = run_build(db_path, name, _params(tmp_path, originals, sharpened), jobs=1, progress=False)
    assert stats.failed == 0
    assert not list((tmp_path / "pics").glob("*.pdf"))

    with open_db(db_path)() as session:
        issue = require_pack(session, name).year_packages[0].issues[0]
        assert issue.full_intermediate_pdf_name == "full_1970_01.pdf"
        assert issue.pages_with_pics_only_intermediate_pdf_name is None


def test_build_writes_both_pdfs_and_fills_the_database(pack, tmp_path):
    db_path, originals, sharpened, name = pack
    params = _params(tmp_path, originals, sharpened)
    stats = run_build(db_path, name, params, jobs=1, progress=False)

    assert (stats.issues, stats.failed, stats.pages, stats.pages_with_pics) == (1, 0, 3, 2)
    full = tmp_path / "full" / "full_1970_01.pdf"
    pics = tmp_path / "pics" / "pages_with_pics_only_1970_01.pdf"
    assert fitz.open(full).page_count == 3
    assert fitz.open(pics).page_count == 2

    with open_db(db_path)() as session:
        pack_row = require_pack(session, name)
        assert pack_row.cleaned_pics_root == str(originals)
        assert pack_row.sharpened_text_pics_root == str(sharpened)
        assert pack_row.full_intermediate_pdf_root == str(tmp_path / "full")
        issue = pack_row.year_packages[0].issues[0]
        assert issue.full_intermediate_pdf_name == "full_1970_01.pdf"
        assert [p.full_pdf_page_idx for p in issue.pages] == [0, 1, 2]
        assert [p.pages_with_pics_only_pdf_page_idx for p in issue.pages] == [None, 0, 1]


def test_pages_are_made_of_the_right_pictures(pack, tmp_path):
    """Основа страницы — заострённая копия, врезка — оригинал, и ровно на своём месте."""
    db_path, originals, sharpened, name = pack
    run_build(db_path, name, _params(tmp_path, originals, sharpened), jobs=1, progress=False)

    document = fitz.open(tmp_path / "full" / "full_1970_01.pdf")
    assert document[0].rect.width == pytest.approx(W * 72 / DPI, abs=0.01)

    plain = document[0].get_pixmap(dpi=DPI)
    assert plain.pixel(200, 300) == pytest.approx(SHARPENED_COLOR, abs=8)

    # Врезка размечена как ``grayscale``, поэтому в PDF она обесцвечена — по этому и видно,
    # что цветность берётся из вида области, а не из содержимого пикселей.
    gray = round(0.299 * ORIGINAL_COLOR[0] + 0.587 * ORIGINAL_COLOR[1] + 0.114 * ORIGINAL_COLOR[2])
    inset = document[1].get_pixmap(dpi=DPI)
    assert inset.pixel(200, 300) == pytest.approx((gray, gray, gray), abs=8)  # внутри врезки
    assert inset.pixel(50, 50) == pytest.approx(SHARPENED_COLOR, abs=8)  # текстовая часть

    # Полосная иллюстрация: заострённой копии под ней не видно нигде.
    cover = document[2].get_pixmap(dpi=DPI)
    assert cover.pixel(50, 50) == pytest.approx(ORIGINAL_COLOR, abs=8)
    assert cover.pixel(W - 5, H - 5) == pytest.approx(ORIGINAL_COLOR, abs=8)


def test_full_page_illustration_does_not_carry_the_sharpened_copy(pack, tmp_path):
    """Заострённая копия под полосной иллюстрацией в файл не кладётся: она невидима."""
    db_path, originals, sharpened, name = pack
    run_build(db_path, name, _params(tmp_path, originals, sharpened), jobs=1, progress=False)

    document = fitz.open(tmp_path / "full" / "full_1970_01.pdf")
    assert len(document[2].get_images()) == 1
    assert len(document[1].get_images()) == 2  # основа плюс врезка


def test_sharpened_copy_of_another_size_is_an_error(pack, tmp_path):
    """Молча растянуть заострённую копию нельзя: разметка привязана к пикселям оригинала."""
    import numpy as np
    from PIL import Image

    db_path, originals, sharpened, name = pack
    Image.fromarray(np.zeros((H // 2, W // 2, 3), np.uint8)).save(sharpened / "1970/01/0010.jpg")

    stats = run_build(db_path, name, _params(tmp_path, originals, sharpened), jobs=1, progress=False)
    assert stats.failed == 1
    assert "не совпадает" in stats.reports[0].reason


def test_skip_if_exists_keeps_the_database_up_to_date(pack, tmp_path):
    """Пропуск готового выпуска не должен стирать номера страниц, записанные прошлым разом."""
    db_path, originals, sharpened, name = pack
    params = _params(tmp_path, originals, sharpened)
    run_build(db_path, name, params, jobs=1, progress=False)

    stats = run_build(db_path, name, params, jobs=1, progress=False)
    assert (stats.skipped, stats.issues, stats.failed) == (1, 0, 0)

    with open_db(db_path)() as session:
        issue = require_pack(session, name).year_packages[0].issues[0]
        assert issue.full_intermediate_pdf_name == "full_1970_01.pdf"
        assert [p.pages_with_pics_only_pdf_page_idx for p in issue.pages] == [None, 0, 1]


def test_grayscale_original_is_cut_and_embedded(pack, tmp_path):
    """Очищенная полоса может лежать одним серым каналом — врезка из неё обязана собраться.

    После того как ``scan_cleanup`` перестал хранить цвет там, где он не размечен, это
    основной случай: цветных полос в паке-1 167 из 12 135.
    """
    import numpy as np
    from PIL import Image

    from ocr_utils.pdf_utils.jpeg_pdf import read_jpeg_info

    db_path, originals, sharpened, name = pack
    # Полоса 0020 — с серой врезкой; кладём её оригинал одним каналом.
    grey = np.full((H, W), 90, np.uint8)
    Image.fromarray(grey, "L").save(originals / "1970" / "01" / "0020.tif")

    stats = run_build(db_path, name, _params(tmp_path, originals, sharpened), jobs=1, progress=False)
    assert stats.failed == 0

    document = fitz.open(tmp_path / "full" / "full_1970_01.pdf")
    page = document[1]
    assert len(page.get_images()) == 2  # основа плюс врезка
    assert page.get_pixmap(dpi=DPI).pixel(200, 300) == pytest.approx((90, 90, 90), abs=8)

    # Врезка встроена одноканальной: лишний цвет тут не из чего взять.
    import pikepdf

    with pikepdf.Pdf.open(tmp_path / "full" / "full_1970_01.pdf") as pdf:
        names = sorted(pdf.pages[1].Resources.XObject.keys())
        assert read_jpeg_info(pdf.pages[1].Resources.XObject[names[1]].read_raw_bytes()).components == 1


def test_margins_are_added_to_the_full_pdf_only(pack, tmp_path):
    """Поля есть в полной PDF и отсутствуют в PAGES_WITH_PICS_ONLY.

    Вторую распознают БЕЗ распрямления строк, кадр там не растёт, и поля только увели бы
    иллюстрации от размеченных для них мест.
    """
    db_path, originals, sharpened, name = pack
    params = _params(tmp_path, originals, sharpened, margin_x_mm=MARGIN_X_MM, margin_y_mm=MARGIN_Y_MM)
    assert run_build(db_path, name, params, jobs=1, progress=False).failed == 0

    full = fitz.open(tmp_path / "full" / "full_1970_01.pdf")
    used_x, used_y = _used_margins(full)
    assert used_x >= round(MARGIN_X_MM / 25.4 * DPI)  # округление только вверх
    assert used_y >= round(MARGIN_Y_MM / 25.4 * DPI)
    for page in full:
        assert page.rect.width == pytest.approx((W + 2 * used_x) * 72 / DPI, abs=0.01)
        assert page.rect.height == pytest.approx((H + 2 * used_y) * 72 / DPI, abs=0.01)

    pics = fitz.open(tmp_path / "pics" / "pages_with_pics_only_1970_01.pdf")
    for page in pics:
        assert page.rect.width == pytest.approx(W * 72 / DPI, abs=0.01)
        assert page.rect.height == pytest.approx(H * 72 / DPI, abs=0.01)


def test_margins_move_the_insets_with_the_page(pack, tmp_path):
    """Врезка размечена в пикселях полосы БЕЗ полей — на странице она обязана сдвинуться."""
    db_path, originals, sharpened, name = pack
    params = _params(tmp_path, originals, sharpened, margin_x_mm=MARGIN_X_MM, margin_y_mm=MARGIN_Y_MM)
    run_build(db_path, name, params, jobs=1, progress=False)

    document = fitz.open(tmp_path / "full" / "full_1970_01.pdf")
    used_x, used_y = _used_margins(document)
    inset = document[1].get_pixmap(dpi=DPI)

    gray = round(0.299 * ORIGINAL_COLOR[0] + 0.587 * ORIGINAL_COLOR[1] + 0.114 * ORIGINAL_COLOR[2])
    assert inset.pixel(200 + used_x, 300 + used_y) == pytest.approx((gray, gray, gray), abs=8)
    assert inset.pixel(50 + used_x, 50 + used_y) == pytest.approx(SHARPENED_COLOR, abs=8)
    # Само поле — заливка цветом бумаги, а не продолжение полосы.
    assert inset.pixel(1, 1) != pytest.approx(SHARPENED_COLOR, abs=8)


def test_margins_do_not_recode_the_page(pack, tmp_path):
    """Полоса под полями остаётся той же самой: DCT-блоки переносятся, а не пережимаются."""
    import numpy as np
    import pikepdf
    from PIL import Image

    db_path, originals, sharpened, name = pack
    run_build(db_path, name, _params(tmp_path, originals, sharpened), jobs=1, progress=False)
    with pikepdf.Pdf.open(tmp_path / "full" / "full_1970_01.pdf") as pdf:
        plain = bytes(pdf.pages[0].Resources.XObject["/Im0"].read_raw_bytes())

    for path in (tmp_path / "full", tmp_path / "pics"):
        for pdf_path in path.glob("*.pdf"):
            pdf_path.unlink()
    params = _params(tmp_path, originals, sharpened, margin_x_mm=MARGIN_X_MM, margin_y_mm=MARGIN_Y_MM)
    run_build(db_path, name, params, jobs=1, progress=False)
    with pikepdf.Pdf.open(tmp_path / "full" / "full_1970_01.pdf") as pdf:
        padded = bytes(pdf.pages[0].Resources.XObject["/Im0"].read_raw_bytes())

    document = fitz.open(tmp_path / "full" / "full_1970_01.pdf")
    used_x, used_y = _used_margins(document)
    before = np.asarray(Image.open(io.BytesIO(plain)).convert("RGB"))
    after = np.asarray(Image.open(io.BytesIO(padded)).convert("RGB"))
    assert after.shape == (H + 2 * used_y, W + 2 * used_x, 3)
    assert np.array_equal(after[used_y : used_y + H, used_x : used_x + W], before)


def test_margins_are_written_to_the_database(pack, tmp_path):
    """В базе по выпуску лежат фактические поля полной PDF, все четыре стороны."""
    db_path, originals, sharpened, name = pack
    params = _params(tmp_path, originals, sharpened, margin_x_mm=MARGIN_X_MM, margin_y_mm=MARGIN_Y_MM)
    run_build(db_path, name, params, jobs=1, progress=False)

    document = fitz.open(tmp_path / "full" / "full_1970_01.pdf")
    used_x, used_y = _used_margins(document)
    with open_db(db_path)() as session:
        issue = require_pack(session, name).year_packages[0].issues[0]
        assert issue.full_intermediate_pdf_margin_left_mm == pytest.approx(used_x / DPI * 25.4, abs=1e-6)
        assert issue.full_intermediate_pdf_margin_right_mm == pytest.approx(used_x / DPI * 25.4, abs=1e-6)
        assert issue.full_intermediate_pdf_margin_top_mm == pytest.approx(used_y / DPI * 25.4, abs=1e-6)
        assert issue.full_intermediate_pdf_margin_bottom_mm == pytest.approx(used_y / DPI * 25.4, abs=1e-6)


def test_skipped_issue_keeps_the_margins_of_the_run_that_built_it(pack, tmp_path):
    """Пропуск готового выпуска не должен переписывать поля сегодняшними ключами."""
    db_path, originals, sharpened, name = pack
    built = _params(tmp_path, originals, sharpened, margin_x_mm=MARGIN_X_MM, margin_y_mm=MARGIN_Y_MM)
    run_build(db_path, name, built, jobs=1, progress=False)
    with open_db(db_path)() as session:
        was = require_pack(session, name).year_packages[0].issues[0].full_intermediate_pdf_margin_left_mm
    assert was > 0

    stats = run_build(db_path, name, _params(tmp_path, originals, sharpened), jobs=1, progress=False)
    assert stats.skipped == 1
    with open_db(db_path)() as session:
        issue = require_pack(session, name).year_packages[0].issues[0]
        assert issue.full_intermediate_pdf_margin_left_mm == was
