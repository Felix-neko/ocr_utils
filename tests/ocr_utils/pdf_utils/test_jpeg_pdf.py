"""Встраивание JPEG в PDF без перекодирования и геометрия врезок."""

import io

import fitz
import numpy as np
import pikepdf
import pytest
from PIL import Image

from ocr_utils.pdf_utils.jpeg_pdf import (
    JpegError,
    _draw,
    Overlay,
    encode_jpeg,
    embed_jpeg,
    make_page,
    overlay_on_page,
    placement_matrix,
    read_jpeg_info,
)

DPI = 600
W, H = 400, 600


def jpeg_bytes(size=(W, H), color=(200, 30, 30), mode="RGB", progressive=False) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, "JPEG", quality=90, progressive=progressive)
    return buffer.getvalue()


def test_read_jpeg_info_reads_size_and_components():
    info = read_jpeg_info(jpeg_bytes())
    assert (info.width, info.height, info.components, info.progressive) == (W, H, 3, False)

    gray = read_jpeg_info(jpeg_bytes(mode="L", color=128))
    assert (gray.components, gray.color_space) == (1, pikepdf.Name.DeviceGray)


def test_progressive_jpeg_is_recognised_and_refused():
    """Прогрессивный JPEG PDF не декодирует — страница вышла бы пустой, а не кривой."""
    data = jpeg_bytes(progressive=True)
    assert read_jpeg_info(data).progressive
    with pytest.raises(JpegError):
        embed_jpeg(pikepdf.Pdf.new(), data)


def test_not_a_jpeg_is_refused():
    with pytest.raises(JpegError):
        read_jpeg_info(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)


def test_page_keeps_the_jpeg_byte_for_byte():
    """Главное свойство модуля: байты картинки в PDF те же, что были в файле."""
    data = jpeg_bytes()
    pdf = pikepdf.Pdf.new()
    pdf.pages.append(make_page(pdf, data, DPI))
    buffer = io.BytesIO()
    pdf.save(buffer)

    reopened = pikepdf.Pdf.open(io.BytesIO(buffer.getvalue()))
    stored = reopened.pages[0].Resources.XObject["/Im0"].read_raw_bytes()
    assert stored == data


def test_page_size_comes_from_dpi():
    """Полоса 600 dpi должна стать листом в дюймах, а не в пикселях."""
    pdf = pikepdf.Pdf.new()
    pdf.pages.append(make_page(pdf, jpeg_bytes(), DPI))
    box = [float(v) for v in pdf.pages[0].mediabox]
    assert box == pytest.approx([0, 0, W * 72 / DPI, H * 72 / DPI])


def test_page_refuses_a_picture_of_another_size():
    pdf = pikepdf.Pdf.new()
    with pytest.raises(JpegError):
        make_page(pdf, jpeg_bytes(), DPI, page_px=(W + 1, H))


def _render(pdf: pikepdf.Pdf) -> "fitz.Pixmap":
    buffer = io.BytesIO()
    pdf.save(buffer)
    document = fitz.open("pdf", buffer.getvalue())
    return document[0].get_pixmap(dpi=DPI)


def test_overlay_lands_exactly_on_its_rectangle():
    rect = (100, 200, 300, 400)
    pdf = pikepdf.Pdf.new()
    overlay = Overlay(jpeg_bytes((rect[2] - rect[0], rect[3] - rect[1]), (20, 220, 20)), rect)
    pdf.pages.append(make_page(pdf, jpeg_bytes(), DPI, (overlay,)))

    pixmap = _render(pdf)
    assert pixmap.pixel(200, 300) == pytest.approx((20, 220, 20), abs=6)  # внутри врезки
    assert pixmap.pixel(101, 201) == pytest.approx((20, 220, 20), abs=6)  # её верхний угол
    assert pixmap.pixel(98, 198) == pytest.approx((200, 30, 30), abs=6)  # пиксель за краем
    assert pixmap.pixel(50, 50) == pytest.approx((200, 30, 30), abs=6)  # далеко от врезки


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_overlay_on_an_existing_rotated_page(rotation):
    """Врезка на готовую страницу с ``/Rotate`` и сдвинутым mediabox.

    Содержимое рисуется в НЕПОВЁРНУТОМ пространстве mediabox, а ``/Rotate`` разворачивает
    страницу уже при показе, поэтому при повороте на четверть mediabox лежит «поперёк»
    видимого листа. Собираем именно такую страницу — как её отдал бы распознаватель — и
    проверяем по отрисовке: читатель должен увидеть врезку там же, где она была бы на
    неповёрнутой странице.
    """
    width_pt, height_pt = W * 72 / DPI, H * 72 / DPI
    box_w, box_h = (height_pt, width_pt) if rotation in (90, 270) else (width_pt, height_pt)
    shift_x, shift_y = 20.0, 30.0
    mediabox = (shift_x, shift_y, shift_x + box_w, shift_y + box_h)

    pdf = pikepdf.Pdf.new()
    image = embed_jpeg(pdf, jpeg_bytes())
    matrix = placement_matrix((0, 0, W, H), (W, H), mediabox, rotation)
    page = pikepdf.Page(
        pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=pikepdf.Array([*mediabox]),
                Rotate=rotation,
                Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=image)),
                Contents=pikepdf.Stream(pdf, _draw(pikepdf.Name("/Im0"), matrix)),
            )
        )
    )
    pdf.pages.append(page)

    rect = (100, 200, 300, 400)
    fragment = jpeg_bytes((rect[2] - rect[0], rect[3] - rect[1]), (20, 220, 20))
    overlay_on_page(pdf, pdf.pages[0], fragment, rect, (W, H))

    pixmap = _render(pdf)
    # Растеризация отдаёт страницу уже развёрнутой, поэтому координаты пикселей те же, что
    # и на неповёрнутой: в этом и смысл проверки.
    assert (pixmap.width, pixmap.height) == (W, H)
    assert pixmap.pixel(200, 300) == pytest.approx((20, 220, 20), abs=6)
    assert pixmap.pixel(101, 201) == pytest.approx((20, 220, 20), abs=6)
    assert pixmap.pixel(50, 50) == pytest.approx((200, 30, 30), abs=6)


def test_placement_matrix_refuses_a_page_of_another_shape():
    """Страница, переставшая быть подобной кадру, — это не повод растянуть врезку."""
    with pytest.raises(ValueError, match="не подобна"):
        placement_matrix((0, 0, 10, 10), (W, H), (0.0, 0.0, 100.0, 100.0))


def test_encode_jpeg_respects_the_gray_flag():
    """``gray`` приходит из вида области, а не угадывается по пикселям."""
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[..., 0] = 255
    assert read_jpeg_info(encode_jpeg(array, gray=False, quality=90)).components == 3
    assert read_jpeg_info(encode_jpeg(array, gray=True, quality=90)).components == 1
