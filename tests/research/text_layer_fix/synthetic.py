"""Синтетические PDF для тестов текстового слоя: поток «как у FineReader» и текст PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz

from research.text_layer_fix.rewrite import DEFAULT_FONT_PATH, InsertFont

FONT_RESOURCE = "F0"


def finereader_like_pdf(path: Path, with_image: bool = True) -> tuple[Path, InsertFont]:
    """PDF с одной страницей и потоком в манере FineReader: спаны трёх форм, ``cm`` внутри
    первого спана, по ``Tj`` на глиф, невидимый текст. Тексты: «Таблица» (обычный, растянутый),
    «и» (только Td), «Резервы» (повёрнутая матрица, читается снизу вверх).

    Args:
        path: Куда сохранить.
        with_image: Положить ли на страницу картинку (для сверки md5).

    Returns:
        Путь и шрифт, которым закодированы глифы.
    """
    font = InsertFont(DEFAULT_FONT_PATH)
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    if with_image:
        pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 40, 40), 0)
        pix.clear_with(200)
        page.insert_image(fitz.Rect(20, 20, 380, 580), pixmap=pix)
    page.insert_font(fontname=FONT_RESOURCE, fontfile=str(DEFAULT_FONT_PATH))

    def glyph_run(text: str, size: float, stretch: float, x: float, y: float) -> bytes:
        codes = font.encode(text)
        parts = [f"1 0 0 1 0 3 cm BT /{FONT_RESOURCE} {size} Tf 3 Tr {stretch} 0 0 1 {x} {y} Tm ".encode()]
        advance = 0.0
        for i, (char, code) in enumerate(zip(text, codes)):
            if i:
                parts.append(f"{advance:.3f} 0 Td ".encode())
            parts.append(b"<" + code.hex().encode() + b">Tj ")
            advance = font.length(char, size)
        parts.append(b"ET")
        return b"".join(parts)

    stream = b"q /Span <</MCID 0>> BDC " + glyph_run("Таблица", 10.0, 1.2, 100.0, 500.0) + b" EMC "
    code = font.encode("и")[0]
    stream += (
        f"/Span <</MCID 1>> BDC BT /{FONT_RESOURCE} 8 Tf 200 450 Td <".encode() + code.hex().encode() + b">Tj ET EMC "
    )
    codes = font.encode("Резервы")
    parts = [f"/Span <</MCID 2>> BDC BT /{FONT_RESOURCE} 9 Tf 0 1 -1 0 50 200 Tm ".encode()]
    for i, (char, c) in enumerate(zip("Резервы", codes)):
        if i:
            parts.append(f"{font.length('Резервы'[i - 1], 9.0):.3f} 0 Td ".encode())
        parts.append(b"<" + c.hex().encode() + b">Tj ")
    parts.append(b"ET EMC Q")
    stream += b"".join(parts)
    if page.get_contents():
        xref = page.get_contents()[0]
        doc.update_stream(xref, page.read_contents() + b"\n" + stream)
    else:
        # Страница без картинки не имеет потока: заводим новый объект-поток и вешаем на страницу.
        xref = doc.get_new_xref()
        doc.update_object(xref, "<<>>")
        doc.update_stream(xref, stream)
        page.set_contents(xref)
    doc.save(str(path))
    doc.close()
    return path, font


def pymupdf_text_pdf(path: Path) -> Path:
    """PDF, куда текст вписан самим PyMuPDF (многоглифные строки, массивы ``TJ``)."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text(
        fitz.Point(50, 100), "Снабжение 1966", fontsize=12, fontname="noto", fontfile=str(DEFAULT_FONT_PATH)
    )
    page.insert_text(fitz.Point(50, 130), "Hello world", fontsize=11, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path
