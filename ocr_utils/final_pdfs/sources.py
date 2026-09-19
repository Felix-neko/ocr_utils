"""Пары PDF «с коррекцией / без» по выпуску и сверка страницы без коррекции с полосой из базы.

Промежуточная PDF собрана из заострённой полосы (при ``rotate_cw`` — повёрнутой) с полями
``margin_x`` слева и справа и ``margin_y`` сверху и снизу; FineReader без коррекции геометрии
сохраняет образ страницы пиксель в пиксель. Поэтому размер образа страницы без коррекции
обязан равняться повёрнутой полосе плюс поля — это и привязка «страница ↔ полоса», и
защита от подмены страниц.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from ocr_utils.final_pdfs.plan import IssuePlan, PagePlan
from ocr_utils.pdf_utils.intermediate_pdfs import MM_PER_INCH


@dataclass(frozen=True)
class Margins:
    """Поля промежуточной PDF в пикселях полосы (по X слева и справа, по Y сверху и снизу)."""

    x_px: int
    y_px: int


def margins_px(margin_x_mm: float, margin_y_mm: float, dpi: int) -> Margins:
    """Поля в миллиметрах → пиксели при dpi полосы.

    Args:
        margin_x_mm: Поле слева и справа, мм (для пака-1 — 12.192: 12 мм, округлённые вверх до MCU JPEG).
        margin_y_mm: Поле сверху и снизу, мм (6.096).
        dpi: Разрешение полосы.

    Returns:
        :class:`Margins`.
    """
    return Margins(int(round(margin_x_mm / MM_PER_INCH * dpi)), int(round(margin_y_mm / MM_PER_INCH * dpi)))


@dataclass(frozen=True)
class IssuePair:
    """Пара PDF одного выпуска."""

    geo: Path
    nogeo: Path
    pages: int  # число страниц (одинаковое в обоих файлах и равное числу полос)


def pair_issue_pdfs(geo_dir: Path, nogeo_dir: Path, plan: IssuePlan) -> IssuePair:
    """Найти оба PDF выпуска и проверить, что числа страниц совпадают между собой и с базой.

    Args:
        geo_dir: Папка PDF с коррекцией геометрии.
        nogeo_dir: Папка PDF без коррекции.
        plan: План выпуска (имя файла — ``plan.full_pdf_name``).

    Returns:
        :class:`IssuePair`.

    Raises:
        FileNotFoundError: Нет одного из файлов.
        ValueError: Числа страниц расходятся.
    """
    geo, nogeo = geo_dir / plan.full_pdf_name, nogeo_dir / plan.full_pdf_name
    for path in (geo, nogeo):
        if not path.is_file():
            raise FileNotFoundError(f"нет PDF выпуска: {path}")
    with fitz.open(str(geo)) as a, fitz.open(str(nogeo)) as b:
        if len(a) != len(b):
            raise ValueError(f"{plan.full_pdf_name}: страниц с коррекцией {len(a)}, без коррекции {len(b)}")
        if len(a) != len(plan.pages):
            raise ValueError(f"{plan.full_pdf_name}: в PDF {len(a)} страниц, в базе {len(plan.pages)} полос")
        return IssuePair(geo, nogeo, len(a))


def main_image_size(page: fitz.Page) -> tuple[int, int] | None:
    """Размер (ширина, высота) самого большого образа страницы в пикселях; ``None`` без образов."""
    infos = page.get_image_info()
    if not infos:
        return None
    main = max(infos, key=lambda info: info["width"] * info["height"])
    return int(main["width"]), int(main["height"])


def verify_page_geometry(nogeo_page: fitz.Page, page_plan: PagePlan, margins: Margins) -> None:
    """Сверить образ страницы без коррекции с полосой из базы: повёрнутая полоса + поля.

    Args:
        nogeo_page: Страница PDF без коррекции геометрии.
        page_plan: Полоса из базы (размер, поворот).
        margins: Поля промежуточной PDF в пикселях.

    Raises:
        ValueError: Размер образа не равен ожидаемому — страницы не соответствуют полосам.
    """
    if nogeo_page.rotation:
        raise ValueError(f"страница {nogeo_page.number + 1}: /Rotate {nogeo_page.rotation}, сборщик ждёт 0")
    actual = main_image_size(nogeo_page)
    width, height = page_plan.file_size
    expected = (width + 2 * margins.x_px, height + 2 * margins.y_px)
    if actual != expected:
        raise ValueError(
            f"страница {nogeo_page.number + 1}: образ {actual}, а полоса {page_plan.original_rel_path} с полями "
            f"даёт {expected} — порядок страниц не совпадает с базой или поля другие"
        )
