"""Пары PDF «с коррекцией / без» по выпуску и сверка страницы без коррекции с полосой из базы.

Промежуточная PDF собрана из заострённой полосы (при ``rotate_cw`` — повёрнутой) с полями
``margin_x`` слева и справа и ``margin_y`` сверху и снизу; FineReader без коррекции геометрии
сохраняет образ страницы пиксель в пиксель. Поэтому размер образа страницы без коррекции
обязан равняться повёрнутой полосе плюс поля — это и привязка «страница ↔ полоса», и
защита от подмены страниц.

ПОЛЯ МЕРЯЮТСЯ ПО КАЖДОЙ СТРАНИЦЕ. Поле задаётся в мм и в пиксели переводится по dpi полосы, а
потом округляется вверх до MCU её JPEG (8 px без субдискретизации цветности, 16 px при 4:2:0):
12 мм дают 288 px при 600 dpi, но 224 или 216 px при 450. В базе у выпуска записано наибольшее
из применённых полей (``Issue.full_intermediate_pdf_margin_*_mm``) — номинал ``N``. Поле
конкретной страницы обязано быть ``N`` или ``N − 8`` (та же цель, выровненная на MCU 8 вместо 16);
любое другое — чужая полоса на этом месте.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from ocr_utils.final_pdfs.plan import IssuePlan, PagePlan
from ocr_utils.pdf_utils.padding import JPEG_DCT_BLOCK_SIZE
from ocr_utils.units import MM_PER_INCH


@dataclass(frozen=True)
class Margins:
    """Поля промежуточной PDF в пикселях полосы (по X слева и справа, по Y сверху и снизу)."""

    x_px: int
    y_px: int


def margins_px(margin_x_mm: float, margin_y_mm: float, dpi: int) -> Margins:
    """Поля в миллиметрах → пиксели при dpi полосы.

    Args:
        margin_x_mm: Поле слева и справа, мм (у пака-1 — 12.192: 12 мм, округлённые вверх до MCU JPEG
            при 600 dpi, то есть 288 px).
        margin_y_mm: Поле сверху и снизу, мм (6.096 — 144 px при 600 dpi).
        dpi: Разрешение полосы.

    Returns:
        :class:`Margins`.
    """
    return Margins(int(round(margin_x_mm / MM_PER_INCH * dpi)), int(round(margin_y_mm / MM_PER_INCH * dpi)))


def nominal_margins(plan: IssuePlan, page_plan: PagePlan) -> Margins:
    """Номинальные поля выпуска (наибольшие из применённых) в пикселях этой полосы.

    Args:
        plan: План выпуска с полями в мм (из базы или из опций CLI).
        page_plan: Полоса — по её dpi мм переводятся в пиксели.

    Returns:
        :class:`Margins`.

    Raises:
        ValueError: У выпуска нет полей ни в базе, ни в опциях.
    """
    if plan.margin_x_mm is None or plan.margin_y_mm is None:
        raise ValueError(f"{plan.rel_path}: в базе нет полей промежуточной PDF — задать --margin-x-mm и --margin-y-mm")
    return margins_px(plan.margin_x_mm, plan.margin_y_mm, page_plan.dpi)


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


def page_margins(nogeo_page: fitz.Page, page_plan: PagePlan, nominal: Margins) -> Margins:
    """Поля страницы без коррекции по её образу, со сверкой с полосой из базы (повёрнутая полоса + поля).

    Args:
        nogeo_page: Страница PDF без коррекции геометрии.
        page_plan: Полоса из базы (размер, поворот).
        nominal: Номинальные поля выпуска в пикселях этой полосы (:func:`nominal_margins`).

    Returns:
        Фактические поля этой страницы в пикселях: ``N`` или ``N − 8`` по каждой оси (см. модуль).

    Raises:
        ValueError: Образ не равен полосе с допустимыми полями — страницы не соответствуют полосам.
    """
    if nogeo_page.rotation:
        raise ValueError(f"страница {nogeo_page.number + 1}: /Rotate {nogeo_page.rotation}, сборщик ждёт 0")
    actual = main_image_size(nogeo_page)
    width, height = page_plan.file_size
    if actual is not None:
        extra_x, extra_y = actual[0] - width, actual[1] - height
        # Поля симметричны: прибавка к стороне чётна; поле — номинал или номинал минус MCU 8.
        allowed_x = (nominal.x_px, nominal.x_px - JPEG_DCT_BLOCK_SIZE)
        allowed_y = (nominal.y_px, nominal.y_px - JPEG_DCT_BLOCK_SIZE)
        if extra_x % 2 == 0 and extra_y % 2 == 0 and extra_x // 2 in allowed_x and extra_y // 2 in allowed_y:
            return Margins(extra_x // 2, extra_y // 2)
    expected = (width + 2 * nominal.x_px, height + 2 * nominal.y_px)
    raise ValueError(
        f"страница {nogeo_page.number + 1}: образ {actual}, а полоса {page_plan.original_rel_path} с полями "
        f"даёт {expected} (или на 16 px меньше по оси) — порядок страниц не совпадает с базой или поля другие"
    )
