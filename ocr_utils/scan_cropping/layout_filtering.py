"""Отсев паразитных (артефактных) блоков Surya layout из расчёта crop-зоны."""

import cv2
import numpy as np

# --- Паразитные (артефактные) блоки layout ---------------------------------
# На пустых/почти пустых страницах Surya иногда рисует один блок почти на всю
# страницу. Такой блок раздувает crop-зону (``crop_ext_with_layout`` тянется до
# него) и мешает закраске пальцев. Когда область книги уже известна (page_mask,
# зелёный контур на оверлее), отсеиваем такие блоки из расчёта crop-зоны по двум
# эвристикам (см. ``classify_parasitic_layouts``):
#   H1: площадь блока > LAYOUT_PARASITE_MAX_BOOK_AREA_FRAC площади книги;
#   H2: блок раздувает грубый axis-aligned bbox книги более чем на
#       LAYOUT_PARASITE_MAX_BBOX_GROWTH_FRAC (по площади).
# Паразитные блоки на debug-оверлее рисуются ПУНКТИРОМ (нормальные — сплошным).
#
# Порог H1 = 0.50 выбран консервативно: лучше пропустить часть гигантских
# паразитных блоков, чем ложно пометить настоящий крупный блок. Для справки по
# данным: паразитный блок = целая пустая страница ≈ 45%+ площади книги, а самый
# крупный НОРМАЛЬНЫЙ одиночный блок (плотное оглавление) ≈ 33%. При 0.50 почти
# пустая страница IMG_0104 (44.7%) под H1 не подпадёт — это осознанный компромисс.
LAYOUT_PARASITE_MAX_BOOK_AREA_FRAC = 0.50  # H1
LAYOUT_PARASITE_MAX_BBOX_GROWTH_FRAC = 0.30  # H2 (N)


def classify_parasitic_layouts(
    layout_polys: "list[np.ndarray]",
    book_mask: np.ndarray,
    layout_pad_px: "int | tuple[int, int]",
    max_book_area_frac: float = LAYOUT_PARASITE_MAX_BOOK_AREA_FRAC,
    max_bbox_growth_frac: float = LAYOUT_PARASITE_MAX_BBOX_GROWTH_FRAC,
) -> "list[bool]":
    """Для каждого полигона layout — флаг «паразитный» (артефакт на пустой странице).

    Требует уже известную область книги ``book_mask`` (силуэт разворота E1, тот же,
    что рисуется зелёным на debug-оверлее). Блок паразитный, если выполнено ЛЮБОЕ:

    H1 — площадь блока больше ``max_book_area_frac`` площади книги (целая пустая
         страница ≈ половина разворота; нормальные блоки — куски страницы, мельче).
    H2 — блок раздувает грубый axis-aligned bbox книги более чем на
         ``max_bbox_growth_frac``: берём объединение bbox книги и bbox блока (с
         запасом ``layout_pad_px``, как при защите/рисовании) и сравниваем площадь с
         площадью одного bbox книги. Ловит блоки, вылезшие за пределы страниц.

    Возвращает список флагов, выровненный с ``layout_polys``.
    """
    if not layout_polys:
        return []
    book_area = int(np.count_nonzero(book_mask))
    bx, by, bw, bh = cv2.boundingRect((book_mask > 0).astype(np.uint8))
    book_bbox_area = bw * bh
    px, py = layout_pad_px if isinstance(layout_pad_px, (tuple, list)) else (layout_pad_px, layout_pad_px)

    flags: list[bool] = []
    for poly in layout_polys:
        area = float(cv2.contourArea(poly.reshape(-1, 1, 2).astype(np.float32)))
        # H1 — блок занимает слишком большую долю самой книги.
        h1 = book_area > 0 and area > max_book_area_frac * book_area
        # H2 — блок сильно раздувает грубый bbox книги.
        h2 = False
        if book_bbox_area > 0:
            xs, ys = poly[:, 0], poly[:, 1]
            lminx, lminy = xs.min() - px, ys.min() - py
            lmaxx, lmaxy = xs.max() + px, ys.max() + py
            uw = max(bx + bw, lmaxx) - min(bx, lminx)
            uh = max(by + bh, lmaxy) - min(by, lminy)
            growth = uw * uh / book_bbox_area - 1.0
            h2 = growth > max_bbox_growth_frac
        flags.append(bool(h1 or h2))
    return flags
