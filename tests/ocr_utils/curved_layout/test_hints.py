"""Вспомогательная информация на входе разбора: запретная маска, рамки-запреты, боковые области."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.engines.ink import InkEngine
from ocr_utils.curved_layout.from_layout import build
from ocr_utils.curved_layout.hints import LayoutHints, OrientedZone, barrier_rules, barrier_separators
from ocr_utils.curved_layout.orient import back_points
from ocr_utils.curved_layout.from_text_layer import hints_of
from ocr_utils.curved_layout.page import _forward_points, analyse_gray
from ocr_utils.text_layer_fix import VERSION
from ocr_utils.text_layer_fix.cache import cache_path, save_page
from tests.ocr_utils.curved_layout.synthetic import column_page

ENGINE = InkEngine()
SCALE = WORK_DPI / RENDER_DPI
# Имя PDF в кэше `text_layer_fix`: подпапка, в которой лежит JSON страницы.
CACHE_PDF = "doc.pdf"


def _work_size(page: np.ndarray) -> tuple[int, int]:
    """Размер рабочей копии страницы ``(ширина, высота)``."""
    return int(round(page.shape[1] * SCALE)), int(round(page.shape[0] * SCALE))


def test_forbidden_mask_leaves_no_text_where_it_is_not_allowed():
    """Краска под запретной маской не даёт ни строк, ни блоков."""
    page = column_page(columns=2)
    width, height = _work_size(page)
    allowed = np.ones((height, width), dtype=bool)
    allowed[:, : width // 2] = False  # левую половину объявляем растром
    hints = LayoutHints(text_allowed=allowed)
    plain = analyse_gray(page, ENGINE)
    masked = analyse_gray(page, InkEngine(hints=hints), hints=hints)
    assert len(plain.axes) > len(masked.axes) > 0
    assert all(np.asarray(axis.points)[:, 0].min() >= width // 2 - 2 for axis in masked.axes)


def test_barrier_edges_become_separators_and_rules():
    """Рамка-запрет даёт две вертикальные полосы запрета сцепки и две горизонтальные черты."""
    separators = barrier_separators(((100, 200, 300, 400),))
    rules = barrier_rules(((100, 200, 300, 400),))
    assert [(x0, x1) for x0, x1, _, _ in separators] == [(100, 101), (299, 300)]
    assert all(y0 == 200 and y1 == 400 for _, _, y0, y1 in separators)
    assert [(rule.y0, rule.y1) for rule in rules] == [(200, 201), (399, 400)]


def test_sideways_zone_gives_vertical_axes():
    """Область с боковым текстом разбирается повёрнутой: оси выходят почти вертикальными.

    Прямая страница поворачивается на 90° против часовой (текст ложится боком), и область
    объявляется повёрнутой на 90 по часовой — столько нужно, чтобы вернуть текст в прямое
    положение.
    """
    upright_page = column_page(columns=1)
    sideways = np.rot90(upright_page, 1)  # против часовой: текст лёг боком
    width, height = _work_size(sideways)
    hints = LayoutHints(zones=(OrientedZone((0, 0, width, height), 90),))
    analysis = analyse_gray(sideways, InkEngine(), hints=hints)
    assert len(analysis.axes) >= 5
    for axis in analysis.axes:
        points = np.asarray(axis.points, dtype=np.float64)
        run = points[:, 0].max() - points[:, 0].min()
        rise = points[:, 1].max() - points[:, 1].min()
        assert rise > 3.0 * max(run, 1e-6), "ось боковой строки должна идти по вертикали"
        assert 0 <= points[:, 0].min() and points[:, 0].max() <= width
        assert 0 <= points[:, 1].min() and points[:, 1].max() <= height


def test_rotation_round_trip():
    """Прямое и обратное отображение точек взаимно обратны при всех поворотах."""
    points = np.array([[3.0, 7.0], [11.0, 2.0], [0.0, 0.0]])
    size = (40, 25)
    for rotate_cw in (0, 90, 180, 270):
        forward = _forward_points(points, size, rotate_cw)
        assert back_points(forward, size, rotate_cw) == pytest.approx(points)


def test_build_puts_the_whole_page_first():
    """Область «вся полоса» идёт первой, боковые врезки — за ней: последующие старше."""
    hints = build(forbidden=[], barriers=[(1, 2, 3, 4)], sideways=[(10, 10, 60, 200)], width=600, height=800)
    assert hints.zones[0].box == (0, 0, 600, 800) and hints.zones[0].rotate_cw == 0
    assert hints.zones[1].sideways and hints.zones[1].box == (10, 10, 60, 200)
    assert hints.barriers == ((1, 2, 3, 4),)


def test_build_drops_a_hair_thin_zone():
    """Врезка тоньше порога отдельной областью не делается: разбирать в ней нечего."""
    hints = build(forbidden=[], barriers=[], sideways=[(10, 10, 12, 200)], width=600, height=800)
    assert hints.zones == ()


def test_empty_hints_change_nothing():
    """Пустая подсказка даёт тот же разбор, что и её отсутствие."""
    page = column_page(columns=2)
    plain = analyse_gray(page, ENGINE)
    same = analyse_gray(page, InkEngine(hints=LayoutHints()), hints=LayoutHints())
    assert len(plain.axes) == len(same.axes) and len(plain.blocks) == len(same.blocks)


def _cache_dir(tmp_path, cells, version=VERSION, error=None, name=CACHE_PDF, index=0):
    """Папка кэша ``text_layer_fix`` с одной страницей и заданными ячейками.

    ``raster_dpi`` ставится равным ``WORK_DPI``, чтобы боксы ячеек задавались прямо в пикселях
    рабочей копии и тест не зависел от масштабирования.
    """
    payload = {"version": version, "raster_dpi": WORK_DPI, "tables": [{"cells": list(cells)}], "zones": []}
    if error is not None:
        payload["error"] = error
    save_page(cache_path(tmp_path, name, index), payload)
    return tmp_path


def _cell(box, rotate_cw=None):
    """Словарь ячейки кэша: бокс по центрам линеек и внутренность на пиксель внутрь."""
    x0, y0, x1, y1 = box
    return {"box": list(box), "inner": [x0 + 1, y0 + 1, x1 - 1, y1 - 1], "rotate_cw": rotate_cw}


def _work(page: np.ndarray) -> np.ndarray:
    """Рабочая копия страницы: тот же размер, что у координат подсказок."""
    width, height = _work_size(page)
    return cv2.resize(page, (width, height), interpolation=cv2.INTER_AREA)


def test_only_an_inked_cell_becomes_an_area(tmp_path):
    """Ячейка с текстом становится областью разбора, пустая — нет; боксы обеих идут в запреты."""
    page = column_page(columns=1)
    work = _work(page)
    inked, blank = (100, 100, 300, 200), (5, 5, 55, 55)
    cache = _cache_dir(tmp_path, [_cell(inked), _cell(blank)])
    hints = hints_of(CACHE_PDF, 0, cache, work)
    boxes = [zone.box for zone in hints.zones]
    assert inked in boxes and blank not in boxes
    assert hints.barriers == (inked, blank)
    # Первой идёт полоса целиком: старшинство — у последующих областей.
    assert boxes[0] == (0, 0, work.shape[1], work.shape[0])


def test_a_foreign_or_broken_cache_is_ignored(tmp_path):
    """Кэш чужой версии или с ошибкой разбора не меняет подсказок: страница пойдёт как прежде."""
    page = column_page(columns=1)
    work = _work(page)
    cell = [_cell((100, 100, 300, 200))]
    assert hints_of(CACHE_PDF, 0, _cache_dir(tmp_path / "old", cell, version=VERSION + 99), work).empty
    assert hints_of(CACHE_PDF, 0, _cache_dir(tmp_path / "bad", cell, error="сбой"), work).empty
    assert hints_of(CACHE_PDF, 0, tmp_path / "нет-такой-папки", work).empty


@pytest.mark.parametrize("rotate_cw, vertical", [(90, True), (180, False)])
def test_a_rotated_cell_is_parsed_turned_upright(tmp_path, rotate_cw, vertical):
    """Ячейка с поворотом разбирается выпрямленной: у 90 оси вертикальны, у 180 — горизонтальны.

    Страница поворачивается против часовой (90) или на пол-оборота (180), и ячейка объявляется
    повёрнутой ровно настолько, чтобы текст вернулся в прямое положение.
    """
    turns = 1 if rotate_cw == 90 else 2
    page = np.rot90(column_page(columns=1), turns)
    work = _work(page)
    box = (0, 0, work.shape[1], work.shape[0])
    cache = _cache_dir(tmp_path, [_cell(box, rotate_cw=rotate_cw)])
    hints = hints_of(CACHE_PDF, 0, cache, work)
    analysis = analyse_gray(page, InkEngine(), hints=hints)
    assert len(analysis.axes) >= 5
    for axis in analysis.axes:
        points = np.asarray(axis.points, dtype=np.float64)
        run = points[:, 0].max() - points[:, 0].min()
        rise = points[:, 1].max() - points[:, 1].min()
        assert (rise > 3.0 * max(run, 1e-6)) == vertical


def test_a_block_stays_inside_its_cell(tmp_path):
    """Две ячейки одна над другой: каждый блок лежит в своей, через ребро ячейки не перетекает."""
    page = column_page(columns=1)
    work = _work(page)
    height, width = work.shape[:2]
    top, bottom = (40, 40, width - 40, height // 2), (40, height // 2, width - 40, height - 40)
    cache = _cache_dir(tmp_path, [_cell(top), _cell(bottom)])
    hints = hints_of(CACHE_PDF, 0, cache, work)
    analysis = analyse_gray(page, InkEngine(), hints=hints)
    assert len(analysis.blocks) >= 2
    for block in analysis.blocks:
        polygon = np.asarray(block.envelope.polygon, dtype=np.float64)
        box = (polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max())
        assert any(
            box[0] >= cell[0] - 2 and box[1] >= cell[1] - 2 and box[2] <= cell[2] + 2 and box[3] <= cell[3] + 2
            for cell in (top, bottom)
        ), f"блок {box} вышел за свою ячейку"
