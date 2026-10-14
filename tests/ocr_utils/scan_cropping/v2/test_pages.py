"""Тесты страниц scan_cropping v2 на синтетике: трапеция, класс «плоско / приподнята», защита контента, кэш surya."""

import json

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.content_guard import protect_content
from ocr_utils.scan_cropping.v2.page_pose import PageFeatures, PagePose, PageQuad, classify_poses
from ocr_utils.scan_cropping.v2.pipeline import LAYOUT_VERSION, read_layout, read_layout_labels
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad


def features(keystone: float = 4.0) -> PageFeatures:
    """Признаки страницы с заданным сходом сторон трапеции.

    Args:
        keystone: Сход сторон, °.

    Returns:
        Признаки.
    """
    return PageFeatures(fill=0.98, keystone=keystone, width_share=1.0, light_share=1.0, finger=0.0, curl=0.5)


def test_trapezoid_vertices():
    """Трапеция маски находится с точностью до 2 px по каждой вершине."""
    truth = np.array([[100, 80], [520, 60], [560, 420], [80, 400]], np.float64)
    mask = np.zeros((500, 640), np.uint8)
    cv2.fillPoly(mask, [truth.astype(np.int32)], 1)
    quad = fit_enclosing_quad(mask > 0)
    assert quad is not None
    for v in truth:
        assert np.min(np.linalg.norm(quad - v, axis=1)) <= 2.0


def test_poses_single_and_both():
    """Одна страница — плоско; сильный сход у обеих — обе плоско; сильный сход у одной — она приподнята."""
    assert classify_poses([features(30.0)]) == [PagePose.FLAT]
    assert classify_poses([features(30.0), features(28.0)]) == [PagePose.FLAT, PagePose.FLAT]
    assert classify_poses([features(25.0), features(4.5)]) == [PagePose.RAISED, PagePose.FLAT]
    assert classify_poses([features(3.7), features(12.9)]) == [PagePose.FLAT, PagePose.RAISED]
    assert classify_poses([features(4.0), features(6.5)]) == [PagePose.FLAT, PagePose.FLAT]


def page(mask: np.ndarray, pose: PagePose) -> PageQuad:
    """Страница с маской и классом (трапеция и признаки не нужны защите).

    Args:
        mask: Маска страницы.
        pose: Класс.

    Returns:
        Страница.
    """
    return PageQuad(mask, None, features(), pose)


def test_protect_flat_only_inside_zone():
    """Защита: блок плоской страницы добавляется только внутри «не фона без рук»; блок приподнятой — нет."""
    paper = np.zeros((200, 400), bool)
    paper[20:180, 20:190] = True
    paper[20:180, 210:380] = True
    left, right = paper.copy(), paper.copy()
    left[:, 200:] = False
    right[:, :200] = False
    zone = np.zeros_like(paper)
    zone[10:190, 10:390] = True
    # Блоки (px кадра = px сегментации при seg=1): у низа каждой страницы, свисают за бумагу и за зону.
    block_left = np.array([[40, 160], [180, 160], [180, 199], [40, 199]], np.float32)
    block_right = np.array([[230, 160], [370, 160], [370, 199], [230, 199]], np.float32)
    pages = [page(left, PagePose.FLAT), page(right, PagePose.RAISED)]
    added = protect_content(paper, [block_left, block_right], pages, zone, 1.0, (1, 1))
    assert added[180:190, 60:160].all()  # под бумагой левой страницы, внутри зоны
    assert not added[190:, :].any()  # вне зоны
    assert not added[:, 200:].any()  # приподнятая страница не защищается
    assert not (added & paper).any()


def test_read_layout_versions(tmp_path):
    """Прежний формат layout.json (список) — None, новый — полигоны."""
    old = tmp_path / "old.json"
    old.write_text(json.dumps([[[0, 0], [1, 0], [1, 1]]]))
    assert read_layout(old) is None
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"version": LAYOUT_VERSION, "blocks": [[[0, 0], [1, 0], [1, 1]]]}))
    polys = read_layout(new)
    assert polys is not None and len(polys) == 1 and polys[0].shape == (3, 2)
    assert read_layout(tmp_path / "missing.json") is None
    assert read_layout_labels(new) is None  # без типов


def test_read_layout_labels(tmp_path):
    """Версия 4 — блоки с типами; версия 3 — блоки читаются, типов нет."""
    path = tmp_path / "layout.json"
    block = [[0, 0], [1, 0], [1, 1]]
    path.write_text(json.dumps({"version": 4, "keystone": True, "blocks": [block, block], "labels": ["Text", "Table"]}))
    assert read_layout_labels(path) == ["Text", "Table"]
    path.write_text(json.dumps({"version": 3, "keystone": True, "blocks": [block]}))
    assert len(read_layout(path)) == 1 and read_layout_labels(path) is None


def test_protect_no_pages():
    """Без страниц (обложка) защита ничего не добавляет."""
    paper = np.zeros((100, 100), bool)
    zone = np.ones_like(paper)
    block = np.array([[10, 10], [60, 10], [60, 40], [10, 40]], np.float32)
    assert not protect_content(paper, [block], [], zone, 1.0, (1, 1)).any()


def test_long_rules_keep_thin_lines_not_words():
    """Длинная тонкая линия и контур рамки — крупная краска; плотный «сгусток слова» той же длины — нет."""
    from ocr_utils.scan_cropping.v2.ink_fill import long_rules

    ink = np.zeros((400, 600), bool)
    ink[50:54, 50:550] = True  # линия 4 px
    cv2.rectangle(ink.view(np.uint8), (100, 120), (500, 300), 1, 3)  # контур рамки
    ink[350:366, 50:200] = True  # «слово» высотой 16 px
    rules = long_rules(ink)
    assert rules[50:54, 50:550].all()
    assert rules[120, 300] and rules[300, 300]
    assert not rules[350:366, 50:200].any()


def test_straight_strips_dropped_frames_kept():
    """Одиночная прямая полоса (тень у торца) не закрашивается, замкнутая рамка — закрашивается."""
    from ocr_utils.scan_cropping.v2.ink_fill import drop_straight_strips

    bold = np.zeros((500, 700), bool)
    bold[20:480, 30:40] = True  # полоса 10 px вдоль края
    cv2.rectangle(bold.view(np.uint8), (150, 60), (650, 440), 1, 12)  # рамка
    kept = drop_straight_strips(bold)
    assert not kept[20:480, 30:40].any()
    assert kept[56:64, 300].all() and kept[300, 146:154].all()


def test_gain_kept_only_near_ink():
    """Добавка второго прохода принимается кусками, касающимися закрашенной краски."""
    from ocr_utils.scan_cropping.v2.ink_fill import touching_ink

    ink = np.zeros((200, 200), bool)
    ink[50:150, 50:55] = True
    added = np.zeros_like(ink)
    added[50:150, 55:80] = True  # поле за рамкой
    added[10:30, 150:190] = True  # колебание SAM 3 вдали
    kept = touching_ink(added, ink)
    assert kept[50:150, 55:80].all() and not kept[10:30, 150:190].any()
