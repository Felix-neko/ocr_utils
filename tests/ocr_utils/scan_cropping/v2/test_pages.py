"""Тесты страниц scan_cropping v2 на синтетике: трапеция, класс «плоско / приподнята», защита контента, кэш surya."""

import json

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.content_guard import protect_content
from ocr_utils.scan_cropping.v2.page_pose import PageFeatures, PagePose, PageQuad, classify_poses
from ocr_utils.scan_cropping.v2.pipeline import LAYOUT_VERSION, read_layout, read_layout_labels
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad


def features(keystone: float = 4.0, outer_ratio: float = 1.0) -> PageFeatures:
    """Признаки страницы с заданным сходом сторон трапеции.

    Args:
        keystone: Сход сторон, °.
        outer_ratio: Знак схода — внешняя сторона трапеции к стороне у сгиба.

    Returns:
        Признаки.
    """
    return PageFeatures(
        fill=0.98, keystone=keystone, width_share=1.0, light_share=1.0, finger=0.0, curl=0.5, outer_ratio=outer_ratio
    )


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


def test_poses_sign_of_keystone():
    """Сильный сход у одной страницы: внешний край длиннее, чем у соседки, — приподнята; короче (≤ 0.95) — опущена.

    Числа — пак-6: МТС 1990/01 IMG_0875 (левая свисает, 0.92 при соседке 0.96), поднятая рукой — 1.03–1.45.
    """
    flat = features(3.7, 0.96)
    assert classify_poses([features(8.0, 0.92), flat]) == [PagePose.LOWERED, PagePose.FLAT]
    assert classify_poses([flat, features(25.0, 1.2)]) == [PagePose.FLAT, PagePose.RAISED]
    # Знака нет (обложка, разрезанная на две «страницы»: 0.98 при соседке 1.0) — как раньше, приподнята.
    assert classify_poses([features(11.0, 0.98), features(3.0, 1.0)]) == [PagePose.RAISED, PagePose.FLAT]
    # Короче соседки, но не короче порога — тоже без знака.
    assert classify_poses([features(9.0, 0.97), features(3.0, 0.99)]) == [PagePose.RAISED, PagePose.FLAT]


def test_outer_ratio_by_side():
    """Внешняя сторона — слева у левой страницы, справа у правой; у некрайней и без трапеции — 1."""
    from ocr_utils.scan_cropping.v2.page_pose import outer_ratio

    # Левая сторона 80 px, правая 100 px (по часовой: верх-лево, верх-право, низ-право, низ-лево).
    quad = np.array([[0, 10], [100, 0], [100, 100], [0, 90]], np.float64)
    assert abs(outer_ratio(quad, True) - 0.8) < 1e-6
    assert abs(outer_ratio(quad, False) - 1.25) < 1e-6
    assert outer_ratio(quad, None) == 1.0 and outer_ratio(None, True) == 1.0


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


def test_protect_lowered_like_flat():
    """Опущенная страница (внешний край свисает) защищается так же, как плоская."""
    paper = np.zeros((200, 400), bool)
    paper[20:180, 20:190] = True
    paper[20:180, 210:380] = True
    left, right = paper.copy(), paper.copy()
    left[:, 200:] = False
    right[:, :200] = False
    zone = np.zeros_like(paper)
    zone[10:190, 10:390] = True
    block_left = np.array([[40, 160], [180, 160], [180, 199], [40, 199]], np.float32)
    block_right = np.array([[230, 160], [370, 160], [370, 199], [230, 199]], np.float32)
    pages = [page(left, PagePose.LOWERED), page(right, PagePose.FLAT)]
    added = protect_content(paper, [block_left, block_right], pages, zone, 1.0, (1, 1))
    assert added[180:190, 60:160].all()  # опущенная левая — защищена
    assert added[180:190, 250:350].all()  # плоская правая — защищена
    assert not added[190:, :].any()


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


def test_select_instance_blank_page_needs_size_and_confidence():
    """Экземпляр без печати берётся, только если ≥ 30 % самого большого и уверенность ≥ 0.5; с печатью — всегда."""
    from ocr_utils.scan_cropping.v2.paper import select_instance

    hulls = np.zeros((100, 200), bool)
    hulls[10:90, 110:190] = True  # печать только на правой странице
    left = np.zeros((100, 200), bool)
    left[5:95, 5:95] = True  # пустая левая страница
    strip = np.zeros((100, 200), bool)
    strip[5:95, 0:4] = True  # полоса торца
    largest = int(left.sum())
    assert select_instance(left, 0.9, hulls, largest)
    assert not select_instance(left, 0.4, hulls, largest)
    assert not select_instance(strip, 0.9, hulls, largest)
    right = np.zeros((100, 200), bool)
    right[5:95, 105:195] = True
    assert select_instance(right, 0.3, hulls, largest)


def test_merge_inserts_sticker_joins_page_strip_does_not():
    """Наклейка внутри выпуклой оболочки страницы (с вырезом под неё) сливается; соседняя узкая страница — нет."""
    from ocr_utils.scan_cropping.v2.paper import merge_inserts

    page = np.zeros((200, 300), bool)
    page[10:190, 100:290] = True
    page[120:170, 100:180] = False  # вырез под наклейку, открытый к левому краю страницы
    sticker = np.zeros((200, 300), bool)
    sticker[120:170, 100:180] = True
    narrow = np.zeros((200, 300), bool)
    narrow[10:190, 60:95] = True  # узкая страница у корешка, рядом со страницей
    out = merge_inserts([(0.9, page), (0.5, sticker), (0.7, narrow)])
    assert len(out) == 2
    merged = max(out, key=lambda t: t[1].sum())[1]
    assert merged[140, 120] and merged[50, 200]
    assert any(m[50, 70] and not m[50, 200] for _, m in out)


def test_refine_colour_without_print():
    """Разворот без печати (пустые форзацы): уточнение склейки не падает, затравки цвета — по бумаге SAM 3."""
    import cv2

    from ocr_utils.scan_cropping.v2.glue_refine import refine_colour

    h, w = 500, 800
    bgr = np.full((h, w, 3), 15, np.uint8)
    book = np.zeros((h, w), bool)
    book[40:460, 60:740] = True
    bgr[book] = (60, 40, 30)
    base = np.zeros((h, w), bool)
    base[60:440, 80:385] = True
    base[60:440, 415:720] = True
    gap = np.zeros((h, w), bool)
    gap[60:440, 385:415] = True
    bgr[base | gap] = (200, 225, 235)
    bgr[60:90, 385:415] = (90, 50, 30)  # кусок обложки у края щели
    r = refine_colour(bgr, book, base, base | gap, np.zeros((h, w), bool), 70)
    assert not r.paper[70, 400] and r.paper[300, 400]
    assert cv2.countNonZero(r.paper.astype(np.uint8)) > 0


def pocket_scene():
    """Разворот на коврике: две светлые страницы, у наружных краёв — светлый торец стопки полосой 10 мм.

    Returns:
        ``(bgr, книга, фон, бумага, оболочки печати)`` на масштабе сегментации (~6.9 px/мм).
    """
    import cv2

    h, w = 1600, 2400
    bgr = np.full((h, w, 3), 15, np.uint8)
    book = np.zeros((h, w), bool)
    book[100:1500, 100:2300] = True
    bg = ~book
    bgr[book] = (60, 40, 30)  # крышка
    paper = np.zeros((h, w), bool)
    paper[150:1450, 220:1190] = True
    paper[150:1450, 1210:2180] = True
    bgr[paper] = (205, 225, 235)
    bgr[150:1450, 150:220] = (190, 215, 228)  # торец стопки слева — светлая полоса 10 мм
    hulls = np.zeros((h, w), bool)
    for x0 in (300, 1290):
        for y in range(250, 1350, 40):
            cv2.line(bgr, (x0, y), (x0 + 800, y), (30, 30, 30), 3)
        hulls[240:1360, x0 - 5 : x0 + 805] = True
    return bgr, book, bg, paper, hulls


def test_fill_page_pockets_corner_pocket_added_fore_edge_not():
    """Карман в углу страницы, не взятый SAM 3 (выемка, открытая наружу), добавляется; полоса торца — нет."""
    from ocr_utils.scan_cropping.v2.paper import fill_page_pockets

    bgr, book, bg, paper, hulls = pocket_scene()
    paper = paper.copy()
    paper[900:1450, 220:800] = False  # карман 84×80 мм в левом нижнем углу левой страницы
    bgr[900:1450, 220:800] = (215, 232, 240)
    added = fill_page_pockets(bgr, paper, book, bg, hulls)
    assert added[1200, 500] and added[1440, 230]
    assert not added[500, 180]  # торец выше кармана (рядом с карманом полоса торца может войти в рамку кармана)
    assert not added[120, 1200]  # крышка у корешка


def test_fill_page_pockets_insert_across_spine_keeps_right_page():
    """Вклейка через стык на низу правой страницы: вклейка и низ правой страницы — бумага, правая страница цела."""
    from ocr_utils.scan_cropping.v2.paper import fill_page_pockets

    bgr, book, bg, paper, hulls = pocket_scene()
    paper = paper.copy()
    paper[1000:1450, 1000:1700] = False  # SAM 3 вырезал вклейку из обеих страниц
    bgr[1000:1450, 1000:1700] = (225, 236, 242)
    added = fill_page_pockets(bgr, paper, book, bg, hulls)
    full = paper | added
    assert full[1300, 1500] and full[1300, 1100] and full[1440, 1650]


def test_fill_page_pockets_closed_book_cover_not_added():
    """Закрытая книга со светлой крышкой, бумаги почти нет: крышка — не карман."""
    from ocr_utils.scan_cropping.v2.paper import fill_page_pockets

    h, w = 1600, 2400
    bgr = np.full((h, w, 3), 15, np.uint8)
    book = np.zeros((h, w), bool)
    book[100:1500, 700:1700] = True
    bgr[book] = (215, 228, 235)  # светлая крышка
    paper = np.zeros((h, w), bool)
    paper[300:340, 900:960] = True  # пятнышко «бумаги» SAM 3
    assert not fill_page_pockets(bgr, paper, book, ~book, np.zeros((h, w), bool)).any()


class FakeSam3:
    """Подставной SAM 3: по каждой подсказке — заданные экземпляры ``(уверенность, маска)``."""

    def __init__(self, answers: dict) -> None:
        self.answers, self.asked = answers, []

    def set_image(self, bgr) -> None:
        pass

    def __call__(self, text):
        import torch
        from types import SimpleNamespace

        self.asked.append(text[0])
        found = self.answers.get(text[0], [])
        if not found:
            return [SimpleNamespace(masks=None)]
        masks = torch.tensor(np.stack([m for _, m in found]).astype(np.float32))
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=masks), boxes=SimpleNamespace(conf=torch.tensor([s for s, _ in found]))
            )
        ]


def test_sam3_fallback_prompt():
    """Запасная подсказка зовётся только при пустых основных и берёт обложку с печатью."""
    from ocr_utils.scan_cropping.v2.paper import sam3_pages

    cover = np.zeros((40, 30), bool)
    cover[5:35, 5:25] = True
    hulls = np.zeros_like(cover)
    hulls[10:20, 8:22] = True
    # Основные пусты, «magazine cover» ниже своего порога (крышка подшивки на пороге SAM 3) — ничего.
    fake = FakeSam3({"magazine cover": [(0.25, cover)]})
    raw, labels = sam3_pages(fake, np.zeros((40, 30, 3), np.uint8), hulls)
    assert not raw.any() and labels.max() == 0
    # Основные пусты — берётся «magazine cover» выше её порога.
    fake = FakeSam3({"magazine cover": [(0.35, cover)]})
    raw, labels = sam3_pages(fake, np.zeros((40, 30, 3), np.uint8), hulls)
    assert fake.asked == ["book page", "paper page", "magazine cover"]
    assert (raw == cover).all() and labels.max() == 1
    # Основные нашли страницу — запасная не зовётся.
    fake = FakeSam3({"book page": [(0.9, cover)], "magazine cover": [(0.9, ~cover)]})
    raw, _ = sam3_pages(fake, np.zeros((40, 30, 3), np.uint8), hulls)
    assert fake.asked == ["book page", "paper page"] and (raw == cover).all()


def test_sam3_fallback_on_small_share():
    """Основные взяли одну плашку (< 35 % книги) — добавляется «magazine cover», плашка сливается с обложкой."""
    from ocr_utils.scan_cropping.v2.paper import sam3_pages

    cover = np.zeros((40, 30), bool)
    cover[5:35, 5:25] = True
    plaque = np.zeros_like(cover)
    plaque[8:14, 8:20] = True
    hulls = plaque.copy()
    book = cover.copy()
    image = np.zeros((40, 30, 3), np.uint8)
    fake = FakeSam3({"book page": [(0.9, plaque)], "magazine cover": [(0.5, cover)]})
    raw, labels = sam3_pages(fake, image, hulls, book=book)
    assert fake.asked[-1] == "magazine cover" and (raw == cover).all() and labels.max() == 1
    # Без маски книги — прежнее поведение: что-то найдено, запасная не зовётся.
    fake = FakeSam3({"book page": [(0.9, plaque)], "magazine cover": [(0.5, cover)]})
    raw, _ = sam3_pages(fake, image, hulls)
    assert "magazine cover" not in fake.asked and (raw == plaque).all()
    # Основные накрыли книгу — запасная не зовётся.
    fake = FakeSam3({"book page": [(0.9, cover)], "magazine cover": [(0.5, cover)]})
    sam3_pages(fake, image, hulls, book=book)
    assert "magazine cover" not in fake.asked
