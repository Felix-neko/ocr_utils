"""Тесты границ растрового фото (``page_layout.raster.photo_edges``) на синтетике: снимок с растровой сеткой на бумаге и подписью рядом — рамка по краю и кривой четырёхугольник; светлый снимок, виньетка, изогнутая сторона, два снимка под одной затравкой, плотная подпись — не фото."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.page_layout.raster.photo_edges import (
    EdgeClass,
    Side,
    find_photos,
    looks_like_text_block,
    robust_poly,
    step_on_profile,
)

DPI = 300.0
PX_PER_MM = DPI / 25.4


def side_deltas_mm(box, ref) -> dict:
    """Промах сторон рамки против истинной, мм со знаком (плюс — снаружи)."""
    k = 25.4 / DPI
    return {Side.TOP: (ref[1] - box[1]) * k, Side.BOTTOM: (box[3] - ref[3]) * k,
            Side.LEFT: (ref[0] - box[0]) * k, Side.RIGHT: (box[2] - ref[2]) * k}  # fmt: skip


def mm(v: float) -> int:
    """мм → px синтетики."""
    return int(round(v * PX_PER_MM))


def paper(h_mm: float = 160, w_mm: float = 140, seed: int = 0) -> np.ndarray:
    """Бумага 252 с лёгким шумом."""
    rng = np.random.default_rng(seed)
    return np.clip(252 + rng.normal(0, 1.5, (mm(h_mm), mm(w_mm))), 0, 255).astype(np.float32)


def halftone(h: int, w: int, seed: int = 1, light: bool = False) -> np.ndarray:
    """Растровый снимок: плавный тон с растровой сеткой периода 0.45 мм.

    Args:
        h, w: Размер, px.
        seed: Зерно.
        light: Светлый снимок (тон 200–235).

    Returns:
        Серый float32.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tone = 120 + 60 * np.sin(xx / w * 3.0) * np.cos(yy / h * 2.0)
    if light:
        tone = 218 + 15 * np.sin(xx / w * 3.0)
    period = 0.45 * PX_PER_MM
    screen = 0.5 + 0.5 * np.cos(2 * np.pi * xx / period) * np.cos(2 * np.pi * yy / period)
    return np.clip(tone + (screen - 0.5) * 60, 0, 255).astype(np.float32)


def put_text_line(img: np.ndarray, x0: int, y0: int, length: int, height: int = mm(1.8)) -> None:
    """Строка «текста»: чёрные буквы-прямоугольники с бумагой между ними."""
    x = x0
    while x < x0 + length:
        img[y0 : y0 + height, x : x + mm(0.9)] = 30
        x += mm(1.6)


def photo_page(light: bool = False, caption: bool = True) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Полоса с одним снимком 80×60 мм на (30, 40) мм и подписью под ним в 1.5 мм.

    Returns:
        ``(серый uint8, рамка снимка px)``.
    """
    img = paper()
    x0, y0, x1, y1 = mm(30), mm(40), mm(110), mm(100)
    img[y0:y1, x0:x1] = halftone(y1 - y0, x1 - x0, light=light)
    if caption:
        put_text_line(img, mm(35), y1 + mm(1.5), mm(70))
        put_text_line(img, mm(35), y1 + mm(5.0), mm(60))
    return img.astype(np.uint8), (x0, y0, x1, y1)


def test_sharp_photo_box_and_quad():
    """Снимок с резким краем: рамка до 0.3 мм от края, все стороны чёткие, подпись не захвачена."""
    img, true = photo_page()
    # Затравка как у детектора: шире снимка и с подписью.
    seed = (true[0] - mm(3), true[1] - mm(2), true[2] + mm(2), true[3] + mm(9))
    photos = find_photos(img, [seed], DPI)
    assert len(photos) == 1
    photo = photos[0]
    assert photo.edge_class is EdgeClass.SHARP
    deltas = side_deltas_mm(photo.box, true)
    assert all(abs(d) < 0.3 for d in deltas.values()), deltas
    assert photo.corners is not None
    for fit in photo.sides:
        assert fit.resid_p90_mm < 0.1


def test_light_photo_grows_to_edge():
    """Светлый снимок (тон 200–235): рамка всё равно по краю, а не по тёмному ядру."""
    img, true = photo_page(light=True)
    seed = (true[0] - mm(2), true[1] - mm(2), true[2] + mm(2), true[3] + mm(9))
    photo = find_photos(img, [seed], DPI)[0]
    deltas = side_deltas_mm(photo.box, true)
    assert all(abs(d) < 0.5 for d in deltas.values()), deltas


def test_vignette_is_not_sharp():
    """Виньетка (тон плавно уходит в бумагу на 6 мм): ни одна сторона не чёткая."""
    img = paper()
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, rx, ry = mm(70), mm(70), mm(35), mm(30)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    alpha = np.clip((1.25 - d) / 0.25, 0, 1)
    img = (img * (1 - alpha) + halftone(h, w) * alpha).astype(np.uint8)
    seed = (cx - rx - mm(4), cy - ry - mm(4), cx + rx + mm(4), cy + ry + mm(4))
    photo = find_photos(img, [seed], DPI)[0]
    assert photo.edge_class is not EdgeClass.SHARP


def test_curved_side_fit():
    """Изогнутый верх (прогиб 0.8 мм): подгонка второй степени ловит прогиб, остаток мал."""
    img = paper()
    x0, x1, y1 = mm(30), mm(110), mm(100)
    xs = np.arange(x0, x1)
    top = mm(40) + (mm(0.8) * (1 - ((xs - (x0 + x1) / 2) / ((x1 - x0) / 2)) ** 2)).astype(int)
    block = halftone(y1 - mm(38), x1 - x0)
    for i, x in enumerate(xs):
        img[top[i] : y1, x] = block[top[i] - mm(38) : y1 - mm(38), i]
    img = img.astype(np.uint8)
    photo = find_photos(img, [(x0 - mm(3), mm(36), x1 + mm(3), y1 + mm(3))], DPI)[0]
    top_fit = photo.side(Side.TOP)
    assert top_fit.bow_mm == pytest.approx(0.8, abs=0.2)
    assert top_fit.resid_p90_mm < 0.15


def test_two_photos_in_one_seed_split():
    """Два снимка с зазором 4 мм под одной затравкой — две находки."""
    img = paper()
    img[mm(20) : mm(70), mm(10) : mm(60)] = halftone(mm(70) - mm(20), mm(60) - mm(10))
    img[mm(20) : mm(70), mm(64) : mm(120)] = halftone(mm(70) - mm(20), mm(120) - mm(64), seed=2)
    photos = find_photos(img.astype(np.uint8), [(mm(8), mm(18), mm(122), mm(72))], DPI)
    assert len(photos) == 2


def test_step_on_profile_and_robust_poly():
    """Ступенька на профиле находится с точностью до доли пикселя; выброс не сбивает прямую."""
    s = np.arange(-40, 30, dtype=float)
    prof = np.where(s < 0.0, 80.0, 250.0)
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    pos, contrast, width = step_on_profile(prof[3:-3], s[3:-3], 250.0, 6, 14)
    assert abs(pos + 0.5) <= 1.0 and contrast > 150  # до 1 px (0.08 мм при 300 dpi): у ступеньки, сглаженной
    # коробкой, максимум производной плоский
    x = np.arange(100, dtype=float)
    y = 0.02 * x + 5
    y[50] += 30
    coef, inlier = robust_poly(x, y, 2, 1.0, 5.0)
    assert not inlier[50] and inlier.sum() == 99
    assert np.allclose(np.polyval(coef, x[inlier]), y[inlier], atol=0.05)


def test_dense_caption_is_not_photo():
    """Подпись в 4 строки, склеенная замыканием, — набор (межстрочные просветы), фото из неё не делается."""
    img = paper()
    # Набор 9–10 пт: строчные 2 мм, шаг строк 4 мм (межстрочный просвет 2 мм), буквы через 1.6 мм.
    for k in range(4):
        put_text_line(img, mm(30), mm(60) + k * mm(4.0), mm(80), height=mm(2.0))
    img = img.astype(np.uint8)
    seed = (mm(28), mm(58), mm(112), mm(78))
    photos = find_photos(img, [seed], DPI)
    assert len(photos) == 1 and not photos[0].sides  # затравка как есть, без сторон
    sky = np.full((mm(30), mm(60)), 230.0)
    assert not looks_like_text_block(sky, 252.0, DPI)


def test_one_sharp_side_gives_no_quad():
    """Снимок, у которого чёткая только одна сторона (остальные растворяются виньеткой), — без четырёхугольника:
    правило четырёх сторон."""
    img = paper()
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0, x1, y0, y1 = mm(30), mm(110), mm(40), mm(100)
    # Плавный спад к бумаге на 6 мм у верха, низа и правой стороны; левая — резкая.
    fade = np.minimum.reduce([np.clip((yy - y0) / mm(6), 0, 1), np.clip((y1 - yy) / mm(6), 0, 1),
                              np.clip((x1 - xx) / mm(6), 0, 1)])  # fmt: skip
    fade[:, :x0] = 0
    img = (img * (1 - fade) + halftone(h, w) * fade).astype(np.uint8)
    photo = find_photos(img, [(x0 - mm(3), y0 - mm(3), x1 + mm(3), y1 + mm(3))], DPI)[0]
    assert photo.corners is None and photo.edge_class is not EdgeClass.SHARP


def test_cutout_object_inside_box():
    """Вырезка без фона: тело снимка и тонкая стрела, уходящая за затравку вверх на 20 мм, — рамка охватывает стрелу,
    четырёхугольника нет, объекта за рамкой нет."""
    img = paper()
    x0, y0, x1, y1 = mm(30), mm(60), mm(90), mm(100)
    img[y0:y1, x0:x1] = halftone(y1 - y0, x1 - x0)
    # Стрела 0.6 мм толщиной, от тела вверх-вправо.
    for t in np.linspace(0, 1, 800):
        cx, cy = int(mm(70) + t * mm(15)), int(y0 - t * mm(20))
        img[cy : cy + mm(0.6), cx : cx + mm(0.6)] = 60
    img = img.astype(np.uint8)
    photo = find_photos(img, [(x0 - mm(2), y0 - mm(2), x1 + mm(2), y1 + mm(2))], DPI)[0]
    assert photo.box[1] <= y0 - mm(19.5) and photo.box[2] >= mm(85)
    assert photo.corners is None
    assert photo.outside_mm2["total"] <= 3.0


def test_light_sky_plate_grows_box():
    """Снимок со светлым небом (ровный тон на 8 уровней темнее бумаги) над тёмной частью: рамка доходит до верха неба."""
    img = paper()
    x0, x1, y_sky, y_dark, y1 = mm(30), mm(110), mm(30), mm(60), mm(100)
    img[y_sky:y_dark, x0:x1] = 244
    img[y_dark:y1, x0:x1] = halftone(y1 - y_dark, x1 - x0)
    photo = find_photos(img.astype(np.uint8), [(x0, y_dark - mm(2), x1, y1 + mm(2))], DPI)[0]
    assert abs(photo.box[1] - y_sky) < mm(1.0)


def test_spline_side_follows_wave():
    """Длинная сторона с волной (две полуволны по 0.3 мм на 120 мм): сплайн с узлом на 25 мм повторяет её с остатком
    меньше 0.06 мм, многочлен 2-й степени — нет."""
    from ocr_utils.page_layout.raster.photo_edges import SideModel, robust_poly, robust_spline

    a = np.arange(0, mm(120), mm(1.0)).astype(float)
    c = 500 + mm(0.3) * np.sin(2 * np.pi * a / mm(120))
    coef, inl = robust_poly(a, c, 2, mm(0.15), mm(0.6))
    poly = SideModel(np.asarray(coef))
    spline, _ = robust_spline(a, c, poly, inl, DPI, mm(0.15), mm(0.6))
    assert spline.knots is not None
    assert np.abs(spline(a) - c).max() < mm(0.06) < np.abs(poly(a) - c).max()
    # За краем — продолжение касательной (без разлёта кубики).
    assert abs(float(spline(np.array([a[-1] + mm(5)]))[0]) - c[-1]) < mm(1.0)


def test_smooth_shadow_and_text_do_not_grow_box():
    """Гладкая серая полоса (тень у края листа, 20 уровней ниже бумаги) слева от снимка и строка текста над ним в рамку
    не входят: рамка — по краю снимка."""
    img, true = photo_page(caption=False)
    img = img.astype(np.float32)
    img[:, mm(5) : true[0] - mm(1)] = np.minimum(img[:, mm(5) : true[0] - mm(1)], 231)
    put_text_line(img, true[0], true[1] - mm(3.2), mm(70))
    img = img.astype(np.uint8)
    photo = find_photos(img, [(true[0] - mm(3), true[1] - mm(4), true[2] + mm(2), true[3] + mm(2))], DPI)[0]
    deltas = side_deltas_mm(photo.box, true)
    assert deltas[Side.LEFT] < 0.5 and deltas[Side.TOP] < 0.5, deltas


def test_shade_wedge_left_of_photo_keeps_sharp_side():
    """Затенённая бумага (239) и гладкий серый клин тени (216) шириной 1.5 мм слева от снимка (1972/02 IMG_0054_1L):
    точки края — на кромке снимка, а не на внешнем краю клина; левая сторона чёткая, четырёхугольник есть."""
    img, true = photo_page(caption=False)
    img = img.astype(np.float32)
    img[:, : true[0]] = np.minimum(img[:, : true[0]], 239)
    img[:, true[0] - mm(1.5) : true[0]] = 216
    img = img.astype(np.uint8)
    photo = find_photos(img, [(true[0] - mm(3), true[1] - mm(2), true[2] + mm(2), true[3] + mm(2))], DPI)[0]
    left = photo.side(Side.LEFT)
    assert left.sharp(), (left.cover, left.resid_p90_mm, left.width_median_mm)
    assert photo.corners is not None
    assert abs(side_deltas_mm(photo.box, true)[Side.LEFT]) < 0.3


def test_diverging_corners_give_no_quad():
    """Крутые подгонки сторон (точки края по сору) — итерация углов расходится: углов нет, а не inf/NaN в рамке."""
    from ocr_utils.page_layout.raster.photo_edges import SideFit, SideModel, corners_of

    def fit(side: Side, slope: float, shift: float) -> SideFit:
        a = np.arange(10.0)
        return SideFit(side, a, a, a, a, np.ones(10, bool), SideModel(np.array([slope, shift])), (0.0, 9.0), 10, 0.1)

    sides = {Side.TOP: fit(Side.TOP, 3.0, 0.0), Side.BOTTOM: fit(Side.BOTTOM, 0.0, 100.0),
             Side.LEFT: fit(Side.LEFT, 3.0, 0.0), Side.RIGHT: fit(Side.RIGHT, 0.0, 100.0)}  # fmt: skip
    assert corners_of(sides, (0.0, 0.0, 100.0, 100.0)) is None


def test_pale_narrow_part_above_photo_is_covered():
    """Бледная растровая часть снимка уже стороны над ним (макушка кепки над белым небом, 1975/11 IMG_0054_1L):
    верх рамки накрывает её, хотя полоска над снимком покрыта растром на треть ширины."""
    img, true = photo_page(caption=False)
    img = img.astype(np.float32)
    # Часть 25 мм шириной и 3 мм высотой над серединой снимка: бледный растр — светлый фон 248 и отдельные точки
    # 3 px (0.25 мм) тоном 200 по узлам сетки 0.45 мм (как у макушки кепки: точек 16–35 % площади).
    px0, px1, py0 = true[0] + mm(28), true[0] + mm(53), true[1] - mm(3)
    img[py0 : true[1], px0:px1] = 248
    period = 0.45 * PX_PER_MM
    for cy in np.arange(py0 + period / 2, true[1], period):
        for cx in np.arange(px0 + period / 2, px1, period):
            y, x = int(round(cy)), int(round(cx))
            img[y - 1 : y + 2, x - 1 : x + 2] = 200
    img = img.astype(np.uint8)
    from ocr_utils.page_layout.raster.photo_edges import SEARCH_MM, grow_parts, page_maps

    seed = (true[0] - mm(3), true[1] - mm(5), true[2] + mm(2), true[3] + mm(2))
    window = (seed[0] - mm(SEARCH_MM), seed[1] - mm(SEARCH_MM), seed[2] + mm(SEARCH_MM), seed[3] + mm(SEARCH_MM))
    maps = page_maps(img, window, DPI, seed)
    # Верх нечёткий (у снимка 1975/11 — белое небо), остальные стороны чёткие и не растут.
    box = grow_parts(maps, tuple(float(v) for v in true), (False, True, True, True), DPI)
    assert box[1] <= py0 + mm(0.5), (box, py0)
    assert box[0] == true[0] and box[2] == true[2] and box[3] == true[3]
    # Гладкая тень над снимком (на 25 уровней темнее бумаги, без растра) рамку не тянет.
    plain, true2 = photo_page(caption=False)
    plain = plain.astype(np.float32)
    plain[true2[1] - mm(4) : true2[1], true2[0] + mm(28) : true2[0] + mm(53)] = 227
    maps2 = page_maps(plain.astype(np.uint8), window, DPI, seed)
    box2 = grow_parts(maps2, tuple(float(v) for v in true2), (False, True, True, True), DPI)
    assert box2[1] >= true2[1] - mm(0.3), (box2, true2)
