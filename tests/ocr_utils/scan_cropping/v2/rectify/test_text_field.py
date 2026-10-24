"""Тесты второго преобразования на синтетике: поле снимает синусную волну строк, дубли с разными целями усредняются, без данных поле нулевое, поле гладкое внутри страницы и не затекает через корешок; выравнивание освещения выравнивает фон и не трогает текст."""

import numpy as np

from research.page_rectify.coons import Layout, PageRect
from research.page_rectify.illumination import equalize
from research.page_rectify.coons import PageRect as Rect
from research.page_rectify.text_field import (
    RuleAxis,
    SampleKind,
    Samples,
    FieldMode,
    barrier_cuts,
    far_zeros,
    field_mode,
    find_rules,
    fit_field,
    inner_rules,
    orphan_rules,
    zero_samples,
)

DPI = 300.0  # 11.8 px/мм
W, H = 1600, 1200


def layout() -> Layout:
    """Холст W×H с двумя страницами встык посередине."""
    return Layout(W, H, (PageRect(40, 40, W / 2, H - 40), PageRect(W / 2, 40, W - 40, H - 40)), W / 2)


def wave_lines(amplitude: float, period: float) -> Samples:
    """Выборки осей строк, изогнутых синусом ``amplitude · sin(2πx / period)``: цель — медианная высота строки."""
    parts = []
    for y0 in np.arange(100, H - 100, 40.0):
        xs = np.arange(80, W - 80, 30.0)
        ys = y0 + amplitude * np.sin(2 * np.pi * xs / period)
        target = np.median(ys)
        at = np.stack([xs, np.full_like(xs, target)], 1)
        src = np.stack([xs, ys], 1)
        parts.append(
            Samples(at, ys - target, np.ones(len(xs)), src, np.full(len(xs), SampleKind.LINE), np.zeros(len(xs), int))
        )
    return sum(parts, Samples.empty())


def test_field_follows_wave():
    """Волна строк с периодом 600 px (~50 мм) и размахом ±6 px (~0.5 мм) почти целиком снимается полем.

    Высота цели строки свободна, поэтому сравнение — с точностью до постоянного сдвига каждой строки: строка должна
    стать прямой, а где она встанет — решает поле.
    """
    v = wave_lines(6.0, 600.0)
    field = fit_field(Samples.empty(), v, layout(), DPI, 4.0, 10.0)
    dx, dy = field.dense(W, H)
    pred = dy[v.at[:, 1].round().astype(int), v.at[:, 0].round().astype(int)]
    # Доля снятой волны: дисперсия остатка за вычетом сдвига каждой строки против дисперсии самой волны.
    resid = np.concatenate(
        [(pred - v.value)[v.group == g] - (pred - v.value)[v.group == g].mean() for g in np.unique(v.group)]
    )
    wave = np.concatenate([v.value[v.group == g] - v.value[v.group == g].mean() for g in np.unique(v.group)])
    assert 1 - resid.var() / wave.var() > 0.9
    assert np.abs(dx).max() < 1e-3  # горизонтальных данных нет — dx остаётся нулём


def test_duplicates_average():
    """Одна строка из двух блоков с целями, расходящимися на 4 px, — поле встаёт посередине."""
    xs = np.arange(300, 1300, 30.0)
    at = np.stack([xs, np.full_like(xs, 600.0)], 1)
    ones = np.ones(len(xs))
    a = Samples(at, ones * 2.0, ones, at, np.full(len(xs), SampleKind.LINE), np.zeros(len(xs), int))
    b = Samples(at + [0, 0.5], ones * -2.0, ones, at, np.full(len(xs), SampleKind.LINE), np.zeros(len(xs), int))
    field = fit_field(Samples.empty(), a + b, layout(), DPI, 4.0, 20.0)
    _, dy = field.dense(W, H)
    assert np.abs(dy[600, 400:1200]).max() < 0.3


def test_no_data_zero_field():
    """Без выборок поле — ноль."""
    field = fit_field(Samples.empty(), Samples.empty(), layout(), DPI, 4.0, 20.0)
    dx, dy = field.dense(W, H)
    assert np.abs(dx).max() == 0 and np.abs(dy).max() == 0


def test_field_smooth():
    """Поле без рывков внутри страницы: соседние пиксели отличаются меньше чем на 0.2 px даже при волне ±6 px."""
    field = fit_field(Samples.empty(), wave_lines(6.0, 600.0), layout(), DPI, 4.0, 10.0)
    _, dy = field.dense(W, H)
    assert np.abs(np.diff(dy, axis=0)).max() < 0.2
    steps = np.abs(np.diff(dy, axis=1))
    split = int(np.ceil(W / 2))
    steps[:, split - 1] = 0  # шов на линии разреза — разрыв намеренный
    assert steps.max() < 0.2


def test_pages_independent():
    """Поле левой страницы не затекает в правую: без выборок справа поле правее полосы смешивания у корешка — ноль."""
    v = wave_lines(6.0, 600.0)
    left = v.take(v.at[:, 0] < W / 2 - 40)
    field = fit_field(Samples.empty(), left, layout(), DPI, 4.0, 10.0)
    _, dy = field.dense(W, H)
    # Правее полосы смешивания у корешка (справа текста нет — полоса до BLEND_MAX_MM) поле правой страницы — ноль.
    assert np.abs(dy[:, int(np.ceil(field.blend[1])) :]).max() < 1e-6
    assert np.abs(dy[:, : W // 2]).max() > 3


def test_equalize_flattens_background():
    """Фон с перепадом 150…240 после выравнивания — белый, буквы остаются тёмными, оттенок бумаги сохраняет сдвиг."""
    h, w = 600, 400
    ramp = np.linspace(150, 240, w, dtype=np.float32)[None, :].repeat(h, 0)
    img = np.stack([ramp - 10, ramp, ramp + 5], -1)
    img[200:210, 50:350] = 20  # «строка» текста
    img = np.clip(img, 0, 255).astype(np.uint8)
    out, background = equalize(img, np.ones((h, w), bool))
    paper = out[300:500, 20:380].astype(np.float32).mean(axis=2)
    assert paper.min() > 235
    assert out[200:210, 60:340].mean() < 80
    # Синий канал темнее красного и после выравнивания (сдвиг яркости, а не деление по каналам).
    assert out[400, 100, 0] <= out[400, 100, 2]


def test_find_rules():
    """Изогнутая горизонтальная черта и вертикальная линейка находятся, строка «букв» — нет; осевая черты — по ней."""
    dpi = 300.0
    img = np.full((900, 1200), 255, np.uint8)
    xs = np.arange(100, 1100)
    ys = (300 + 8 * np.sin(xs / 300)).round().astype(int)
    for x, y in zip(xs, ys):
        img[y - 1 : y + 2, x] = 20  # черта 3 px ≈ 0.25 мм, изгиб ±8 px
    img[400:850, 600:603] = 20  # вертикальная линейка 450 px ≈ 38 мм
    for x0 in range(100, 1100, 40):
        img[600:630, x0 : x0 + 25] = 20  # «буквы»: пробеги короче отрезка открытия
    rules = find_rules(img, dpi, np.zeros(2))
    axes = sorted(r.axis.value for r in rules)
    assert axes == [RuleAxis.HORIZONTAL.value, RuleAxis.VERTICAL.value]
    horizontal = next(r for r in rules if r.axis is RuleAxis.HORIZONTAL)
    expected = 300 + 8 * np.sin(horizontal.points[:, 0] / 300)
    assert np.abs(horizontal.points[:, 1] - expected).max() < 2.0


def test_find_rules_double_and_footnote():
    """Двойная линейка колонтитула (толстая 1 мм + тонкая под ней) и короткая черта над сноской (18 мм) находятся."""
    dpi = 300.0
    img = np.full((900, 1200), 255, np.uint8)
    img[200:212, 100:1100] = 20  # толстая черта 12 px ≈ 1.0 мм
    img[218:221, 100:1100] = 20  # тонкая под ней через 0.5 мм
    img[600:603, 100:313] = 20  # черта над сноской 213 px ≈ 18 мм
    rules = find_rules(img, dpi, np.zeros(2))
    ys = sorted(float(np.median(r.points[:, 1])) for r in rules if r.axis is RuleAxis.HORIZONTAL)
    assert len(ys) == 2
    assert 200 <= ys[0] <= 221 and abs(ys[1] - 601) < 2


def test_inner_rules_drop_edge():
    """Линейка у нижнего края страницы (край тени) отбрасывается, линейка внутри — остаётся."""
    dpi = 300.0
    img = np.full((900, 1200), 255, np.uint8)
    img[300:303, 100:1100] = 20  # внутри страницы
    img[880:883, 100:1100] = 20  # в 0.3 мм от низа прямоугольника 0…885
    rules = inner_rules(find_rules(img, dpi, np.zeros(2)), Rect(0, 0, 1200, 885), dpi)
    assert len(rules) == 1 and abs(np.median(rules[0].points[:, 1]) - 301) < 2


class _Axis:
    """Ось строки для :func:`barrier_cuts`: горизонтальная ломаная от ``x0`` до ``x1`` на высоте ``y``."""

    def __init__(self, x0: float, x1: float, y: float) -> None:
        self.points = np.array([[x0, y], [x1, y]], np.float64)
        self.x0, self.x1 = x0, x1


def test_barrier_cuts_between_wrap_and_full_width():
    """Линия раздела под рисунком — между последним рядом обтекания и первым рядом на всю ширину, а не по ребру рамки."""
    # Рисунок справа на y 100–300; обтекание слева (x 10–180) идёт до y 320, ниже — полная ширина (x 10–390).
    axes = [_Axis(10, 390, y) for y in (40, 60, 80)]
    axes += [_Axis(10, 180, y) for y in range(100, 321, 20)]
    axes += [_Axis(10, 390, y) for y in range(340, 421, 20)]
    cuts = barrier_cuts(((200, 95, 390, 300),), axes, 400.0)
    ys = sorted(c[0][1] for c in cuts)
    assert ys == [90.0, 330.0]
    assert all(c[0][0] == 0.0 and c[1][0] == 400.0 for c in cuts)


def sloped_lines(ys: list[float], slope_deg: float, x0: float = 100.0, x1: float = 700.0) -> Samples:
    """Выборки осей строк левой страницы с наклоном ``slope_deg``: цель — медианная высота строки."""
    parts = []
    for y0 in ys:
        xs = np.arange(x0, x1, 30.0)
        yy = y0 + np.tan(np.radians(slope_deg)) * (xs - x0)
        target = np.median(yy)
        n = len(xs)
        at = np.stack([xs, np.full(n, target)], 1)
        parts.append(Samples(at, yy - target, np.ones(n), np.stack([xs, yy], 1), np.full(n, SampleKind.LINE),
                             np.zeros(n, int)))  # fmt: skip
    return sum(parts, Samples.empty())


def test_field_mode():
    """Режим страницы: 8 строк — полный; три длинные строки на 44 мм по высоте — разреженный; одна строка или две
    рядом — без поля."""
    assert field_mode(8, sloped_lines([100.0 + 30 * i for i in range(8)], 1.0), DPI) is FieldMode.FULL
    assert field_mode(3, sloped_lines([120.0, 380.0, 640.0], 1.0), DPI) is FieldMode.SPARSE
    assert field_mode(1, sloped_lines([120.0], 1.0), DPI) is FieldMode.NONE
    assert field_mode(2, sloped_lines([120.0, 300.0], 1.0), DPI) is FieldMode.NONE  # 15 мм по высоте


def test_sparse_field_near_lines_only():
    """Разреженный режим: наклон трёх строк поле снимает, а вдали от них (в 30 мм ниже последней) гаснет к нулю."""
    lay = layout()
    v = sloped_lines([120.0, 380.0, 640.0], 2.0)
    zeros = zero_samples(lay.rects[0], DPI) + zero_samples(lay.rects[1], DPI)
    field = fit_field(far_zeros(zeros, Samples.empty(), DPI), v + far_zeros(zeros, v, DPI), lay, DPI, 4.0, 10.0)
    _, dy = field.dense(W, H)
    pred = dy[v.at[:, 1].round().astype(int), v.at[:, 0].round().astype(int)]
    resid = np.concatenate(
        [(pred - v.value)[v.group == g] - (pred - v.value)[v.group == g].mean() for g in np.unique(v.group)]
    )
    tilt = np.concatenate([v.value[v.group == g] - v.value[v.group == g].mean() for g in np.unique(v.group)])
    assert 1 - resid.var() / tilt.var() > 0.8
    far_y = int(640 + 30 * DPI / 25.4)
    assert np.abs(dy[far_y, 100:700]).max() < 0.1 * DPI / 25.4


def test_closed_frame_sides_are_rules():
    """Замкнутая рамка вокруг текста: 4 стороны — линейки-рамки, проходят и «сирот», и «край» (снаружи бумага)."""
    dpi = 300.0
    img = np.full((1300, 1100), 255, np.uint8)
    img[100:104, 100:1000] = 20
    img[1196:1200, 100:1000] = 20
    img[100:1200, 100:104] = 20
    img[100:1200, 996:1000] = 20
    for y in range(200, 1100, 60):
        for x0 in range(160, 940, 40):
            img[y : y + 25, x0 : x0 + 25] = 20  # «буквы» внутри рамки
    rules = orphan_rules(find_rules(img, dpi, np.zeros(2)), img, np.zeros(2), dpi, ())
    assert len(rules) == 4 and all(r.frame for r in rules)
    # Прямоугольник страницы вплотную к рамке (стороны ближе 4 мм к краю): снаружи бумага — рамка остаётся.
    kept = inner_rules(rules, Rect(60, 60, 1040, 1240), dpi, img, np.zeros(2))
    assert len(kept) == 4
    # Снаружи — тёмный фон (край листа): стороны у края отбрасываются.
    dark = img.copy()
    dark[:90], dark[1210:], dark[:, :90], dark[:, 1010:] = 0, 0, 0, 0
    assert len(inner_rules(rules, Rect(60, 60, 1040, 1240), dpi, dark, np.zeros(2))) == 0
