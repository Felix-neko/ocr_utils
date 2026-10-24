"""Тесты второго преобразования на синтетике: поле снимает синусную волну строк, дубли с разными целями усредняются, без данных поле нулевое, поле гладкое внутри страницы и не затекает через корешок; выравнивание освещения выравнивает фон и не трогает текст."""

import numpy as np

from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from ocr_utils.scan_cropping.v2.rectify.illumination import equalize
from ocr_utils.scan_cropping.v2.rectify.coons import PageRect as Rect
from ocr_utils.scan_cropping.v2.rectify.text_field import (
    RuleAxis,
    SampleKind,
    Samples,
    FieldMode,
    PageRegion,
    PageText,
    RegionKindR,
    band_fits_page,
    band_profile,
    band_samples,
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
    рядом — полоса; одна линейка без строк текста, короткая строка или «строка» с рваной осью — без поля."""
    assert field_mode(8, sloped_lines([100.0 + 30 * i for i in range(8)], 1.0), DPI) is FieldMode.FULL
    assert field_mode(3, sloped_lines([120.0, 380.0, 640.0], 1.0), DPI) is FieldMode.SPARSE
    assert field_mode(1, sloped_lines([120.0], 1.0), DPI) is FieldMode.BAND
    assert field_mode(2, sloped_lines([120.0, 300.0], 1.0), DPI) is FieldMode.BAND  # 15 мм по высоте
    assert field_mode(1, sloped_lines([120.0], 1.0), DPI, text_lines=0) is FieldMode.NONE  # одна линейка
    assert field_mode(1, sloped_lines([120.0], 1.0, x1=400.0), DPI) is FieldMode.NONE  # 25 мм < 40
    noisy = sloped_lines([120.0], 0.0)
    noisy = Samples(noisy.at, noisy.value + np.where(np.arange(len(noisy)) % 2, 6.0, -6.0), *noisy.astuple()[2:])
    assert field_mode(1, noisy, DPI) is FieldMode.NONE  # СКО 0.5 мм — край листа, не строка
    assert field_mode(0, Samples.empty(), DPI) is FieldMode.NONE


def test_band_field_one_line():
    """Режим полосы: одна строка 51 мм внизу левой страницы с наклоном 1.5° и прогибом 0.6 мм выпрямляется, а вдали
    поле — её изгиб, затухающий по высоте (в 76 мм выше — доля exp(−(76/170)²) ≈ 0.82), без волн; поперёк (dx) — нули;
    правая страница не задета."""
    lay = layout()
    xs = np.arange(100.0, 710.0, 10.0)
    sag = 0.6 * DPI / 25.4
    yy = 1000.0 + np.tan(np.radians(1.5)) * (xs - 100) + sag * (1 - ((xs - 400) / 300) ** 2)
    target = float(np.median(yy))
    n = len(xs)
    v = Samples(np.stack([xs, np.full(n, target)], 1), yy - target, np.ones(n), np.stack([xs, yy], 1),
                np.full(n, SampleKind.LINE), np.zeros(n, int))  # fmt: skip
    assert field_mode(1, v, DPI) is FieldMode.BAND
    profile = band_profile(v, DPI)
    assert np.abs(profile.at(xs) - v.value).max() < 0.05 * DPI / 25.4  # многочлен 2-й степени ловит наклон и прогиб
    zeros = zero_samples(lay.rects[0], DPI)
    v_all = v + band_samples(zeros, v, profile, DPI) + zero_samples(lay.rects[1], DPI)
    h_all = far_zeros(zeros, Samples.empty(), DPI) + zero_samples(lay.rects[1], DPI)
    field = fit_field(h_all, v_all, lay, DPI, 4.0, 10.0)
    dx, dy = field.dense(W, H)
    # Строка выпрямлена: остаток после поля (без постоянной) — меньше 10 % размаха.
    pred = dy[np.round(v.at[:, 1]).astype(int), xs.astype(int)]
    rest = (v.value - pred) - (v.value - pred).mean()
    assert np.ptp(rest) < 0.1 * np.ptp(v.value)
    # Вдали (в 76 мм выше строки) — тот же изгиб с долей ≈ 0.82, гладкий: от многочлена 2-й степени — до 0.1 мм (у
    # конца строки многочлен переходит в продление прямой).
    far_y = int(target - 76 * DPI / 25.4)
    far = dy[far_y, 100:700]
    share = float(np.polyfit(profile.at(np.arange(100, 700.0)) - profile.at(np.arange(100, 700.0)).mean(),
                             far - far.mean(), 1)[0])  # fmt: skip
    assert 0.7 < share < 0.95
    cols = np.arange(100, 700.0)
    assert np.abs(far - np.polyval(np.polyfit(cols, far, 2), cols)).max() < 0.1 * DPI / 25.4
    assert np.abs(dx[:, : W // 2 - 40]).max() < 0.05 * DPI / 25.4
    # Правая страница — за полосой смешивания у корешка (до 8 мм от него) — не задета.
    assert np.abs(dy[:, W // 2 + 120 :]).max() < 0.05 * DPI / 25.4


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


def test_side_tilt_rejects_slanted_sides():
    from ocr_utils.scan_cropping.v2.rectify.text_field import SIDE_MAX_TILT_DEG, side_tilt

    ys = np.linspace(0, 300, 20)
    upright = np.stack([100 + 0.01 * ys, ys], 1)  # 0.6° — настоящая отбивка
    slanted = np.stack([100 + 0.08 * ys, ys], 1)  # 4.6° — центрованные строки ячейки
    assert side_tilt(upright) <= SIDE_MAX_TILT_DEG < side_tilt(slanted)


def test_raised_start_span_superscript():
    """Цифра сноски в начале строки (низ на 0.4 строчной выше базы, за ней пробел, база к ней поднята) — участок у
    начала строки; без знака, у тире в начале и при загибе строки (подняты и знак, и соседи) участка нет."""
    from types import SimpleNamespace

    from ocr_utils.page_layout.text_blocks.baseline_axis import SMOOTH_XH
    from ocr_utils.scan_cropping.v2.rectify.text_field import raised_start_span

    letters = np.array([[x, 100, x + 7, 110] for x in range(40, 400, 9)], float)  # строчные: высота 10, база 110
    base = np.stack([np.arange(20.0, 400.0, 6.0), np.full(64, 110.0)], 1)
    assert raised_start_span(SimpleNamespace(glyphs=letters), base) is None
    mark = np.array([[22, 96, 26, 106]], float)  # «³»: высота 10, низ на 4 px выше базы, пробел 14 px
    pulled = base.copy()
    pulled[:8, 1] -= np.linspace(4, 0, 8)  # сплайн базы поднят к знаку
    span = raised_start_span(SimpleNamespace(glyphs=np.vstack([mark, letters])), pulled)
    assert span is not None and span[0] < 20 and span[1] == 26 + SMOOTH_XH * 10
    dash = np.array([[22, 104, 30, 106]], float)  # тире: высота 0.2 строчной
    assert raised_start_span(SimpleNamespace(glyphs=np.vstack([dash, letters])), pulled) is None
    # Загиб начала строки: низы глифов поднимаются к началу линейно — знак на продолжении прямой соседей.
    curl = np.vstack([mark, letters])
    curl[:, [1, 3]] -= np.clip((80 - curl[:, :1]) * 0.1, 0, None)
    assert raised_start_span(SimpleNamespace(glyphs=curl), pulled) is None


def test_band_fits_page():
    """Режим полосы годится для пустой страницы со строкой (и для фото вдали от неё), но не для страницы со схемой или
    повёрнутым текстом вдали от строки и не для «строки» у нижнего края листа."""
    rect = PageRect(40, 40, W / 2, H - 40)
    v = sloped_lines([900.0], 1.0, 100.0, 700.0)
    cell = 300.0 / 25.4  # клетка карты краски — 1 мм
    ink = np.zeros((int(H / cell), int(W / cell)), np.float32)
    ink[76:78, 9:59] = 0.3  # сама строка
    page = PageText(None, np.zeros(2), 1.0, ink=ink)
    assert band_fits_page(page, v, rect, DPI)
    # Схема вдали от строки: 40 × 30 мм с долей краски 0.15 — 180 мм².
    scheme = ink.copy()
    scheme[10:40, 10:50] = 0.15
    assert not band_fits_page(PageText(None, np.zeros(2), 1.0, ink=scheme), v, rect, DPI)
    # Та же краска — фото (растр): не в счёт.
    photo = (PageRegion(RegionKindR.RASTER, (10 * cell, 10 * cell, 50 * cell, 40 * cell)),)
    assert band_fits_page(PageText(None, np.zeros(2), 1.0, regions=photo, ink=scheme), v, rect, DPI)
    # «Строка» в 3 мм от низа страницы — край листа.
    assert not band_fits_page(page, sloped_lines([H - 40 - 35.0], 0.0, 100.0, 700.0), rect, DPI)
