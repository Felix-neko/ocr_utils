"""Тесты продления кромки на синтетике: кромка едет вниз на заданный сдвиг с той же формой, бумага над старой кромкой не меняется, щель — бумага; раскладка и стирание слов."""

import numpy as np

from research.cut_lines import synth as S
from research.cut_lines.extend import BottomEdge, extend_bottom, full_page_mask, page_bottom_edge
from research.cut_lines.fragments import Strip, assign_columns, nominal_layout, paste_words, refine_layout, shared_shape


def curved_sheet(h: int = 400, w: int = 600, paper: int = 210) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Лист с изогнутой нижней кромкой на тёмном фоне, под кромкой — тень.

    Returns:
        ``(кадр BGR, маска бумаги, кромка по столбцам)``.
    """
    xs = np.arange(w)
    edge = 250 + 30 * np.sin(xs / w * np.pi)
    img = np.full((h, w, 3), 12, np.uint8)
    mask = np.zeros((h, w), bool)
    rows = np.arange(h)[:, None]
    mask = rows < edge[None, :]
    img[mask] = paper
    shadow = (rows >= edge[None, :]) & (rows < edge[None, :] + 20)
    img[shadow] = 40
    return img, mask, edge


def test_extend_moves_edge_with_same_shape() -> None:
    """После продления на d кромка (переход бумага → тень) лежит на d ниже в каждом столбце."""
    img, mask, edge = curved_sheet()
    d = 40
    out = extend_bottom(img, mask, d, below=30, feather=2).image
    gray = out.mean(axis=2)
    for x in (50, 300, 550):
        new_edge = np.nonzero(gray[:, x] < 100)[0][0]
        assert abs(new_edge - (edge[x] + d)) <= 3


def test_extend_keeps_paper_above_edge_and_fills_gap() -> None:
    """Пиксели бумаги выше старой кромки (минус полоса ободка) не меняются; щель — тон бумаги."""
    img, mask, edge = curved_sheet()
    d = 40
    out = extend_bottom(img, mask, d, below=30, feather=2).image
    rows = np.arange(img.shape[0])[:, None]
    untouched = rows < (edge[None, :] - 8)
    assert np.array_equal(out[untouched], img[untouched])
    gap = (rows >= edge[None, :] + 2) & (rows < edge[None, :] + d - 2)
    assert abs(float(out[gap].mean()) - 210) < 6


def test_extend_grows_canvas_when_edge_near_bottom() -> None:
    """Кромка у самого низа кадра: холст дорастает снизу на недостающие строки."""
    img, mask, edge = curved_sheet(h=300)
    ext = extend_bottom(img, mask, 60, below=30, feather=2)
    assert ext.pad > 0
    assert ext.image.shape[0] == img.shape[0] + ext.pad


def test_page_bottom_edge_refines_on_full_frame() -> None:
    """Кромка по маске ½ кадра уточняется на полном кадре по спаду яркости."""
    img, mask, edge = curved_sheet()
    half = mask[::2, ::2]
    gray = img.mean(axis=2).astype(np.uint8)
    found = page_bottom_edge(half, gray)
    assert found is not None
    mid = (found.x0 + found.x1) // 2
    assert abs(found.y[mid - found.x0] - edge[mid]) <= 3
    full = full_page_mask(half, found, gray.shape)
    assert full[int(edge[mid]) - 5, mid] and not full[int(edge[mid]) + 5, mid]


def glyph(width: int = 10, top: int = 14, height: int = 12) -> S.Glyph:
    """Прямоугольная буква: краска от базовой линии вверх на ``height``."""
    ratio = np.ones((top + 4, width + 2 * S.GLYPH_PAD, 3), np.float32)
    ratio[top - height : top, S.GLYPH_PAD : S.GLYPH_PAD + width] = 0.1
    return S.Glyph(ratio, top, width, "base")


def test_assign_columns_by_width_when_counts_differ() -> None:
    """Лишний узкий отрезок (номер страницы) отбрасывается, тексты идут слева направо."""
    cols = assign_columns([(0, 300), (320, 340), (400, 700)], ["левая", "правая"])
    assert [(c.x0, c.x1, c.text) for c in cols] == [(0, 300, "левая"), (400, 700, "правая")]


def test_justified_layout_reaches_right_edge() -> None:
    """С выключкой по формату последняя буква строки упирается в правый край, пробелы одинаковые."""
    lib = S.Library()
    for ch in "абв":
        lib.add(ch, glyph())
    placed, missing, missing_words = nominal_layout("ав ва ав", lib, 12.0, 0, 200, ragged=False)
    assert not missing and not missing_words
    last = placed[-1]
    assert last.u + S.GLYPH_PAD + last.glyph.width == 200
    starts = [p.u for p in placed if p.word in (1, 2)]
    gaps = [starts[0] - (placed[1].u + placed[1].glyph.width), starts[2] - (placed[3].u + placed[3].glyph.width)]
    assert gaps[0] == gaps[1]


def test_refine_layout_snaps_to_visible_tops() -> None:
    """Буквы находят свои видимые верхушки: номинальный сдвиг на 6 px исправляется."""
    lib = S.Library()
    lib.add("а", glyph())
    h, w = 20, 120
    img = np.full((h, w, 3), 200, np.uint8)
    base_v, top = 16, 14
    truth = [20, 36, 52]
    for u in truth:
        img[base_v - 12 : base_v, u : u + 10] = 20
    edge_v = np.full(w, base_v + 2.0)  # срезаны только самые низы букв
    strip = Strip(S.Axis(0, w, np.array([0.0, base_v]), 12.0), 0, (None, None), img, base_v, edge_v)
    placed = [S.Placed(u + 6 - S.GLYPH_PAD, lib.pick("а"), 0) for u in truth]
    refined, dv = refine_layout(placed, strip, 12.0)
    assert dv == 0
    assert [p.u + S.GLYPH_PAD for p in refined] == truth


def test_paste_words_replaces_cut_word_whole_and_keeps_visible() -> None:
    """Слово у кромки стирается целиком и набирается заново; слово выше кромки не трогается."""
    lib = S.Library()
    lib.add("а", glyph())
    h, w = 24, 160
    img = np.full((h, w, 3), 200, np.uint8)
    base_v = 18
    img[base_v - 12 : base_v, 20:30] = 20  # старая «буква» срезанного слова
    img[base_v - 12 : base_v, 100:110] = 20  # целое слово
    edge_v = np.full(w, base_v - 4.0)
    edge_v[80:] = h + 50.0  # правая часть строки целиком над кромкой
    frame = img.copy()
    axis = S.Axis(0, w, np.array([0.0, base_v]), 12.0)
    maps = S.strip_maps(axis, base_v, h - base_v, 0, w)
    strip = Strip(axis, 0, maps, img, base_v, edge_v)
    placed = [S.Placed(24 - S.GLYPH_PAD, lib.pick("а"), 0), S.Placed(100 - S.GLYPH_PAD, lib.pick("а"), 1)]
    out, box, replaced = paste_words(frame, strip, placed, set(), (0, w))
    assert replaced == 1
    assert out[base_v - 6, 21].mean() > 150  # старая буква стёрта
    assert out[base_v - 6, 28].mean() < 60  # новая на месте
    assert np.array_equal(out[:, 95:115], frame[:, 95:115])
    # Края окна не темнее бумаги (ремап без чёрного бордера): строки над и под окном.
    assert abs(float(out[1, 10:70].mean()) - 200) < 3 and abs(float(out[h - 1, 10:70].mean()) - 200) < 3


def test_paste_words_lays_glyphs_on_paper_with_grain() -> None:
    """Бумага под новыми буквами — поверхность окружения с зерном, а не плоский тон."""
    lib = S.Library()
    lib.add("а", glyph())
    h, w = 60, 200
    rng = np.random.default_rng(1)
    img = np.clip(200 + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    base_v = 54
    img[base_v - 12 : base_v, 20:30] = 20
    edge_v = np.full(w, base_v - 4.0)
    axis = S.Axis(0, w, np.array([0.0, base_v]), 12.0)
    strip = Strip(axis, 0, S.strip_maps(axis, base_v, h - base_v, 0, w), img, base_v, edge_v)
    placed = [S.Placed(24 - S.GLYPH_PAD, lib.pick("а"), 0)]
    out, _, replaced = paste_words(img.copy(), strip, placed, set(), (0, w))
    assert replaced == 1
    window = out[base_v - 10 : base_v + 4, 40:60].astype(float)  # залитая бумага справа от буквы
    assert abs(window.mean() - 200) < 5 and window.std() > 1.5


def test_axis_from_points_follows_edge_lift() -> None:
    """Точки «парабола + подъём у края»: кривая оси держит подъём (ошибка у края < 2 px)."""
    xs = np.arange(0, 1200, 3, dtype=float)
    truth = 100 + 0.00002 * (xs - 600) ** 2 - np.where(xs < 200, (200 - xs) * 0.08, 0.0)
    rng = np.random.default_rng(0)
    ys = truth + rng.normal(0, 0.7, len(xs))
    ys[::17] += 9  # выносные элементы
    axis = S.axis_from_points(xs, ys, 27.0, 0, 1200)
    assert axis is not None and axis.curve_x is not None
    err = np.abs(axis.y(xs) - truth)
    assert err[xs < 150].max() < 3.5 and np.median(err) < 1.0


def test_shared_shape_recovers_common_curve_and_pitch() -> None:
    """Четыре строки одной формы со сдвигами 50 px: общая кривая и сдвиги с постоянным шагом."""
    xs = np.arange(0, 900, 4, dtype=float)
    shape = 0.00003 * (xs - 450) ** 2 - np.where(xs < 150, (150 - xs) * 0.1, 0.0)
    rng = np.random.default_rng(2)
    rows = [(xs, 300 + k * 50 + shape + rng.normal(0, 0.8, len(xs))) for k in range(4)]
    found = shared_shape(rows, 27.0, 0, 900)
    assert found is not None
    curve, offsets = found
    pitches = np.diff(offsets)
    assert np.all(np.abs(pitches - 50) < 1.5)
    base = S.Axis(0, 900, np.array([0.0, 0.0]), 27.0, curve[0], curve[1])
    assert np.abs(base.y(xs) + offsets[0] - (300 + shape)).max() < 3.5


def test_extend_moves_gutter_gap_with_edge() -> None:
    """Две страницы с V-щелью у корешка: над новой бумагой тёмных пикселей нет, клин уехал на d вниз."""
    h, w, paper = 400, 600, 210
    img = np.full((h, w, 3), 10, np.uint8)
    rows = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    edge = np.where(xs < 300, 250 - np.maximum(0, 60 - (300 - xs)) * 1.0, 240.0)  # левая кромка поднимается к корешку
    mask = (rows < edge) & (xs != 300)
    img[mask] = paper
    d = 40
    ext = extend_bottom(img, mask, d, below=30, feather=2)
    out = ext.image.mean(axis=2)
    # Старое место клина, накрытое сдвинутой левой страницей (строки < кромка + d), — бумага.
    old_gap = (rows >= edge) & (rows < 240) & (xs >= 262) & (xs < 298)
    covered = old_gap & (rows < edge + d)
    assert out[covered].min() > 150
    # Клин на новом месте (сдвиг на d) — тёмный.
    new_gap_rows = np.nonzero(old_gap[:, 270])[0] + d
    assert out[new_gap_rows, 270].max() < 60


def test_extend_fills_frame_bottom_columns_with_paper() -> None:
    """Лист уходит за нижнюю рамку кадра: под старой высотой кадра на d строк бумага, без повторов буквы."""
    h, w, paper = 300, 400, 210
    img = np.full((h, w, 3), paper, np.uint8)
    img[h - 12 : h, 100:110] = 20  # буква, срезанная рамкой кадра
    mask = np.ones((h, w), bool)
    ext = extend_bottom(img, mask, 60, below=30, feather=2, frame_cols=np.ones(w, bool))
    assert ext.pad > 0
    below = ext.image[h + 2 : h + 58, 90:120].mean(axis=2)
    assert below.min() > 150


def test_extend_frame_columns_follow_neighbours() -> None:
    """Пробег столбцов у рамки между столбцами с настоящей кромкой: кромка в нём достраивается по соседям, под
    ней — полоса кромки того же тона, что у соседей."""
    h, w, paper = 400, 600, 210
    img = np.full((h, w, 3), 12, np.uint8)
    rows = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    edge = np.where((xs >= 200) & (xs < 400), h, 300.0)  # середина уходит за рамку
    mask = rows < edge
    img[mask] = paper
    frame_cols = (np.arange(w) >= 200) & (np.arange(w) < 400)
    ext = extend_bottom(img, mask, 40, below=30, feather=2, frame_cols=frame_cols)
    out = ext.image.mean(axis=2)
    # В середине видимая бумага доходит до рамки (h), поэтому кромка не выше h: бумага до h + 40, ниже — тёмная
    # полоса как у соседей.
    assert out[h + 1 : h + 38, 300].min() > 150
    assert out[h + 45 : h + 65, 300].max() < 60
