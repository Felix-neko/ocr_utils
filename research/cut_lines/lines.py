"""Срезанные краем листа строки: строки печати донора (2) после совмещения, заходящие за видимую бумагу основы (1); зона и альфа вклейки."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from research.cut_lines.register import VAR_MIN, local_ncc

# Краска донора: окно и сдвиг адаптивного порога (масштаб совмещения, ~6.9 px/мм), уровни серого.
INK_BLOCK, INK_OFFSET = 31, 18
# Смыкание краски в строки: высота и длина прямоугольника, px (межбуквенный и межсловный промежуток, не межстрочный).
LINE_CLOSE = (5, 25)
# Смыкание при разрезке слипшихся строк: тоньше по вертикали, короче вдоль, px.
SPLIT_CLOSE = (2, 15)
# Строка короче этой площади — сор, px².
LINE_MIN_AREA = 300
# Строка срезана, если за видимой бумагой (1) лежит не меньше этой доли её площади и не меньше CUT_MIN_PX px.
CUT_MIN_SHARE = 0.03
CUT_MIN_PX = 40
# Строка должна лежать в зоне своей страницы не меньше этой долей (чужая страница при плохом совмещении — нет).
OWN_PAGE_SHARE = 0.8
# Зона страницы: раздутие маски страницы (1) наружу, px (~6 см: все строки донора, целиком ушедшие за кромку, до поля).
PAGE_REACH = 420
# Корешок: полоса шириной в эту долю ширины страницы по обе стороны от её края у корешка — не дорисовывается
# (у корешка текст донора убит).
SPINE_FRAC = 0.08
# Сам сгиб: полуширина полосы, куда вклейка не заходит никогда, в долях ширины страницы.
FOLD_FRAC = 0.02
# Бумага донора: светлее этой доли медианы видимой бумаги (1) и хрома меньше предела (обложка красная, коврик тёмный).
PAPER2_LIGHT = 0.6
PAPER2_MAX_CHROMA = 25.0
# Шов в бумаге (1): срезанная строка раздувается на полстроки по вертикали и немного по горизонтали, px.
SEAM_REACH = (9, 15)
# Продление листа донора от срезанной строки к кромке листа (2): по вертикали, px.
EXTEND_REACH = 16
# Строка текста: толщина сгустка (медиана высоты по столбцам, px, масштаб совмещения ~6.9 px/мм), удлинение, глифы (площадь, высота) и их число.
LINE_HEIGHT = (6, 45)
# Строка толще этой доли типичной высоты буквы — две слипшиеся строки (режутся заново).
MAX_THICK_REL = 2.6
LINE_ASPECT = 3.0
GLYPH_AREA = (8, 900)
GLYPH_HEIGHT = (4, 30)
MIN_GLYPHS = 6
# Буква — не ниже этой доли типичной высоты буквы страницы (умолчание, если мерить не по чему); букв не реже, чем
# на каждые GLYPH_SPACING высот буквы.
GLYPH_REL_HEIGHT = 0.55
DEFAULT_GLYPH_HEIGHT = 12.0
GLYPH_SPACING = 4.0
# Проверка совмещения строки: полоса над и под строкой, px; минимум видимых пикселей с краской; порог медианы
# корреляции.
CHECK_REACH = 80
CHECK_MIN_PX = 150
LINE_NCC_MIN = 0.6
# Поле вокруг срезанной строки, где берётся краска донора (по вертикали, по горизонтали), px.
TEXT_PAD = (4, 8)
# Строки — одна колонка, если перекрываются по горизонтали не меньше этой доли более короткой.
ROW_OVERLAP = 0.3
# Наибольший промежуток между колонками, который смыкается на кромке, px.
COLUMN_GAP = 60
# Продление полосы строки вбок от её концов (в столбцах своей страницы), px.
SIDE_REACH = 60
# Сглаживание огибающей строки вдоль неё: окно медианы и σ гаусса, px.
PROFILE_MEDIAN = 41
PROFILE_SIGMA = 12.0
# Сглаживание стыка новой кромки с настоящей (радиус закрытия), px.
EDGE_SMOOTH = 30
# Размытие альфы внутри бумаги (1), px.
ALPHA_BLUR = 4.0


class Side(str, Enum):
    """Край страницы, которым срезана строка."""

    BOTTOM = "bottom"
    TOP = "top"
    OUTER = "outer"


@dataclass(frozen=True)
class CutLine:
    """Срезанная строка.

    Поля:
        side: Край страницы.
        share: Доля площади строки за видимой бумагой (1).
        box: Рамка строки ``(x0, y0, x1, y1)`` на масштабе совмещения.
    """

    side: Side
    share: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class CutResult:
    """Итог поиска по странице.

    Поля:
        lines: Срезанные строки.
        cut: Маска срезанных строк (bool).
        alpha: Альфа вклейки донора (float32 0..1).
        missing: Бумага донора за видимой бумагой (1) в зоне страницы (bool).
        extension: Продление листа (1) за его кромку — новая бумага с чёткой кромкой (bool).
        pieces: Куски строк ряда по отдельности (маски кадра на масштабе совмещения) — для синтеза по строкам.
        text: Где брать краску донора — срезанные строки с небольшим полем (bool); в остальной вклейке только бумага.
        paper2: Бумага донора в координатах (1) (bool).
    """

    lines: list[CutLine]
    cut: np.ndarray
    alpha: np.ndarray
    missing: np.ndarray
    extension: np.ndarray
    paper2: np.ndarray
    text: np.ndarray
    pieces: list[np.ndarray] = field(default_factory=list)


def ink_lines(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Строки печати: адаптивная краска, смыкание вдоль строки.

    Args:
        gray: Серое.

    Returns:
        ``(краска, строки)`` — булевы маски.
    """
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, INK_BLOCK, INK_OFFSET)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return ink > 0, cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones(LINE_CLOSE, np.uint8)) > 0


def glyph_height(ink: np.ndarray, region: np.ndarray) -> float:
    """Типичная высота буквы на странице: медиана высот пятен краски размера буквы в ``region``.

    Args:
        ink: Краска (bool).
        region: Где мерить (bool), обычно видимая бумага страницы.

    Returns:
        Высота, px (по умолчанию :data:`DEFAULT_GLYPH_HEIGHT`, если пятен мало).
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats((ink & region).astype(np.uint8), connectivity=8)
    area, gh = stats[1:, cv2.CC_STAT_AREA], stats[1:, cv2.CC_STAT_HEIGHT]
    sel = (area >= GLYPH_AREA[0]) & (area <= GLYPH_AREA[1]) & (gh >= GLYPH_HEIGHT[0]) & (gh <= GLYPH_HEIGHT[1])
    return float(np.median(gh[sel])) if sel.sum() >= 50 else DEFAULT_GLYPH_HEIGHT


def line_thickness(comp: np.ndarray) -> float:
    """Толщина сгустка-строки: медиана его высоты по столбцам.

    Высота рамки у длинной наклонённой или изогнутой строки (страница в одну колонку) в разы больше самой строки;
    толщина по столбцам от наклона не зависит.

    Args:
        comp: Маска сгустка в рамке (bool).

    Returns:
        Толщина, px.
    """
    cols = comp.sum(axis=0)
    cols = cols[cols > 0]
    return float(np.median(cols)) if len(cols) else 0.0


def is_text_line(comp: np.ndarray, ink: np.ndarray, x_height: float) -> bool:
    """Похож ли сгусток на строку текста: вытянут вдоль, в нём не меньше :data:`MIN_GLYPHS` букв высотой с буквы
    страницы, стоящих достаточно плотно.

    Торец стопки листов и кромка обложки тоже дают тёмные полосы, которые смыкаются в «строки»: вертикальные
    (вдоль торца) и горизонтальные (линии между листами у нижнего обреза). Пятна в них — мелкая крапинка, а не буквы.
    Высота глифа сравнивается с типичной буквой страницы, а не с высотой сгустка: у срезанной строки сгусток
    захватывает тень кромки и выходит выше.

    Args:
        comp: Маска сгустка в его рамке (bool).
        ink: Краска в той же рамке (bool).
        x_height: Типичная высота буквы страницы, px (:func:`glyph_height`).

    Returns:
        True — строка текста.
    """
    hgt, wid = line_thickness(comp), comp.shape[1]
    if not (LINE_HEIGHT[0] <= hgt <= min(LINE_HEIGHT[1], MAX_THICK_REL * x_height)) or wid < LINE_ASPECT * hgt:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats((ink & comp).astype(np.uint8), connectivity=8)
    area, gh = stats[1:, cv2.CC_STAT_AREA], stats[1:, cv2.CC_STAT_HEIGHT]
    # Буква — пятно размера буквы высотой с буквы страницы (с запасом на прописные и срезанные снизу).
    letter = (gh >= GLYPH_REL_HEIGHT * x_height) & (gh <= 2.5 * x_height)
    glyphs = (area >= GLYPH_AREA[0]) & (area <= GLYPH_AREA[1]) & letter
    # Плотность: в строке текста буква примерно на каждую высоту буквы; требуем хотя бы на каждые GLYPH_SPACING.
    return int(glyphs.sum()) >= max(MIN_GLYPHS, wid / (GLYPH_SPACING * x_height))


def donor_paper(warped: np.ndarray, reference: float) -> np.ndarray:
    """Бумага донора в координатах (1): светлое и малонасыщенное, сомкнутое, без дыр.

    Args:
        warped: Донор BGR после совмещения.
        reference: Медиана серого видимой бумаги (1) (донор приводится к ней по порогу доли).

    Returns:
        Маска бумаги (bool).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    light = (gray > PAPER2_LIGHT * reference) & (chroma < PAPER2_MAX_CHROMA)
    closed = cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    return ndimage.binary_fill_holes(closed)


def spine_zone(page: np.ndarray, other: np.ndarray | None, frac: float = SPINE_FRAC) -> np.ndarray:
    """Полоса у корешка страницы: столбцы около её края, обращённого к соседней странице.

    Args:
        page: Маска страницы (bool).
        other: Маска соседней страницы или None (тогда корешок не исключается).
        frac: Полуширина полосы в долях ширины страницы.

    Returns:
        Маска полосы (bool).
    """
    out = np.zeros(page.shape, bool)
    if other is None or not other.any() or not page.any():
        return out
    xs = np.nonzero(page.any(axis=0))[0]
    x0, x1 = int(xs[0]), int(xs[-1])
    # Корешок — с той стороны страницы, где соседняя.
    spine = x1 if np.nonzero(other.any(axis=0))[0].mean() > (x0 + x1) / 2 else x0
    half = int(frac * (x1 - x0))
    out[:, max(0, spine - half) : spine + half] = True
    return out


def side_of(part: np.ndarray, page: np.ndarray) -> Side:
    """Край страницы, ближайший к срезанной части строки.

    Args:
        part: Часть строки за видимой бумагой (bool).
        page: Маска страницы (bool).

    Returns:
        Край.
    """
    ys, xs = np.nonzero(page)
    cy, cx = np.nonzero(part)
    my, mx = cy.mean(), cx.mean()
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    # Расстояния до краёв; горизонтальные края — до левого и правого (какой из них наружный, решает корешок выше).
    d = {Side.TOP: my - y0, Side.BOTTOM: y1 - my, Side.OUTER: min(mx - x0, x1 - mx)}
    return min(d, key=d.get)


def smooth_profile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Сглаженный профиль вдоль строки: пропуски заполняются линейно, затем медиана и гауссово сглаживание.

    Args:
        values: Значения по позициям вдоль строки (float).
        valid: Где значение есть (bool).

    Returns:
        Сглаженный профиль той же длины.
    """
    idx = np.arange(len(values))
    filled = np.interp(idx, idx[valid], values[valid])
    filled = ndimage.median_filter(filled, size=PROFILE_MEDIAN, mode="nearest")
    return ndimage.gaussian_filter1d(filled, PROFILE_SIGMA, mode="nearest")


def line_band(comp: np.ndarray, side: Side, reach: int, page: np.ndarray) -> np.ndarray:
    """Полоса срезанной строки с гладкой внешней кромкой: от строки до сглаженной огибающей строки плюс поле ``reach``.

    Для нижнего края по каждому столбцу берётся нижняя точка строки, ряд сглаживается вдоль строки — кромка не
    повторяет хвосты букв. Для верхнего — верхняя точка, для наружного — крайняя по строкам.

    Args:
        comp: Маска строки во весь кадр (bool).
        side: Край страницы.
        reach: Поле за огибающей, px.
        page: Маска своей страницы (bool): вбок полоса продлевается на :data:`SIDE_REACH` px, но не дальше столбцов
            (для наружного края — строк) страницы, чтобы новая кромка сходилась с настоящей без уступа.

    Returns:
        Маска полосы (bool).
    """
    h, w = comp.shape
    if side == Side.OUTER:
        side_t = Side.BOTTOM if comp.any(axis=0).nonzero()[0].mean() > w / 2 else Side.TOP
        return line_band(comp.T, side_t, reach, page.T).T
    out = np.zeros((h, w), bool)
    cols = np.nonzero(comp.any(axis=0))[0]
    # Вбок — до SIDE_REACH px, но только в столбцах страницы.
    page_cols = page.any(axis=0)
    x0, x1 = max(0, int(cols[0]) - SIDE_REACH), min(w, int(cols[-1]) + 1 + SIDE_REACH)
    sub = comp[:, x0:x1]
    has = sub.any(axis=0)
    top = np.where(has, np.argmax(sub, axis=0), 0).astype(float)
    bottom = np.where(has, h - 1 - np.argmax(sub[::-1], axis=0), 0).astype(float)
    top, bottom = smooth_profile(top, has), smooth_profile(bottom, has)
    # Внешняя огибающая (кромка нового листа): строка плюс поле.
    edge = bottom + reach if side == Side.BOTTOM else top - reach
    # За концами строки огибающая плавно (smoothstep на SIDE_REACH px) сходит к настоящей кромке страницы (1) в этом
    # столбце — новая кромка встречает старую без ступеньки.
    sub_page = page[:, x0:x1]
    page_has = sub_page.any(axis=0)
    if side == Side.BOTTOM:
        own = np.where(page_has, h - 1 - np.argmax(sub_page[::-1], axis=0), edge).astype(float)
    else:
        own = np.where(page_has, np.argmax(sub_page, axis=0), edge).astype(float)
    pos = np.arange(x0, x1)
    beyond = np.maximum(cols[0] - pos, pos - cols[-1]).clip(0) / SIDE_REACH
    t = np.clip(beyond, 0, 1)
    t = t * t * (3 - 2 * t)
    edge = edge * (1 - t) + own * t
    rows = np.arange(h)[:, None]
    if side == Side.BOTTOM:
        out[:, x0:x1] = (rows >= np.minimum(top, own)[None, :] - 1) & (rows <= edge[None, :])
    else:
        out[:, x0:x1] = (rows >= edge[None, :]) & (rows <= np.maximum(bottom, own)[None, :] + 1)
    return out & page_cols[None, :]


def line_candidates(
    lines: np.ndarray, ink: np.ndarray, x_height: float
) -> list[tuple[tuple[slice, slice], np.ndarray]]:
    """Сгустки-строки; слипшиеся по вертикали (выше :data:`LINE_HEIGHT`) режутся заново более тонким смыканием.

    У корешка и загиба лист изогнут, строки наклонены, и смыкание вдоль строки склеивает соседние строки в один
    высокий сгусток. Внутри его рамки краска смыкается окном :data:`SPLIT_CLOSE` — промежуток между строками остаётся.

    Args:
        lines: Маска строк (bool).
        ink: Краска (bool).
        x_height: Типичная высота буквы страницы, px: толще :data:`MAX_THICK_REL` её — слипшиеся строки.

    Returns:
        Пары «рамка, маска в рамке».
    """
    out: list[tuple[tuple[slice, slice], np.ndarray]] = []
    labels, _ = ndimage.label(lines)
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        comp = labels[sl] == i
        if line_thickness(comp) <= min(LINE_HEIGHT[1], MAX_THICK_REL * x_height):
            out.append((sl, comp))
            continue
        fine = cv2.morphologyEx((ink[sl] & comp).astype(np.uint8), cv2.MORPH_CLOSE, np.ones(SPLIT_CLOSE, np.uint8)) > 0
        sub_labels, _ = ndimage.label(fine & comp)
        for j, sub in enumerate(ndimage.find_objects(sub_labels), start=1):
            box = (slice(sl[0].start + sub[0].start, sl[0].start + sub[0].stop),
                   slice(sl[1].start + sub[1].start, sl[1].start + sub[1].stop))  # fmt: skip
            out.append((box, sub_labels[sub] == j))
    return out


def edge_row(texts: list[tuple[tuple[slice, slice], np.ndarray]], side: Side) -> list[int]:
    """Крайний ряд страницы: строки, ниже (для верха — выше) которых в их колонке нет других строк.

    Args:
        texts: Строки текста страницы — пары «рамка, маска».
        side: ``BOTTOM`` или ``TOP``.

    Returns:
        Номера строк крайнего ряда в ``texts``.
    """
    rows = []
    for sl, comp in texts:
        ys = np.nonzero(comp)[0]
        rows.append((sl[0].start + float(np.median(ys)), sl[1].start, sl[1].stop, sl[0].stop - sl[0].start))
    out = []
    for i, (cy, x0, x1, hgt) in enumerate(rows):
        beyond = False
        for j, (cy2, a0, a1, _) in enumerate(rows):
            overlap = min(x1, a1) - max(x0, a0)
            if j == i or overlap <= ROW_OVERLAP * min(x1 - x0, a1 - a0):
                continue
            # Другая строка той же колонки дальше к краю — эта не крайняя.
            if (cy2 > cy + 0.5 * hgt) if side == Side.BOTTOM else (cy2 < cy - 0.5 * hgt):
                beyond = True
                break
        if not beyond:
            out.append(i)
    return out


def line_registered(sl: tuple[slice, slice], ncc: np.ndarray, visible_ink: np.ndarray) -> bool:
    """Совмещена ли строка: корреляция донора и (1) в видимой краске около строки (сама строка и строки над/под ней).

    Плохо совмещённая страница (мало точек, расфокус, чужая пара) даёт строку, поставленную не туда, — её не
    дорисовываем: лучше срез, чем чужой текст.

    Args:
        sl: Рамка строки.
        ncc: Карта корреляции донора и (1).
        visible_ink: Видимая бумага (1) с краской (bool).

    Returns:
        True — совмещена.
    """
    # Окно — столбцы самой строки (вбок у корешка лежит загнутый край соседней страницы, он с донором не совпадает),
    # по высоте — несколько строк над ней: у короткой строки (колонтитул) так набирается видимая краска.
    window = (slice(max(0, sl[0].start - CHECK_REACH), sl[0].stop + CHECK_REACH // 2), sl[1])
    region = visible_ink[window]
    if region.sum() < CHECK_MIN_PX:
        return False
    return float(np.median(ncc[window][region])) >= LINE_NCC_MIN


def same_row(texts: list[tuple[tuple[slice, slice], np.ndarray]], i: int, ids: list[int]) -> bool:
    """Лежит ли строка ``i`` на одном уровне со срезанными строками ``ids``: её пиксели лежат на оси срезанных
    строк (парабола по их пикселям), а не на строку выше или ниже.

    Args:
        texts: Строки страницы — пары «рамка, маска».
        i: Проверяемая строка.
        ids: Срезанные строки того же края.

    Returns:
        True — тот же ряд.
    """
    # Ось срезанных строк: парабола по всем их пикселям (строки у загиба изогнуты); толщина — медианная.
    xs_all, ys_all, thick = [], [], []
    for j in ids:
        sl, comp = texts[j]
        ys, xs = np.nonzero(comp)
        xs_all.append(sl[1].start + xs)
        ys_all.append(sl[0].start + ys)
        thick.append(line_thickness(comp))
    xs_all, ys_all = np.concatenate(xs_all), np.concatenate(ys_all)
    deg = 2 if np.ptp(xs_all) > 200 else 1
    poly = np.polyfit(xs_all, ys_all, deg)
    sl, comp = texts[i]
    ys, xs = np.nonzero(comp)
    xs, ys = sl[1].start + xs, sl[0].start + ys
    # Строка того же ряда: медианное отклонение её пикселей от оси срезанных строк меньше 0.7 толщины строки.
    return float(np.median(np.abs(ys - np.polyval(poly, xs)))) < 0.7 * float(np.median(thick))


def find_cut_lines(
    warped: np.ndarray,
    paper1: np.ndarray,
    page: np.ndarray,
    other: np.ndarray | None,
    reference: float,
    gray_a: np.ndarray,
    sides: frozenset = frozenset(Side),
) -> CutResult:
    """Срезанные строки страницы и альфа вклейки донора.

    Args:
        warped: Донор BGR после совмещения (координаты (1), масштаб совмещения).
        paper1: Видимая бумага (1) (маска v2).
        page: Маска страницы (1).
        other: Маска соседней страницы или None.
        reference: Медиана серого видимой бумаги (1).
        gray_a: Серое (1) на масштабе совмещения: проверка совмещения строки по видимой краске около неё.
        sides: Какие края страницы дорисовывать (по умолчанию все).

    Returns:
        Срезанные строки, их маска, альфа и «чего не видно».
    """
    h, w = page.shape
    zone = (
        cv2.dilate(page.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PAGE_REACH + 1,) * 2)) > 0
    )
    if other is not None and other.any():
        # Своя половина разворота: ближе к своей странице, чем к соседней (граница — посередине, по корешку).
        own = cv2.distanceTransform((~page).astype(np.uint8), cv2.DIST_L2, 5) < cv2.distanceTransform(
            (~other).astype(np.uint8), cv2.DIST_L2, 5
        )
        zone &= own & ~other
    # Совмещение по месту: корреляция донора и (1) в окне (для проверки каждой срезанной строки).
    ncc, var = local_ncc(gray_a, cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    # Видимая краска основы (буквы), а не любой перепад: у кромки листа перепад «бумага — коврик» и тень дают
    # высокую дисперсию, но корреляция там ничего не говорит о совмещении строк.
    base_ink, _ = ink_lines(gray_a)
    base_ink = cv2.dilate(base_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    visible_ink = (cv2.erode(paper1.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & base_ink & (var > VAR_MIN)
    spine = spine_zone(page, other)
    paper2 = donor_paper(warped, reference)
    # За кромкой — вне внешнего контура бумаги (1): дырки внутри маски бумаги v2 (заголовки, линейки) не в счёт.
    missing = paper2 & ~ndimage.binary_fill_holes(paper1) & zone & ~spine
    ink, lines = ink_lines(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    # Строки — только в зоне своей страницы: иначе нижние строки двух страниц сливаются через корешок в один сгусток.
    lines &= paper2 & zone
    x_height = glyph_height(ink, page & paper1)
    # Строки текста своей страницы (донор) и среди них — срезанные.
    texts: list[tuple[tuple[slice, slice], np.ndarray]] = []
    found: list[CutLine] = []
    cut_ids: dict[Side, list[int]] = {side: [] for side in Side}
    for sl, comp in line_candidates(lines, ink, x_height):
        area = int(comp.sum())
        if area < LINE_MIN_AREA or (comp & zone[sl]).sum() / area < OWN_PAGE_SHARE:
            continue
        if not is_text_line(comp, ink[sl], x_height):
            continue
        texts.append((sl, comp))
        part = comp & missing[sl]
        n_out = int(part.sum())
        if n_out < CUT_MIN_PX or n_out / area < CUT_MIN_SHARE:
            continue
        if not line_registered(sl, ncc, visible_ink):
            continue
        full_part = np.zeros((h, w), bool)
        full_part[sl] = part
        side = side_of(full_part, page)
        if side not in sides:
            continue
        found.append(CutLine(side, n_out / area, (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)))
        cut_ids[side].append(len(texts) - 1)
    cut = np.zeros((h, w), bool)
    bands: list[np.ndarray] = []
    pieces: list[np.ndarray] = []
    for side, ids in cut_ids.items():
        if not ids:
            continue
        if side == Side.OUTER:
            # Наружный край: по строке — своя полоса.
            groups = [[i] for i in ids]
        else:
            # Низ или верх: если повреждена крайняя строка, перерисовывается весь крайний ряд страницы во всех
            # колонках (и неповреждённые слова) — кромка под ним одна и ровная, а не волнистая. В ряд идут только
            # строки на уровне срезанных: конец предыдущей строки над короткой последней тоже «крайний» в своей
            # колонке, но это другой ряд.
            row = [i for i in edge_row(texts, side) if same_row(texts, i, ids)]
            groups = [sorted(set(ids) | set(row))]
        for group in groups:
            union = np.zeros((h, w), bool)
            for i in group:
                sl, comp = texts[i]
                union[sl] |= comp
                piece = np.zeros((h, w), bool)
                piece[sl] = comp
                pieces.append(piece)
            cut |= union
            bands.append(line_band(union, side, EXTEND_REACH, page))
    alpha = np.zeros((h, w), np.float32)
    extension = np.zeros((h, w), bool)
    text = np.zeros((h, w), bool)
    if found:
        filled = ndimage.binary_fill_holes(paper1)
        # Вклейка не заходит только на сам сгиб (узкая полоса): крайний ряд перерисовывается почти до корешка.
        fold = spine_zone(page, other, FOLD_FRAC)
        seam = cv2.dilate(cut.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, SEAM_REACH[::-1])) > 0
        # Полоса строки: от строки до её гладкой огибающей плюс поле (без раздутия вбок: вбок лежат торец и кромка
        # обложки донора); в бумаге (1) — ещё полстроки вокруг для шва.
        band = np.logical_or.reduce(bands)
        # Полосы строк соседних колонок смыкаются по горизонтали: иначе между колонками на кромке остаётся выемка.
        band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 2 * COLUMN_GAP + 1), np.uint8)) > 0
        band |= seam & filled
        band &= zone & ~fold
        # Новый лист = бумага (1) ∪ полоса; стыки новой кромки с настоящей сглаживаются закрытием рядом с полосой —
        # кромка идёт одной плавной линией.
        sheet = filled | band
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_SMOOTH + 1,) * 2)
        near = cv2.dilate(band.astype(np.uint8), k) > 0
        sheet |= (cv2.morphologyEx(sheet.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0) & near & zone & ~fold
        extension = sheet & ~filled
        # Внутри бумаги (1) — мягкий шов до середины межстрочного промежутка; продление листа — целиком из донора;
        # снаружи нового листа — коврик (1) как был: кромка чёткая.
        # Краска донора — только в самих срезанных строках с небольшим полем: вбок за концом строки (у корешка)
        # совмещение неточно, и сдвинутые буквы донора дали бы призрак слова; видимый текст (1) рядом не трогается.
        text = cv2.dilate(cut.astype(np.uint8), np.ones((2 * TEXT_PAD[0] + 1, 2 * TEXT_PAD[1] + 1), np.uint8)) > 0
        # У корешка текст донора повреждён — там краску не берём (строка (1) у корешка остаётся своей).
        text &= zone & ~spine
        inner = text & filled
        alpha = cv2.GaussianBlur((inner | extension).astype(np.float32), (0, 0), ALPHA_BLUR)
        alpha[extension | (cut & text)] = 1.0
        alpha[~sheet] = 0.0
        # На видимой бумаге (1) — только своя страница: у корешка соседняя страница не трогается никогда.
        alpha[paper1 & ~page] = 0.0
        alpha[~zone] = 0.0
    return CutResult(found, cut, alpha, missing, extension, paper2, text, pieces)
