"""Продление нижней кромки листа вниз с сохранением её формы: кромка по маске бумаги (с правкой углов и корешка), торец стопки за краем страницы, маски сдвига, новая бумага с тоном и зерном своей страницы, сборка."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from research.cut_lines import synth as S

# Окно уточнения кромки на полном кадре вокруг границы маски бумаги (½ кадра → ×2), px.
EDGE_REFINE = 12
# Окна сравнения яркости при уточнении кромки: над строкой и под ней (глубже толщины линейки рамки), px.
EDGE_DARK_UP = 6
EDGE_DARK_DEPTH = 14
# Поиск кромки вниз от грубой (нижний лист стопки выглядывает из-под страницы), px; берётся самый нижний спад
# не слабее этой доли сильнейшего; спад слабее этого (уровней) — кромка не уточняется.
EDGE_SEARCH_DOWN = 120
EDGE_LOWER_SHARE = 0.5
EDGE_MIN_DROP = 25.0
# Сглаживание кромки по столбцам: медиана и гаусс, px.
EDGE_MEDIAN = 61
EDGE_SMOOTH = 15
# Столбец страницы считается «телом» страницы, если бумаги в нём не меньше этой доли медианной высоты столбца:
# узкие выступы у углов и под пальцами не задают кромку.
COLUMN_MIN_SHARE = 0.3
# Хвостовые столбцы тела (½ кадра), где низ маски прыгает больше этого за столбец (px полного кадра), отрезаются:
# у корешка маска одной страницы заползает узкой полоской на соседнюю, у наружного угла — задранный уголок.
EDGE_JUMP = 40
EDGE_TAIL = 40
# Правка концов кромки по робастной прямой через низы последних столбцов (px полного кадра): где низ маски выше
# прямой больше чем на EDGE_LIFT (задранный уголок, выемка у корешка), берётся прямая.
EDGE_LINE_SPAN = 300
EDGE_LIFT = 40
# Полоса кромки, которая едет вниз как есть: наружу от кромки (тень, торец стопки), px; и полоса внутрь листа,
# которая заливается бумагой заново (светлый ободок самой кромки и её антиалиас), px.
EDGE_BELOW = 80
EDGE_ABOVE = 6
# За наружным углом страницы полоса кромки берётся и по соседним столбцам (торец стопки под углом едет вниз вместе
# с кромкой, а не ступенькой), px; низ в них — по прямой через последние CORNER_FIT столбцов.
CORNER_REACH = 80
CORNER_FIT = 300
# Перо по контуру сдвинутой полосы, px.
EDGE_FEATHER = 12
# Смыкание маски бумаги по горизонтали у низа: щель корешка между страницами тоже продлевается, px.
SPINE_CLOSE = 121
# Глубина над кромкой, где берётся тон бумаги по столбцам, px; сглаживание тона вдоль x, px.
TONE_DEPTH = 150
TONE_SMOOTH = 25
# Пробег столбцов без чистой бумаги длиннее этого (логотип у кромки) не интерполируется, px.
TONE_GAP_MAX = 12
# Пиксель темнее медианы бумаги страницы на столько уровней — не чистая бумага (полутон, тень), в тон не идёт.
TONE_DARK = 25
# Низ маски ближе этого к нижней рамке кадра — лист срезан рамкой, а не кромкой, px (грубая маска ½ кадра и
# маска на полном кадре).
EDGE_FRAME = 30
FRAME_MARGIN = 6
# Одиночные переключения признака «у рамки» уже этого (столбцов) убираются.
FRAME_RUN = 41
# Достройка кромки у рамки: минимум соседних столбцов с кромкой для прямой; сколько ближайших столбцов дают профиль
# полосы кромки.
FRAME_FIT_MIN = 60
FRAME_PROFILE_COLS = 100
FRAME_ANCHOR = 15
# Размытие вдоль x дорощенных строк холста, px.
PAD_BLUR = 3.0
# Потолок зерна новой бумаги (σ по каналам), уровни.
GRAIN_MAX = 6.0
# Потолок продления вертикального хода яркости бумаги на новой полосе, уровни.
TONE_TREND_MAX = 10.0
# Отступ от краски при замере тона и зерна, px.
INK_PAD = 5
# Торец стопки за наружным краем страницы: не шире этого, px; его столбец светлый (серое выше порога) в полосе
# STACK_DEPTH над кромкой не меньше STACK_SHARE; окно поиска его кромки вокруг продолжения кромки страницы, px.
STACK_MAX = 250
STACK_BRIGHT = 80
STACK_DEPTH = 200
STACK_SHARE = 0.5
STACK_SEARCH = 60
STACK_MISSES = 24


@dataclass(frozen=True)
class BottomEdge:
    """Нижняя кромка страницы на полном кадре.

    Поля:
        x0: Левый столбец тела страницы.
        x1: Правый столбец (не включая).
        y: Строка кромки по столбцам ``[x0, x1)`` (float): первая строка под бумагой.
        frame: По столбцам ``[x0, x1)``: лист уходит за нижнюю рамку кадра (кромки нет, ``y = h − 1``).
    """

    x0: int
    x1: int
    y: np.ndarray
    frame: np.ndarray

    def line(self, side: int, span: int = EDGE_LINE_SPAN) -> np.ndarray:
        """Робастная прямая (коэффициенты ``np.polyval``) через кромку у края ``side`` (−1 левый, +1 правый)."""
        n = min(span, self.x1 - self.x0)
        xs = np.arange(self.x0, self.x0 + n) if side < 0 else np.arange(self.x1 - n, self.x1)
        ys = self.y[:n] if side < 0 else self.y[-n:]
        return robust_line(xs.astype(float), ys)


@dataclass(frozen=True)
class Extension:
    """Итог продления.

    Поля:
        image: Кадр после продления (может быть выше исходного: холст доращивается снизу).
        box: Окно изменений ``(x0, y0, x1, y1)`` в координатах нового кадра.
        pad: На сколько строк холст дорос снизу.
        paper: Маска новой бумаги (щель между старой и новой кромкой) в координатах нового кадра.
        band: Маска сдвинутой полосы кромки (тень, торец, клин корешка) в координатах нового кадра.
    """

    image: np.ndarray
    box: tuple[int, int, int, int]
    pad: int
    paper: np.ndarray
    band: np.ndarray


def robust_line(xs: np.ndarray, ys: np.ndarray, tol: float = EDGE_LIFT) -> np.ndarray:
    """Прямая через точки с отсевом далёких (два прохода).

    Args:
        xs: Абсциссы.
        ys: Ординаты.
        tol: Порог остатка для отсева, px.

    Returns:
        Коэффициенты ``np.polyval`` (наклон, сдвиг).
    """
    if len(xs) < 2:
        return np.array([0.0, float(ys[0]) if len(ys) else 0.0])
    # Старт — горизонталь на медиане, а не МНК по всем точкам: крутой хвост у угла или корешка наклонил бы
    # первую прямую, и отсев оставил бы точки хвоста вместо тела кромки.
    poly = np.array([0.0, float(np.median(ys))])
    for _ in range(3):
        keep = np.abs(np.polyval(poly, xs) - ys) <= tol
        if keep.sum() < 2:
            break
        poly = np.polyfit(xs[keep], ys[keep], 1)
    return poly


def trim_jumps(rows: np.ndarray, jump: float, tail: int) -> tuple[int, int]:
    """Хвостовые столбцы, где низ прыгает больше ``jump`` за столбец (в пределах ``tail`` столбцов от края).

    Args:
        rows: Низ по столбцам.
        jump: Порог прыжка, px.
        tail: Сколько крайних столбцов проверять.

    Returns:
        ``(u0, u1)`` — границы после отрезания.
    """
    n = len(rows)
    diff = np.abs(np.diff(rows))
    u0, u1 = 0, n
    big = np.nonzero(diff[: min(tail, n - 1)] > jump)[0]
    if len(big):
        u0 = int(big.max()) + 1
    big = np.nonzero(diff[max(0, n - 1 - tail) :] > jump)[0]
    if len(big):
        u1 = max(0, n - 1 - tail) + int(big.min()) + 1
    return (u0, u1) if u1 - u0 >= 20 else (0, n)


def straighten_ends(xs: np.ndarray, rows: np.ndarray, span: int, lift: float) -> np.ndarray:
    """Концы кромки по робастной прямой: где низ выше прямой больше чем на ``lift``, берётся прямая.

    Args:
        xs: Столбцы.
        rows: Низ по столбцам (первая строка под бумагой).
        span: Длина отрезка прямой у края, px.
        lift: Допуск подъёма, px.

    Returns:
        Исправленный низ.
    """
    out = rows.copy()
    n = min(span, len(xs))
    if n < 20:
        return out
    for side, sl in ((-1, slice(0, n)), (1, slice(len(xs) - n, len(xs)))):
        poly = robust_line(xs[sl].astype(float), rows[sl], lift)
        line = np.polyval(poly, xs[sl])
        seg = out[sl]
        bad = seg < line - lift
        # Правится только непрерывный пробег нарушений, упирающийся в край (задранный уголок, корешок);
        # нарушение посреди окна — не угол.
        run = np.zeros_like(bad)
        idx = np.arange(len(bad)) if side < 0 else np.arange(len(bad))[::-1]
        for i in idx:
            if not bad[i]:
                break
            run[i] = True
        seg[run] = line[run]
        out[sl] = seg
    return out


def page_bottom_edge(page_half: np.ndarray, gray: np.ndarray, bg_half: np.ndarray | None = None) -> BottomEdge | None:
    """Нижняя кромка страницы: низ маски бумаги (½ кадра) без хвостов с прыжками и с правкой концов по прямой,
    уточнённый на полном кадре по спаду яркости вниз (бумага → тень кромки), сглаженный по столбцам. Столбцы, где
    лист уходит за нижнюю рамку кадра, — по маске фона: низ маски у рамки и под ним нет фона.

    Args:
        page_half: Маска страницы на уменьшенном кадре (bool).
        gray: Серое полного кадра uint8.
        bg_half: Маска фона (scan_cropping v2 ``bg``) на уменьшенном кадре или None (тогда — только по
            расстоянию до рамки).

    Returns:
        Кромка или None, если у страницы нет тела (мало столбцов).
    """
    h, w = gray.shape
    sy, sx = h / page_half.shape[0], w / page_half.shape[1]
    heights = page_half.sum(axis=0)
    cols = np.nonzero(heights > 0)[0]
    if len(cols) < 20:
        return None
    body = cols[heights[cols] >= COLUMN_MIN_SHARE * np.median(heights[cols])]
    if len(body) < 20:
        return None
    u0, u1 = int(body.min()), int(body.max()) + 1
    # Низ маски по столбцам тела (последняя строка бумаги + 1), в полном кадре.
    rows = page_half.shape[0] - 1 - np.argmax(page_half[::-1, u0:u1], axis=0)
    rows = np.where(page_half[:, u0:u1].any(axis=0), rows + 1, np.nan).astype(float)
    idx = np.arange(u1 - u0)
    ok = ~np.isnan(rows)
    rows = np.interp(idx, idx[ok], rows[ok]) * sy
    # Хвосты с прыжками (полоска маски на соседней странице у корешка, задранный уголок) отрезаются.
    t0, t1 = trim_jumps(rows, EDGE_JUMP, EDGE_TAIL)
    u0, u1, rows, idx = u0 + t0, u0 + t1, rows[t0:t1], idx[t0:t1] - t0
    x0, x1 = int(u0 * sx), min(w, int(u1 * sx))
    xs = np.arange(x0, x1)
    coarse = np.interp((xs - x0) / sx, idx, rows)
    coarse = straighten_ends(xs, coarse, EDGE_LINE_SPAN, EDGE_LIFT)
    refined = refine_edge(gray, xs, coarse)
    smooth = ndimage.gaussian_filter1d(ndimage.median_filter(refined, EDGE_MEDIAN, mode="nearest"), EDGE_SMOOTH,
                                       mode="nearest")  # fmt: skip
    # Столбцы, где лист уходит за нижнюю рамку кадра: кромки нет (уточнение поймало бы верх последней строки
    # букв), кромка — сама рамка. Признак — низ маски у рамки и (если есть маска фона) под низом маски нет фона:
    # кромка в 2–30 px над рамкой — настоящая, под ней фон.
    at_frame = coarse >= h - 1 - EDGE_FRAME
    if bg_half is not None:
        bottoms_half = np.round(coarse / sy).astype(int)
        no_bg = np.ones(len(xs), bool)
        for i, x in enumerate(xs):
            u = min(bg_half.shape[1] - 1, int(x / sx))
            pb = min(bg_half.shape[0] - 1, bottoms_half[i])
            no_bg[i] = not bg_half[pb:, u].any()
        at_frame &= no_bg
    at_frame = ndimage.binary_closing(ndimage.binary_opening(at_frame, np.ones(FRAME_RUN)), np.ones(FRAME_RUN))
    smooth[at_frame] = h - 1
    return BottomEdge(x0, x1, np.clip(smooth, 1, h - 1), at_frame)


def refine_edge(gray: np.ndarray, xs: np.ndarray, coarse: np.ndarray, window: int = EDGE_REFINE) -> np.ndarray:
    """Уточнение кромки по столбцам: строка, под которой яркость падает и остаётся низкой (бумага → тень), в окне
    вокруг грубой кромки. Сравниваются средние над строкой и в полосе под ней глубиной :data:`EDGE_DARK_DEPTH`:
    тонкая линейка рамки объявления даёт резкий, но короткий спад и кромкой не считается.

    Args:
        gray: Серое полного кадра uint8.
        xs: Столбцы.
        coarse: Грубая кромка по столбцам.
        window: Полуширина окна, px.

    Returns:
        Уточнённая кромка (float): первая строка под бумагой.
    """
    h = gray.shape[0]
    g = gray.astype(np.float32)
    up, lo = EDGE_DARK_UP, EDGE_DARK_DEPTH
    out = coarse.copy()
    for i, (x, y) in enumerate(zip(xs, coarse)):
        a, b = int(max(up, y - window)), int(min(h - lo, y + EDGE_SEARCH_DOWN))
        if b - a < 2:
            continue
        col = g[:, x]
        ks = np.arange(a, b + 1)
        above = np.array([col[k - up : k].mean() for k in ks])
        below = np.array([col[k + 1 : k + 1 + lo].mean() for k in ks])
        score = above - below
        best = float(score.max())
        if best < EDGE_MIN_DROP:
            continue
        # Самый нижний сильный спад: под листом может выглядывать следующий лист стопки — кромка стопки ниже, и
        # тогда продлевается стопка целиком (иначе полоса кромки дублирует линейки рамки и кромку нижнего листа).
        strong = np.nonzero(score >= EDGE_LOWER_SHARE * best)[0]
        out[i] = ks[strong[-1]]
    return out


def full_page_mask(page_half: np.ndarray, edge: BottomEdge, shape: tuple[int, int]) -> np.ndarray:
    """Маска страницы на полном кадре с низом по уточнённой кромке (столбцы за пределами тела кромки — как в маске).

    Args:
        page_half: Маска страницы на уменьшенном кадре (bool).
        edge: Кромка.
        shape: ``(H, W)`` полного кадра.

    Returns:
        Маска (bool) полного кадра.
    """
    h, w = shape
    mask = cv2.resize(page_half.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    rows = np.arange(h)[:, None]
    ys = np.round(edge.y).astype(int)[None, :]
    sub = mask[:, edge.x0 : edge.x1]
    # Ниже кромки — не бумага; между низом маски и кромкой (если маска короче) — бумага.
    sub &= rows < ys
    top = h - 1 - np.argmax(sub[::-1], axis=0)
    has = sub.any(axis=0)
    fill = (rows > top[None, :]) & (rows < ys) & has[None, :]
    mask[:, edge.x0 : edge.x1] = sub | fill
    # Столбцы вне тела кромки (отрезанные хвосты: полоска на соседней странице, задранный уголок) — не бумага
    # страницы: иначе они задали бы свою «кромку» и перекрасили настоящие пиксели.
    mask[:, : edge.x0] = False
    mask[:, edge.x1 :] = False
    return mask


def stack_columns(book_half: np.ndarray, gray: np.ndarray, edge: BottomEdge, side: int) -> np.ndarray:
    """Торец стопки за наружным краем страницы: столбцы между кромкой страницы и наружным краем книги, светлые в
    полосе над продолжением кромки; их низ — по спаду яркости вокруг прямой кромки.

    Args:
        book_half: Маска книги на уменьшенном кадре (bool).
        gray: Серое полного кадра uint8.
        edge: Кромка страницы.
        side: Наружная сторона страницы: −1 — левая, +1 — правая.

    Returns:
        Маска (bool) полного кадра: столбцы торца от ``STACK_DEPTH`` над кромкой до кромки.
    """
    h, w = gray.shape
    out = np.zeros((h, w), bool)
    sx = w / book_half.shape[1]
    book_cols = np.nonzero(book_half.any(axis=0))[0]
    if len(book_cols) == 0:
        return out
    limit = int(book_cols.min() * sx) if side < 0 else int((book_cols.max() + 1) * sx)
    xs = (
        np.arange(max(0, max(limit, edge.x0 - STACK_MAX)), edge.x0)
        if side < 0
        else np.arange(edge.x1, min(w, min(limit, edge.x1 + STACK_MAX)))
    )
    if len(xs) == 0:
        return out
    poly = edge.line(side)
    order = xs[::-1] if side < 0 else xs
    g = gray.astype(np.float32)
    # Кромка торца идёт своим наклоном (стопка под страницей скошена): столбцы обходятся от страницы наружу, окно
    # поиска — вокруг кромки предыдущего столбца, первый — вокруг продолжения кромки страницы.
    y_prev = float(np.polyval(poly, order[0]))
    misses = 0
    for x in order:
        y_line = y_prev
        a, b = int(max(STACK_DEPTH, y_line - STACK_SEARCH)), int(min(h - EDGE_DARK_DEPTH - 1, y_line + STACK_SEARCH))
        if b - a < 4:
            break
        band = g[a - STACK_DEPTH : a, x]
        col = g[:, x]
        ks = np.arange(a, b + 1)
        score = np.array([col[k - EDGE_DARK_UP : k].mean() - col[k + 1 : k + 1 + EDGE_DARK_DEPTH].mean() for k in ks])
        # Столбец без светлой стопки над кромкой или без спада на кромке — пропуск; подряд STACK_MISSES пропусков —
        # торец кончился (тёмная щель между страницей и стопкой допускается).
        if (band > STACK_BRIGHT).mean() < STACK_SHARE or score.max() < EDGE_MIN_DROP:
            misses += 1
            if misses > STACK_MISSES:
                break
            continue
        misses = 0
        y_edge = int(ks[int(np.argmax(score))])
        y_prev = float(y_edge)
        bright = g[:y_edge, x] > STACK_BRIGHT
        top = max(0, y_edge - STACK_DEPTH)
        dark = np.nonzero(~bright[top:y_edge])[0]
        if len(dark):
            top = top + int(dark.max()) + 1
        out[top:y_edge, x] = True
    return out


def shift_down(mask: np.ndarray, d: int) -> np.ndarray:
    """Маска, сдвинутая на ``d`` строк вниз (сверху — нули)."""
    out = np.zeros_like(mask)
    if d < mask.shape[0]:
        out[d:] = mask[: mask.shape[0] - d]
    return out


def column_tone(img: np.ndarray, good: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Тон бумаги по столбцам и зерно: медиана «хороших» пикселей столбца, интерполяция пустых столбцов,
    сглаживание вдоль x; зерно — σ отклонений хороших пикселей от тона.

    Args:
        img: Окно BGR.
        good: Пиксели чистой бумаги окна (bool).

    Returns:
        ``(тон (w, 3) float32, σ зерна (3,) float32)``.
    """
    h, w = good.shape
    tone = np.full((w, 3), np.nan, np.float32)
    counts = good.sum(axis=0)
    for u in np.nonzero(counts >= 6)[0]:
        tone[u] = np.median(img[good[:, u], u], axis=0)
    valid = ~np.isnan(tone[:, 0])
    if valid.sum() < 2:
        level = np.median(img[good], axis=0) if good.any() else np.full(3, 200.0)
        tone[:] = level
        valid[:] = True
    idx = np.arange(w)
    # Пустые столбцы (краска до самой кромки — логотип, таблица) интерполируются по соседям только если пробег
    # короче TONE_GAP_MAX; длинный пробег получает тон ближайшего чистого столбца с каждой стороны (а не смесь
    # через столбцы с краской — иначе под логотипом тянется серый шлейф).
    tone = np.stack([np.interp(idx, idx[valid], tone[valid, c]) for c in range(3)], axis=1).astype(np.float32)
    labels, count = ndimage.label(~valid)
    for i in range(1, count + 1):
        run = np.nonzero(labels == i)[0]
        if len(run) > TONE_GAP_MAX:
            l_idx = run[0] - 1
            r_idx = run[-1] + 1
            if l_idx >= 0 and r_idx < w:
                mid = (run[0] + run[-1]) // 2
                tone[run[0] : mid + 1] = tone[l_idx]
                tone[mid + 1 : run[-1] + 1] = tone[r_idx]
    tone = ndimage.gaussian_filter1d(tone, TONE_SMOOTH, axis=0, mode="nearest")
    resid = img[good].astype(np.float32) - tone[np.nonzero(good)[1]]
    grain = np.clip(resid.std(axis=0), 0, GRAIN_MAX).astype(np.float32) if good.sum() > 50 else np.zeros(3, np.float32)
    return tone, grain


def frame_extension(
    roi: np.ndarray,
    at_frame: np.ndarray,
    has: np.ndarray,
    bot_cols: np.ndarray,
    bot_m0: np.ndarray,
    d: int,
    below: int,
    real_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Достройка кромки в столбцах, где лист уходит за нижнюю рамку кадра: по каждому пробегу таких столбцов —
    прямая через ближайшие :data:`CORNER_FIT` столбцов с настоящей кромкой (сторона с бо́льшим числом столбцов),
    до неё (плюс ``d``) бумага, под ней полоса кромки шириной ``below`` — медианный профиль полосы тех же соседних
    столбцов, ниже — последняя строка профиля.

    Args:
        roi: Окно BGR.
        at_frame: По столбцам окна: лист у рамки.
        has: По столбцам: есть маска.
        bot_cols: Низ маски по столбцам (float).
        bot_m0: Низ исходной маски по столбцам (int).
        d: Сдвиг кромки, px.
        below: Ширина полосы кромки, px.
        real_rows: Сколько первых строк окна — настоящие строки кадра (ниже — дорощенный холст).

    Returns:
        ``(маска заливки бумагой, синтетическая картинка float32, её маска)``.
    """
    hh, ww = roi.shape[:2]
    rows = np.arange(hh)[:, None]
    fill = np.zeros((hh, ww), bool)
    synth = np.zeros((hh, ww, 3), np.float32)
    synth_mask = np.zeros((hh, ww), bool)
    real = has & ~at_frame
    labels, count = ndimage.label(at_frame)
    us = np.arange(ww, dtype=float)
    for k in range(1, count + 1):
        idx = np.nonzero(labels == k)[0]
        a, b = int(idx[0]), int(idx[-1]) + 1
        left = np.nonzero(real[:a])[0][-CORNER_FIT:].astype(int)
        right = (b + np.nonzero(real[b:])[0][:CORNER_FIT]).astype(int)
        # Кромка в пробеге: якоря — низы ближайших соседей с каждой стороны (медиана FRAME_ANCHOR столбцов);
        # с двумя якорями — прямая между ними (без шва на стыках), с одним — прямая через его сторону,
        # сдвинутая к якорю.
        # Якорь годен, если рядом длинный пробег столбцов с настоящей кромкой над рамкой (узкая полоска у самого
        # угла — не кромка) и кромка в нём не на самой рамке.
        anchors = []
        if len(left) >= FRAME_FIT_MIN and left[-1] - left[0] >= FRAME_FIT_MIN - 1:
            ya = float(np.median(bot_cols[left[-FRAME_ANCHOR:]]))
            if ya < real_rows - 1 - FRAME_MARGIN:
                anchors.append((float(left[-1]), ya))
        if len(right) >= FRAME_FIT_MIN and right[-1] - right[0] >= FRAME_FIT_MIN - 1:
            yb = float(np.median(bot_cols[right[:FRAME_ANCHOR]]))
            if yb < real_rows - 1 - FRAME_MARGIN:
                anchors.append((float(right[0]), yb))
        if not anchors:
            # Соседей с кромкой нет — под листом только бумага (кромка неизвестна).
            fill[:, a:b] = (rows > bot_m0[None, a:b]) & (rows > bot_cols[None, a:b] - 1)
            continue
        if len(anchors) == 2:
            (xa, ya), (xb, yb) = anchors
            y_v = ya + (yb - ya) * (us[a:b] - xa) / max(xb - xa, 1.0)
        else:
            xa, ya = anchors[0]
            sel = left if xa < a else right
            poly = robust_line(us[sel], bot_cols[sel])
            y_v = np.polyval(poly, us[a:b]) + (ya - float(np.polyval(poly, xa)))
        y_v = np.clip(y_v, bot_m0[a:b] + 1, hh - 1)
        y_new = y_v + d
        fill[:, a:b] = (rows > bot_m0[None, a:b]) & (rows <= y_new[None, :])
        # Профиль полосы кромки соседей: строки под их низом, медиана по столбцам (сторона с бо́льшим числом).
        sides = [left if xa < a else right for xa, _ in anchors]
        sel = max(sides, key=len)
        near = (sel[-FRAME_PROFILE_COLS:] if sel is left else sel[:FRAME_PROFILE_COLS]).astype(int)
        prof = np.full((below, 3), np.nan, np.float32)
        for j in range(below):
            ys = (bot_cols[near] + 1 + j).astype(int)
            ok = ys < real_rows
            if ok.sum() >= 3:
                prof[j] = np.median(roi[ys[ok], near[ok]], axis=0)
        valid = ~np.isnan(prof[:, 0])
        if valid.sum() < 3:
            continue
        jj = np.arange(below)
        # Строки профиля без данных (соседи упираются в рамку) — продолжение последней известной.
        prof = np.stack([np.interp(jj, jj[valid], prof[valid, c]) for c in range(3)], axis=1)
        off = rows - y_new[None, :] - 1  # 0 — первая строка полосы
        inside = (off >= 0) & (off < below)
        beyond = off >= below
        sub = np.zeros((hh, b - a, 3), np.float32)
        oi = np.clip(off, 0, below - 1).astype(int)
        sub[:] = prof[oi]
        synth[:, a:b] = sub
        synth_mask[:, a:b] = inside | beyond
    return fill, synth, synth_mask


def extend_bottom(
    img: np.ndarray,
    mask: np.ndarray,
    d: int,
    exclude: np.ndarray | None = None,
    above: int = EDGE_ABOVE,
    below: int = EDGE_BELOW,
    feather: int = EDGE_FEATHER,
    seed: int = 0,
    frame_cols: np.ndarray | None = None,
    spine: tuple[int, int] | None = None,
    pad_color: np.ndarray | None = None,
) -> Extension:
    """Продление листа вниз на ``d`` строк: полоса кромки (тень, торец, клин корешка между страницами) едет вниз
    как есть, освободившаяся щель заливается бумагой с тоном и зерном столбца; пиксели исходной бумаги не меняются.
    Где лист уходит за нижнюю рамку кадра, кромка достраивается по прямой через соседние столбцы с настоящей
    кромкой: до неё бумага, под ней — полоса кромки, синтезированная из тех же соседних столбцов.

    Args:
        img: Кадр BGR uint8.
        mask: Маска бумаги продлеваемых страниц на полном кадре (bool), низ — по кромке.
        d: Сдвиг кромки, px.
        exclude: Пиксели, которые трогать нельзя (чужая страница, пальцы) или None.
        frame_cols: По столбцам кадра: лист уходит за нижнюю рамку (``BottomEdge.frame``) или None — тогда по
            расстоянию низа маски до рамки.
        spine: Отрезок столбцов кадра ``(x0, x1)`` зоны корешка между страницами: только там клин щели между
            кромками едет вниз; None — везде (выемка у наружного угла тоже поехала бы вниз чёрным блоком).
        pad_color: Цвет фона BGR для дорощенных строк холста вне окна продления (рука, рукав, стол под кадром
            обрезаются рамкой честно) или None — повтор последней строки везде.
        above: Полоса кромки внутрь листа, px.
        below: Полоса кромки наружу, px.
        feather: Перо по контуру сдвинутой полосы, px.
        seed: Зерно генератора шума.

    Returns:
        Итог продления.
    """
    h, w = mask.shape
    cols = np.nonzero(mask.any(axis=0))[0]
    empty = np.zeros((h, w), bool)
    if len(cols) == 0:
        return Extension(img, (0, 0, 0, 0), 0, empty, empty)
    bottom = h - 1 - np.argmax(mask[::-1], axis=0)
    frame_all = frame_cols if frame_cols is not None else bottom >= h - 1 - FRAME_MARGIN
    y_bot = bottom[cols]
    # Окно по строкам — вокруг тела кромки (процентили, а не минимум: у наружного угла низ маски уходит вверх);
    # низ окна — и под столбцами у рамки кадра: там достраивается кромка и полоса под ней.
    r0 = int(max(0, np.percentile(y_bot, 2) - d - above - TONE_DEPTH))
    r1 = int(y_bot.max() + d + below + feather + 2)
    pad = max(0, r1 - h)
    if pad:
        # Холст доращивается повтором последней строки, размытым вдоль x: голый повтор даёт вертикальные полосы
        # из букв и кожи, зеркало — перевёрнутый текст.
        img = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_REPLICATE)
        img[h:] = cv2.GaussianBlur(img[h:], (0, 0), sigmaX=PAD_BLUR, sigmaY=0.1)
        mask = np.vstack([mask, np.zeros((pad, w), bool)])
        if exclude is not None:
            exclude = np.vstack([exclude, np.zeros((pad, w), bool)])
    c0 = int(max(0, cols.min() - CORNER_REACH - feather))
    c1 = int(min(w, cols.max() + CORNER_REACH + feather + 1))
    if pad and pad_color is not None:
        img[h:, :c0] = pad_color
        img[h:, c1:] = pad_color
    roi = img[r0:r1, c0:c1]
    m0 = mask[r0:r1, c0:c1]
    # Щель корешка между страницами у низа смыкается только для замера низа и полосы кромки; сам клин щели (gap)
    # не бумага — он уедет вниз вместе с полосой кромки, а его старое место станет бумагой.
    m = cv2.morphologyEx(m0.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, SPINE_CLOSE), np.uint8)) > 0
    rows = np.arange(r1 - r0)[:, None]
    # Клин щели — только ниже собственного низа столбца (между кромкой своей страницы и кромкой соседней); дырки
    # закрытия выше (между задранным уголком и торцом) — не щель.
    has0 = m0.any(axis=0)
    bot_m0 = np.where(has0, m0.shape[0] - 1 - np.argmax(m0[::-1], axis=0), -(10**6))
    # В столбцах без маски между страницами (обрезанные хвосты масок у корешка) низ — интерполяция по соседям:
    # иначе «ниже низа» там вся колонка, и клин щели утащил бы вниз полосу текста.
    in_spine = np.zeros(m.shape[1], bool)
    if spine is not None:
        in_spine[max(0, spine[0] - c0) : max(0, spine[1] - c0)] = True
    us0 = np.arange(m.shape[1])
    bot_ref = bot_m0.astype(float)
    if has0.sum() >= 2:
        bot_ref[~has0] = np.interp(us0[~has0], us0[has0], bot_m0[has0].astype(float))
    bot_m0 = np.where(has0 | in_spine, np.round(bot_ref).astype(int), bot_m0)
    gap = m & ~m0 & (rows > bot_m0[None, :])
    if spine is not None:
        gap &= in_spine[None, :]
    # Полоса кромки — по столбцам под низом маски (не вокруг всего контура: наружный край и верх не трогаются).
    has = m.any(axis=0)
    bot_cols = np.where(has, m.shape[0] - 1 - np.argmax(m[::-1], axis=0), -(10**6)).astype(float)
    # За углами страницы (до CORNER_REACH столбцов) низ берётся по прямой через последние столбцы страницы.
    idx = np.nonzero(has)[0]
    lo, hi = int(idx.min()), int(idx.max())
    reach = np.zeros(len(has), bool)
    reach[max(0, lo - CORNER_REACH) : lo] = True
    reach[hi + 1 : hi + 1 + CORNER_REACH] = True
    us = np.arange(len(has), dtype=float)
    n_fit = min(CORNER_FIT, hi - lo + 1)
    left_line = robust_line(us[lo : lo + n_fit], bot_cols[lo : lo + n_fit])
    right_line = robust_line(us[hi + 1 - n_fit : hi + 1], bot_cols[hi + 1 - n_fit : hi + 1])
    left_reach = slice(max(0, lo - CORNER_REACH), lo)
    right_reach = slice(hi + 1, hi + 1 + CORNER_REACH)
    bot_cols[left_reach] = np.polyval(left_line, us[left_reach])
    bot_cols[right_reach] = np.polyval(right_line, us[right_reach])
    bot_roi = bot_cols[None, :]
    # Столбцы, где лист уходит за нижнюю рамку кадра: кромки там нет, полоса кромки не копируется, ободок не
    # перезаливается; кромка достраивается по соседним столбцам (ниже).
    at_frame = frame_all[c0:c1] & has
    inner = m & (rows > bot_roi - above) & ~at_frame[None, :]
    # Источник полосы — только настоящие строки кадра (ниже h — дорощенный холст).
    band = ~m & (rows > bot_roi) & (rows <= bot_roi + below) & (has | reach)[None, :] & ~at_frame[None, :]
    band &= rows + r0 < h
    m_d = shift_down(m, d)
    band_d = shift_down(band | gap, d) & ~m0
    # Щель под бумагой: от низа своей маски до сдвинутого низа (не «сдвинутая маска минус маска»: у тонкой маски
    # торца, ниже d, та оставляла бы дырку).
    gap_rows = (
        (rows > bot_m0[None, :]) & (rows <= bot_roi + d) & (has0 | (in_spine & has))[None, :] & ~at_frame[None, :]
    )
    virt_fill, synth, synth_mask = frame_extension(roi, at_frame, has, bot_cols, bot_m0, d, below, h - r0)
    fill = (gap_rows | inner | virt_fill) & ~band_d
    ex = None
    if exclude is not None:
        ex = exclude[r0:r1, c0:c1]
        band_d &= ~ex
        fill &= ~ex
    # Тон бумаги: чистые пиксели бумаги над кромкой по столбцам, в двух окнах по глубине — по ним продлевается
    # и вертикальный ход яркости (лист у кромки темнеет от загиба), с потолком :data:`TONE_TREND_MAX`.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ink = S.ink_of(gray)
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((2 * INK_PAD + 1, 2 * INK_PAD + 1), np.uint8)) > 0
    # Чистая бумага: не краска и не темнее бумаги страницы на TONE_DARK (полутоновая заливка логотипа серая и
    # порогом краски не ловится, а в замере тона дала бы серый шлейф под логотипом).
    level = float(np.median(gray[m0])) if m0.any() else 200.0
    clean = m0 & ~inner & ~near_ink & (gray >= level - TONE_DARK)
    half = TONE_DEPTH // 2
    good_up = clean & (rows >= bot_roi - TONE_DEPTH) & (rows < bot_roi - half)
    good_lo = clean & (rows >= bot_roi - half) & (rows < bot_roi - above)
    # Столбцы, где у кромки краска (логотип): нижнее окно пустое, тон берётся из окна выше — без тренда.
    blocked = good_lo.sum(axis=0) < 6
    good_lo |= good_up & blocked[None, :]
    tone_up, _ = column_tone(roi, good_up)
    tone_lo, grain = column_tone(roi, good_lo)
    trend = (tone_lo - tone_up) / half  # уровней на строку
    trend[blocked] = 0.0
    y_ref = bot_roi - half // 2
    ramp = np.clip(trend[None, :, :] * (rows - y_ref)[..., None], -TONE_TREND_MAX, TONE_TREND_MAX)
    paper = np.clip(tone_lo[None, :, :] + ramp + S.paper_grain(roi.shape[:2], grain, seed), 0, 255)
    out = roi.astype(np.float32)
    out[fill] = paper[fill]
    if ex is not None:
        synth_mask &= ~ex
    out[synth_mask] = synth[synth_mask]
    # Полоса кромки: копия исходника со сдвигом, перо наружу.
    shifted = np.zeros_like(out)
    shifted[d:] = roi[: roi.shape[0] - d]
    # Перо — только наружу (в фон), не в новую бумагу: иначе над кромкой проступает призрак строки.
    alpha = cv2.GaussianBlur(band_d.astype(np.float32), (0, 0), feather / 2)
    alpha[band_d] = 1.0
    alpha[(m_d | m0) & ~band_d] = 0.0
    alpha[synth_mask] = 0.0
    if ex is not None:
        alpha[ex] = 0.0
    out = out * (1 - alpha[..., None]) + shifted * alpha[..., None]
    result = img.copy()
    result[r0:r1, c0:c1] = np.clip(out, 0, 255).astype(np.uint8)
    paper_mask = np.zeros(mask.shape, bool)
    paper_mask[r0:r1, c0:c1] = fill & ~inner
    band_mask = np.zeros(mask.shape, bool)
    band_mask[r0:r1, c0:c1] = band_d
    return Extension(result, (c0, r0, c1, r1), pad, paper_mask, band_mask)
