"""Линия сгиба разворота: самый тёмный непрерывный шов по «яркости бумаги без букв» в полосе у раздела страниц v2, сглаженный сплайном; перерезка масок страниц по сгибу."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import maximum_filter1d, median_filter

from research.page_cutting.boundary import bspline_basis

log = logging.getLogger(__name__)

# Полуширина полосы поиска сгиба вокруг раздела v2, мм: раздел по экземплярам SAM уходит от сгиба на 2–8 мм.
BAND_MM = 25.0
# Штраф за удаление шва от середины полосы: на краю полосы — столько уровней яркости (долина сгиба — 20–80 уровней);
# не даёт уйти на вертикальную линейку между колонками.
OFF_CENTRE_PENALTY = 10.0
# Высота окна максимума по вертикали, мм: больше межстрочного шага (4–6 мм) — буквы уходят в просвет между строками,
# а непрерывная по высоте тень сгиба остаётся тёмной.
PAPER_WINDOW_MM = 8.0
# Сглаживание карты вдоль сгиба (гаусс по вертикали), мм.
SMOOTH_ALONG_MM = 6.0
# Наибольший сдвиг шва за строку карты, px: шов почти вертикален.
MAX_STEP_PX = 1
# Отрезков сглаживающего сплайна сгиба по высоте и вес штрафа на вторую разность.
FOLD_KNOTS = 10
FOLD_SMOOTH = 2.0
# Сгиб — долина: светлее него должны быть обе стороны на расстоянии от VALLEY_NEAR_MM до VALLEY_FAR_MM. Край тёмной
# фотографии у корешка — ступенька, не долина (ХиП 1991/02 IMG_1091: шов уходил на край снимка).
VALLEY_NEAR_MM = 1.0
VALLEY_FAR_MM = 6.0
# Итераций и порог Хубера (мм) при сглаживании шва сплайном.
FOLD_IRLS = 6
FOLD_HUBER_MM = 0.7


@dataclass(frozen=True)
class Fold:
    """Линия сгиба разворота.

    Поля:
        points: Точки сгиба (N, 2) px маски, сверху вниз (по строке на точку).
        raw: Шов до сглаживания (N, 2).
        contrast: Насколько шов темнее медианы полосы в его строке, медиана по строкам (уровни яркости 0–255).
    """

    points: np.ndarray
    raw: np.ndarray
    contrast: float

    def x_at(self, y: np.ndarray) -> np.ndarray:
        """Абсцисса сгиба в строках ``y`` (вне его высоты — продолжение крайних точек)."""
        return np.interp(y, self.points[:, 1], self.points[:, 0])


def prior_split(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Раздел страниц v2 по строкам: середина между правым краем левой маски и левым краем правой.

    Args:
        left: Маска левой страницы.
        right: Маска правой страницы.

    Returns:
        ``(строки, абсциссы)`` там, где в строке есть обе страницы, или None.
    """
    rows = np.nonzero(left.any(axis=1) & right.any(axis=1))[0]
    if len(rows) < 20:
        return None
    w = left.shape[1]
    lx = w - 1 - np.argmax(left[rows, ::-1], axis=1)
    rx = np.argmax(right[rows], axis=1)
    return rows, (lx + rx) / 2.0


def paper_light(gray: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Яркость бумаги без букв: максимум по вертикальному окну ``PAPER_WINDOW_MM`` (буквы уходят в межстрочный просвет).

    Args:
        gray: Серая картинка полосы (вдоль сгиба — ось строк).
        px_per_mm: Плотность, px/мм.

    Returns:
        Карта float32 того же размера, сглаженная вдоль сгиба.
    """
    k = max(3, int(round(PAPER_WINDOW_MM * px_per_mm)) | 1)
    light = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))).astype(np.float32)
    sigma = SMOOTH_ALONG_MM * px_per_mm
    return cv2.GaussianBlur(light, (0, 0), sigmaX=0.6, sigmaY=sigma)


def darkest_seam(cost: np.ndarray) -> np.ndarray:
    """Непрерывный шов сверху вниз с наименьшей суммой ``cost`` (сдвиг не больше ``MAX_STEP_PX`` за строку).

    Args:
        cost: Карта стоимости (H, W).

    Returns:
        Столбец шва в каждой строке (H,).
    """
    h, w = cost.shape
    acc = cost[0].astype(np.float64).copy()
    back = np.zeros((h, w), np.int32)
    idx = np.arange(w)
    for r in range(1, h):
        best = acc.copy()
        arg = idx.copy()
        for s in range(1, MAX_STEP_PX + 1):
            # Из столбца слева и справа (со сдвигом s).
            left = np.full(w, np.inf)
            left[s:] = acc[:-s]
            right = np.full(w, np.inf)
            right[:-s] = acc[s:]
            take = left < best
            best[take], arg[take] = left[take], idx[take] - s
            take = right < best
            best[take], arg[take] = right[take], idx[take] + s
        back[r] = arg
        acc = best + cost[r]
    seam = np.zeros(h, np.int32)
    seam[-1] = int(np.argmin(acc))
    for r in range(h - 1, 0, -1):
        seam[r - 1] = back[r, seam[r]]
    return seam


def smooth_fold(y: np.ndarray, x: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Сглаживание шва ``x(y)`` B-сплайном со штрафом (IRLS Хубер), абсциссы в тех же строках."""
    t = (y - y[0]) / max(y[-1] - y[0], 1.0)
    design, knots = bspline_basis(t, FOLD_KNOTS)
    diff = np.diff(np.eye(design.shape[1]), n=2, axis=0)
    penalty = FOLD_SMOOTH * len(t) / design.shape[1] * diff.T @ diff
    w = np.ones_like(x)
    huber = FOLD_HUBER_MM * px_per_mm
    coef = np.zeros(design.shape[1])
    for _ in range(FOLD_IRLS):
        dw = design * w[:, None]
        coef = np.linalg.solve(design.T @ dw + penalty, dw.T @ x)
        r = np.abs(x - design @ coef)
        w = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-9))
    return BSpline(knots, coef, 3, extrapolate=True)(t)


def side_max(light: np.ndarray, near: int, far: int, to_right: bool) -> np.ndarray:
    """Наибольшая яркость по строке на расстоянии ``near…far`` px справа (``to_right``) или слева от каждого столбца."""
    size = far - near + 1
    # maximum_filter1d с окном size, центр сдвинут так, чтобы окно лежало целиком по нужную сторону.
    shift = near + size // 2
    m = maximum_filter1d(light, size=size, axis=1, mode="nearest")
    out = np.empty_like(light)
    if to_right:
        out[:, :-shift] = m[:, shift:]
        out[:, -shift:] = m[:, -1:]
    else:
        out[:, shift:] = m[:, :-shift]
        out[:, :shift] = m[:, :1]
    return out


def valley_cost(light: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Стоимость шва «долина»: яркость минус меньшая из наибольших яркостей слева и справа (отрицательно в долине).

    Ступенька (тёмная фотография у корешка) даёт около нуля: с тёмной стороны светлого нет.

    Args:
        light: Карта «яркость бумаги без букв» полосы.
        px_per_mm: Плотность, px/мм.

    Returns:
        Карта стоимости того же размера.
    """
    near = max(1, int(round(VALLEY_NEAR_MM * px_per_mm)))
    far = max(near + 1, int(round(VALLEY_FAR_MM * px_per_mm)))
    rim = np.minimum(side_max(light, near, far, False), side_max(light, near, far, True))
    # Долина сгиба окружена бумагой: края долины темнее уровня бумаги полосы — снимок или заливка, не сгиб
    # (ХиП 1991/02 IMG_1091: шов уходил в фотографию между её тонами).
    paper = float(np.percentile(light, PAPER_PERCENTILE))
    not_paper = np.maximum(0.0, paper - rim - RIM_PAPER_TOL)
    return light - rim + not_paper


# Края долины сгиба могут быть темнее уровня бумаги полосы не больше чем на столько уровней (тень изгиба у корешка).
RIM_PAPER_TOL = 30.0


def find_fold(gray: np.ndarray, left: np.ndarray, right: np.ndarray, px_per_mm: float) -> Fold | None:
    """Линия сгиба между двумя страницами.

    Полоса ``±BAND_MM`` вокруг прямой по разделу v2 выпрямляется (строка полосы — строка кадра, столбец — сдвиг от
    прямой); в ней — карта «яркость бумаги без букв», нормированная по строке (минус медиана); самый тёмный
    непрерывный шов; сглаживание сплайном.

    Args:
        gray: Кадр в сером (масштаб масок).
        left: Маска левой страницы.
        right: Маска правой страницы.
        px_per_mm: Плотность масштаба масок, px/мм.

    Returns:
        Сгиб или None (страниц в строках мало).
    """
    prior = prior_split(left, right)
    if prior is None:
        return None
    rows, xs = prior
    # Прямая раздела x = a·y + b (Хубер через fitLine) — ось полосы.
    line = cv2.fitLine(np.stack([xs, rows], axis=1).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    vx, vy, x0, y0 = (float(v) for v in line)
    slope = vx / vy if abs(vy) > 1e-6 else 0.0
    half = int(round(BAND_MM * px_per_mm))
    y_all = np.arange(rows.min(), rows.max() + 1, dtype=np.float32)
    centre = x0 + (y_all - y0) * slope
    offs = np.arange(-half, half + 1, dtype=np.float32)
    map_x = (centre[:, None] + offs[None, :]).astype(np.float32)
    map_y = np.repeat(y_all[:, None], len(offs), axis=1).astype(np.float32)
    band = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    light = paper_light(band, px_per_mm)
    cost = valley_cost(light, px_per_mm) + OFF_CENTRE_PENALTY * np.abs(offs)[None, :] / max(half, 1)
    seam = darkest_seam(cost)
    x_raw = centre + offs[seam]
    contrast = float(np.median(-cost[np.arange(len(seam)), seam]))
    x_smooth = smooth_fold(y_all, x_raw, px_per_mm)
    return Fold(np.stack([x_smooth, y_all], axis=1), np.stack([x_raw, y_all], axis=1), contrast)


def split_by_fold(left: np.ndarray, right: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Маски страниц, перерезанные по сгибу: бумага обеих страниц слева от сгиба — левая, справа — правая.

    Args:
        left: Маска левой страницы v2.
        right: Маска правой страницы v2.
        fold: Сгиб.

    Returns:
        ``(левая, правая)``.
    """
    paper = left | right
    h, w = paper.shape
    fx = fold.x_at(np.arange(h, dtype=np.float64))
    cols = np.arange(w)[None, :]
    is_left = cols < fx[:, None]
    return paper & is_left, paper & ~is_left


# Край страницы у корешка по цвету бумаги: просмотр от точки на SCAN_MM внутрь страницы от сгиба к сгибу.
SCAN_MM = 30.0
# Цвет бумаги страницы — перцентиль карты «яркость без букв» по всей странице: фотографии и заливки темнее бумаги,
# медиана на странице со снимком (ХиП 1991/02 IMG_1091) уходит в снимок.
PAPER_PERCENTILE = 80
# Не бумага этой страницы: светлее её больше чем на столько уровней L (белая вклейка, подложенный лист) …
EDGE_LIGHTER_L = 12.0
# … или с другим оттенком: |Δb| в Lab OpenCV больше порога. Пересвеченная бумага (L = 253 и у страницы, и у
# вклейки) различается только оттенком: ПХ 1975/09 IMG_0021 — b 123 у страницы против 127 у белой вклейки.
EDGE_DELTA_B = 3.0
# Слабые признаки вместе: светлее больше чем на EDGE_WEAK_L и оттенок отличается больше чем на EDGE_WEAK_B (пересвеченная
# вклейка ПХ 1975/09 IMG_0021: 253 против 244–251 и b 127 против 124 при шуме оттенка страницы ±3).
EDGE_WEAK_L = 1.5
EDGE_WEAK_B = 2.5
# Построчный эталон яркости бумаги — медиана первых REF_MM просмотра (глубоко в странице), сглаженная вдоль корешка.
REF_MM = 6.0
REF_SMOOTH_MM = 20.0
# Оттенок судится только там, где не темнее бумаги больше чем на столько уровней L: в тени у корешка оттенок плывёт
# (ХиП 1991/02 IMG_1091: затенённое поле правой страницы принималось за чужую бумагу).
TONE_DARK_TOL = 15.0
# Признак «не бумага» должен держаться подряд столько, мм (не реагировать на соринку).
EDGE_RUN_MM = 0.6


def page_spine_edge(bgr: np.ndarray, fold: Fold, page: np.ndarray, spine_right: bool, px_per_mm: float) -> np.ndarray:
    """Край страницы у корешка: где цвет бумаги этой страницы кончается по дороге к сгибу; нет смены — сам сгиб.

    Между страницами бывает вшита белая вклейка или виден подложенный лист (ПХ 1975/09 IMG_0021: полоса ~4 мм белее
    желтоватой страницы). SAM 3 рвёт её на языки, склейка v2 заливает щель — страница получает чужую бумагу. Здесь по
    каждой строке от точки на ``SCAN_MM`` внутрь страницы идём к сгибу по карте «цвет без букв» (яркость — максимум по
    вертикали, оттенок b — медиана) и останавливаемся, где бумага стала заметно светлее или сменила оттенок.
    Потемнение (тень изгиба у корешка) краем не считается: его ловит сгиб.

    Args:
        bgr: Кадр (масштаб масок).
        fold: Сгиб.
        page: Маска страницы (по ней — цвет бумаги).
        spine_right: Корешок справа от страницы (левая страница) — True, слева — False.
        px_per_mm: Плотность, px/мм.

    Returns:
        Точки края (N, 2), сверху вниз, сглаженные сплайном.
    """
    y = fold.points[:, 1].astype(np.float32)
    fx = fold.points[:, 0].astype(np.float32)
    n = int(round(SCAN_MM * px_per_mm))
    # Столбец 0 — глубже в странице, последний — на сгибе; направление внутрь страницы — от сгиба прочь.
    inward = -1.0 if spine_right else 1.0
    offs = np.arange(n + 1, dtype=np.float32)
    map_x = (fx[:, None] + inward * (n - offs[None, :])).astype(np.float32)
    map_y = np.repeat(y[:, None], n + 1, axis=1).astype(np.float32)
    lab = cv2.cvtColor(
        cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), cv2.COLOR_BGR2LAB
    )
    k = max(3, int(round(PAPER_WINDOW_MM * px_per_mm)) | 1)
    light = cv2.dilate(lab[..., 0], cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))).astype(np.float32)
    ref_l, ref_b = paper_colour(bgr, page, k)
    tone = paper_tone(lab[..., 2].astype(np.float32), ref_b, k)
    ref_cols = max(2, int(round(REF_MM * px_per_mm)))
    # Построчный эталон яркости не ниже общего уровня бумаги: строки со снимком в начале просмотра его не занижают.
    row_l = smooth_rows(np.median(light[:, :ref_cols], axis=1), REF_SMOOTH_MM * px_per_mm)
    ref_row = np.maximum(row_l, ref_l)[:, None]
    score = foreign_score(light - ref_row, np.abs(tone - ref_b), light < ref_row - TONE_DARK_TOL)
    score[:, :ref_cols] = 0.0
    # Та же оценка, но оттенок судится и в затенённой бумаге — для своей полосы за краем в ``strip_is_foreign``:
    # иначе плавный сдвиг оттенка у корешка выше порога тени даёт оценку 0, ниже — 1, и линия равной яркости
    # принимается за обрез вклейки (ПХ 1975/08 IMG_0055: край левой страницы в 5 мм от сгиба по ровной бумаге).
    score_open = foreign_score(light - ref_row, np.abs(tone - ref_b), light < ref_row - TONE_DARK_TOL, judge_shade=True)
    score_open[:, :ref_cols] = 0.0
    # Цена края в столбце j: Σ(1 − 2s) от j до сгиба (чужая бумага до сгиба — дешевле, своя — дороже); край на самом
    # сгибе — ноль. Самый дешёвый непрерывный шов по строкам — край страницы.
    dark = light < ref_row - TONE_DARK_TOL
    w = int(round(STEP_MM * px_per_mm))
    step = step_score(light, tone, dark, w)
    # Своя бумага у корешка бывает в тени (темнее эталона), и чужая светлее только её, а не эталона (ПХ 1975/05
    # IMG_0021): к цене края добавляется награда за ступеньку «своя → светлее/другой оттенок» в самом столбце края,
    # а полоса за краем считается чужой с оценкой не ниже ступеньки.
    # (step_fill отключён: размазывание ступеньки по полосе делало край рваным)
    tail = np.cumsum((1.0 - 2.0 * score)[:, ::-1], axis=1)[:, ::-1] - 0.0 * w * step
    cost = np.concatenate([tail, np.zeros((len(tail), 1))], axis=1)
    first = darkest_seam(cost.astype(np.float64))
    if not strip_is_foreign(score, first, int(round(EDGE_RUN_MM * px_per_mm)), FOREIGN_LINE_MM * px_per_mm, score_open):
        # Чужой полосы нет: край страницы — сгиб.
        return np.stack([fx.astype(np.float64), y.astype(np.float64)], axis=1)
    # Строки у верха и низа страницы не годятся: там просмотр уходит за лист.
    margin = int(round(EDGE_END_MM * px_per_mm))
    rows_ok = page_rows(page, y, margin)
    x_edge = fx + inward * (n - first)
    if rows_ok.sum() < 10:
        return np.stack([fx.astype(np.float64), y], axis=1)
    x_smooth = np.interp(y, y[rows_ok], smooth_fold(y[rows_ok], x_edge[rows_ok].astype(np.float64), px_per_mm))
    # Край не заходит за сгиб.
    x_smooth = np.minimum(x_smooth, fx) if spine_right else np.maximum(x_smooth, fx)
    return np.stack([x_smooth, y], axis=1)


# Разрыв в чужой бумаге у корешка (тёмная линия между вклейкой и сгибом), который ещё не обрывает её, мм.
EDGE_GAP_MM = 2.5
# Строки ближе к верху и низу страницы в край не идут, мм.
EDGE_END_MM = 3.0


# Оттенок судится только там, где бумага не темнее эталона больше чем на столько уровней: чужая бумага у корешка
# (вклейка, подложенный лист) не темнее страницы, а тень изгиба темнее и теплее (ХиП 1991/02 IMG_1091: b 134–146 в тени
# при 128 у бумаги).
TONE_SHADE_TOL = 3.0


# Полоса чужой бумаги принимается, если: медиана оценки в ней не ниже FOREIGN_IN, полоса той же ширины по другую
# сторону края (своя бумага) — не выше FOREIGN_OUT, и шире EDGE_RUN_MM она хотя бы в FOREIGN_ROWS строк. Перепад
# освещения даёт размытую оценку по обе стороны края и отбрасывается (ХиП 1991/02 IMG_1168).
FOREIGN_IN = 0.6
FOREIGN_OUT = 0.4
FOREIGN_ROWS = 0.5
# Край чужой полосы (обрез вклейки, подложенного листа) — почти прямая: 90-й перцентиль отклонения края от прямой
# больше порога — не обрез, а пятно света (РИСК 1992/01 IMG_0284), мм.
FOREIGN_LINE_MM = 3.0


def strip_is_foreign(
    score: np.ndarray, first: np.ndarray, run: int, line_px: float, score_open: np.ndarray | None = None
) -> bool:
    """Годится ли найденная полоса «чужой бумаги» у корешка (см. ``FOREIGN_IN``, ``FOREIGN_OUT``, ``FOREIGN_ROWS``).

    Args:
        score: Оценка «чужая бумага» (строки × столбцы, последний столбец — у сгиба).
        first: Столбец края по строкам.
        run: Наименьшая ширина полосы, столбцов.
        line_px: Наибольшее отклонение края полосы от прямой (90-й перцентиль по строкам), px.
        score_open: Оценка с оттенком, судимым и в затенённой бумаге (``foreign_score(..., judge_shade=True)``), — по
            ней меряется своя полоса за краем; None — по ``score``.

    Returns:
        True — полоса чужая бумага, край по ней; False — края нет.
    """
    n = score.shape[1]
    width = n - first
    wide = width >= run
    if wide.mean() < FOREIGN_ROWS:
        log.debug("полоса у корешка: строк шире %.2f — мало", wide.mean())
        return False
    rows = np.nonzero(wide)[0]
    line = np.polyfit(rows, first[rows].astype(np.float64), 1)
    dev = float(np.percentile(np.abs(first[rows] - np.polyval(line, rows)), 90))
    if dev > line_px:
        log.debug("полоса у корешка: край не прямой, отклонение %.1f px", dev)
        return False
    # Своя полоса за краем — по оценке, где оттенок судится и в тени: перепад освещения меняет оттенок по обе стороны
    # края, и снаружи оценка тоже высокая — полоса отвергается.
    outside_score = score if score_open is None else score_open
    inside, outside = [], []
    for i, (row, f, wd) in enumerate(zip(score, first, width)):
        if wd < run:
            continue
        inside.append(row[f:])
        outside.append(outside_score[i][max(0, f - wd) : f])
    vin = float(np.median(np.concatenate(inside)))
    vout = float(np.median(np.concatenate(outside))) if any(len(o) for o in outside) else 1.0
    log.debug("полоса у корешка: строк шире %.2f, оценка внутри %.2f, снаружи %.2f", wide.mean(), vin, vout)
    return vin >= FOREIGN_IN and vout <= FOREIGN_OUT


# Ширина полос по обе стороны кандидата края при местной ступеньке, мм; ступенька светлоты и оттенка, дающая оценку 1.
STEP_MM = 3.0
STEP_L = 6.0
STEP_B = 2.5


def step_score(light: np.ndarray, tone: np.ndarray, dark: np.ndarray, w: int) -> np.ndarray:
    """Местная оценка «чужая бумага начиная с этого столбца и до сгиба»: насколько полоса ``w`` после столбца
    светлее (и другого оттенка), чем полоса ``w`` перед ним; тёмные пиксели не считаются.

    Args:
        light: Яркость без букв (строки × столбцы, последний — у сгиба).
        tone: Оттенок b без букв.
        dark: Тёмные пиксели.
        w: Ширина полос, px.

    Returns:
        Оценка 0…1 на столбец (только для столбцов, после которых до сгиба всё — та же светлая бумага, считается
        вызывающим: здесь — оценка ступеньки в самом столбце, «размазанная» по полосе за ним максимумом).
    """
    ok = (~dark).astype(np.float32)
    lw = light * ok
    bw = tone * ok

    def win_mean(v: np.ndarray, start: int, length: int) -> np.ndarray:
        c = np.concatenate([np.zeros((len(v), 1), np.float32), np.cumsum(v, axis=1)], axis=1)
        n = v.shape[1]
        i0 = np.clip(np.arange(n) + start, 0, n)
        i1 = np.clip(np.arange(n) + start + length, 0, n)
        return c[:, i1] - c[:, i0]

    cnt_a, cnt_b = win_mean(ok, 0, w), win_mean(ok, -w, w)
    la, lb = win_mean(lw, 0, w) / np.maximum(cnt_a, 1), win_mean(lw, -w, w) / np.maximum(cnt_b, 1)
    ta, tb = win_mean(bw, 0, w) / np.maximum(cnt_a, 1), win_mean(bw, -w, w) / np.maximum(cnt_b, 1)
    valid = (cnt_a >= w / 2) & (cnt_b >= w / 2)
    edge = np.where(valid, np.maximum(np.clip((la - lb) / STEP_L, 0, 1), np.clip(np.abs(ta - tb) / STEP_B, 0, 1)), 0)
    return edge.astype(np.float32)


def step_fill(step: np.ndarray, w: int) -> np.ndarray:
    """Оценка полосы ``w`` сразу за ступенькой — не ниже самой ступеньки (максимум по окну, сдвинутому вправо)."""
    m = maximum_filter1d(step, size=w, axis=1, mode="constant", origin=-(w // 2))
    return np.roll(m, w // 2, axis=1) * 0 + m


def foreign_score(lighter: np.ndarray, dtone: np.ndarray, dark: np.ndarray, judge_shade: bool = False) -> np.ndarray:
    """Оценка «чужая бумага» 0…1 по столбцам у корешка: светлее бумаги страницы и/или другой оттенок.

    Args:
        lighter: Насколько светлее эталона бумаги, уровни L.
        dtone: |Δb| от оттенка бумаги.
        dark: Тёмные пиксели (долины у корешка, тень): оценка 0.5 — ни своя, ни чужая.
        judge_shade: Судить оттенок и там, где бумага темнее эталона больше чем на ``TONE_SHADE_TOL`` (для своей
            полосы за краем в ``strip_is_foreign``); False — только в незатенённой бумаге (поиск края).

    Returns:
        Карта оценки (float).
    """
    lit = np.ones_like(lighter, bool) if judge_shade else lighter >= -TONE_SHADE_TOL
    s = np.maximum(
        np.clip(lighter / EDGE_LIGHTER_L, 0.0, 1.0), np.where(lit, np.clip(dtone / (2 * EDGE_DELTA_B), 0.0, 1.0), 0.0)
    )
    # Слабые признаки вместе: чуть светлее и чуть другой оттенок — уже чужая бумага (пересвеченная вклейка).
    weak = (lighter > EDGE_WEAK_L) & (dtone > EDGE_WEAK_B)
    s = np.where(weak, np.maximum(s, 0.8), s)
    return np.where(dark, 0.5, s).astype(np.float32)


def edge_from_fold(other: np.ndarray, dark: np.ndarray, run: int, gap: int) -> int:
    """Столбец начала сплошной «не бумаги», доходящей до сгиба (последний столбец строки), или длина строки.

    Идём от сгиба внутрь страницы, пока «не бумага» (разрывы до ``gap`` столбцов допускаются); край — где она
    кончилась. Если у самого сгиба «не бумаги» меньше ``run`` столбцов — края нет (страница доходит до сгиба).
    Штамп или цветной заголовок посреди страницы до сгиба не дотягиваются и краем не становятся.

    Args:
        other: Признак «не бумага» по столбцам строки (последний — у сгиба).
        dark: Тёмные столбцы (долины у корешка): пропускаются, цепочку не рвут.
        run: Наименьшая ширина «не бумаги» у сгиба, столбцов.
        gap: Наибольший разрыв внутри неё, столбцов.

    Returns:
        Номер столбца края.
    """
    n = len(other)
    idx, misses, start = n - 1, 0, n
    while idx >= 0:
        if dark[idx]:
            idx -= 1
            continue
        if other[idx]:
            start, misses = idx, 0
        else:
            misses += 1
            if misses > gap:
                break
        idx -= 1
    return start if (other[start:] & ~dark[start:]).sum() >= run else n


def page_rows(page: np.ndarray, y: np.ndarray, margin: int) -> np.ndarray:
    """Строки ``y``, отстоящие от верха и низа страницы больше чем на ``margin`` px."""
    rows = np.nonzero(page.any(axis=1))[0]
    if not len(rows):
        return np.zeros(len(y), bool)
    return (y >= rows.min() + margin) & (y <= rows.max() - margin)


def paper_colour(bgr: np.ndarray, page: np.ndarray, k: int) -> tuple[float, float]:
    """Цвет бумаги страницы: яркость L (перцентиль ``PAPER_PERCENTILE`` карты «без букв») и оттенок b у пикселей такой
    яркости (±10 уровней), по маске страницы.

    Args:
        bgr: Кадр.
        page: Маска страницы.
        k: Высота вертикального окна максимума, px.

    Returns:
        ``(L, b)`` в шкале Lab OpenCV.
    """
    ys, xs = np.nonzero(page)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
    light = cv2.dilate(lab[..., 0], cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))).astype(np.float32)
    inside = page[y0:y1, x0:x1]
    level = float(np.percentile(light[inside], PAPER_PERCENTILE))
    paper = inside & (np.abs(lab[..., 0].astype(np.float32) - level) <= 10)
    tone = float(np.median(lab[..., 2][paper])) if paper.any() else float(np.median(lab[..., 2][inside]))
    return level, tone


def paper_tone(b: np.ndarray, ref_b: float, k: int) -> np.ndarray:
    """Оттенок бумаги без букв: из минимума и максимума b по вертикальному окну ``k`` — ближайший к оттенку бумаги.

    Буквы нейтральны (b ≈ 128) и сдвигают медиану к нейтральному — к белой вклейке; в окне выше межстрочного шага
    всегда есть просвет бумаги, и одно из крайних значений — её оттенок.

    Args:
        b: Канал b Lab (float).
        ref_b: Оттенок бумаги страницы.
        k: Высота окна, px.

    Returns:
        Карта оттенка того же размера.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    lo = cv2.erode(b, kernel)
    hi = cv2.dilate(b, kernel)
    return np.where(np.abs(lo - ref_b) <= np.abs(hi - ref_b), lo, hi)


def median_along(img: np.ndarray, k: int) -> np.ndarray:
    """Медиана по вертикальному окну ``k`` строк (по столбцам)."""
    return median_filter(img, size=(k, 1), mode="nearest")


def smooth_rows(v: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово сглаживание ряда по строкам."""
    return cv2.GaussianBlur(v.astype(np.float32)[:, None], (0, 0), sigmaX=0.1, sigmaY=max(sigma, 1.0)).ravel()


def trim_at_edge(mask: np.ndarray, edge: np.ndarray, spine_right: bool) -> np.ndarray:
    """Маска страницы без всего, что за краем у корешка (вне строк края — продолжение крайних точек)."""
    h, w = mask.shape
    ex = np.interp(np.arange(h, dtype=np.float64), edge[:, 1], edge[:, 0])
    cols = np.arange(w)[None, :]
    keep = cols <= ex[:, None] if spine_right else cols >= ex[:, None]
    return mask & keep
