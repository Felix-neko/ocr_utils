"""Строки у сгиба: буквы как связные компоненты, сборка строк от сгиба наружу, концевые и опорные буквы строки.

Всё, что меряется у корешка во второй итерации (сжатие, расфокус, порча черт, буквы под переплётом), сравнивает
концевые буквы строки (последние у левой полосы, первые у правой) с опорными буквами **той же строки** дальше от
сгиба: кегль, гарнитура, печать и освещение у них общие, и отношение их мер — чистый эффект изгиба листа.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Маска краски — та же, что у мер первой итерации (`gutter.glyph_components`): деталь без освещения темнее этого.
INK_DEPTH = 18
DETAIL_SIGMA = 10
# Тело кадра по высоте (без колонтитулов и нижнего поля), доли высоты.
BODY_TOP, BODY_BOTTOM = 0.10, 0.92
# Буква: высота в долях медианной высоты букв, ширина — не больше стольких высот.
GLYPH_H_LO, GLYPH_H_HI, GLYPH_W_HI = 0.55, 1.6, 2.5
# Компоненты для оценки медианной высоты буквы, px полного кадра (450 dpi: корпус ~23–26 px).
SEED_H = (10, 80)
# Сборка строки: соседняя буква — по вертикали не дальше доли высоты буквы, по горизонтали — зазор не больше
# стольких высот (пробел между словами ~0.5–1 h; межколонник — от ~3 h).
LINK_DY, LINK_GAP = 0.45, 2.6
# Строка годится, если в ней не меньше стольких букв.
MIN_LINE_GLYPHS = 10
# Концевых букв строки (ближайших к сгибу) и опорных — начиная с этой позиции от сгиба, не больше стольких.
END_GLYPHS, REF_FROM, REF_MAX = 3, 6, 14
# Строка доходит до края колонки, если её концевая буква не дальше края колонки (робастный) + стольких высот.
EDGE_SLACK = 1.5
# Начало строки ищется не дальше этого от сгиба, px полного кадра (за ним — вторая колонка или поле).
MAX_START = 900
# Буквы дальше этого от сгиба в сборку строк не идут, px (колонка у сгиба целиком, но не вся полоса).
REACH = 1600

LEFT, RIGHT = "L", "R"


@dataclass(frozen=True)
class Glyphs:
    """Буквы кадра — связные компоненты краски высотой с букву.

    Attributes:
        x0, y0, w, h: int[n] — рамки букв, px полного кадра.
        cx, cy: float[n] — центры масс.
        area: int[n] — площадь компоненты, px.
        glyph_h: Медианная высота буквы кадра, px.
    """

    x0: np.ndarray
    y0: np.ndarray
    w: np.ndarray
    h: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    area: np.ndarray
    glyph_h: float


def find_glyphs(gray: np.ndarray) -> Glyphs:
    """Буквы тела кадра.

    Args:
        gray: Полутоновый кадр float32 полного разрешения.

    Returns:
        ``Glyphs``; пустые массивы и NaN-высота, если букв не нашлось.
    """
    height = gray.shape[0]
    detail = gray - cv2.GaussianBlur(gray, (0, 0), DETAIL_SIGMA)
    ink = (detail < -INK_DEPTH).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(ink, connectivity=8)
    stats, centers = stats[1:], centers[1:]
    x0, y0, w, h, area = (stats[:, k] for k in range(5))
    cx, cy = centers[:, 0], centers[:, 1]
    body = (cy > BODY_TOP * height) & (cy < BODY_BOTTOM * height)
    seed = body & (h > SEED_H[0]) & (h < SEED_H[1])
    glyph_h = float(np.median(h[seed])) if seed.any() else float("nan")
    keep = body & (h > GLYPH_H_LO * glyph_h) & (h < GLYPH_H_HI * glyph_h) & (w >= 3) & (w < GLYPH_W_HI * glyph_h)
    return Glyphs(x0[keep], y0[keep], w[keep], h[keep], cx[keep], cy[keep], area[keep], glyph_h)


@dataclass(frozen=True)
class Line:
    """Строка полосы у сгиба: индексы букв от сгиба наружу.

    Attributes:
        glyphs: int[k] — индексы в ``Glyphs`` по порядку от сгиба.
        cy: Высота центра концевой буквы, px.
        edge: Расстояние от линии сгиба до ближнего к сгибу края концевой буквы, px (минус — буква за сгибом).
        at_edge: Строка доходит до края колонки (не концевая строка абзаца у левой полосы, не отступ у правой).
    """

    glyphs: np.ndarray
    cy: float
    edge: float
    at_edge: bool

    @property
    def ends(self) -> np.ndarray:
        """Концевые буквы строки (``END_GLYPHS`` ближайших к сгибу)."""
        return self.glyphs[:END_GLYPHS]

    @property
    def refs(self) -> np.ndarray:
        """Опорные буквы строки (с позиции ``REF_FROM``, не больше ``REF_MAX``)."""
        return self.glyphs[REF_FROM : REF_FROM + REF_MAX]


def fold_distance(g: Glyphs, fold: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Расстояния ближнего и дальнего края букв от линии сгиба для одной полосы.

    Args:
        g: Буквы кадра.
        fold: Прямая сгиба ``x = k*y + b``.
        side: ``L`` или ``R``.

    Returns:
        (ближний край, дальний край) — px; положительно в сторону своей полосы (от сгиба наружу).
    """
    fx = np.polyval(fold, g.cy)
    if side == LEFT:
        return fx - (g.x0 + g.w), fx - g.x0
    return g.x0 - fx, g.x0 + g.w - fx


def side_lines(g: Glyphs, fold: np.ndarray, side: str) -> list[Line]:
    """Строки полосы у сгиба: сборка букв снаружи к сгибу с продолжением наклона строки.

    Вдали от сгиба строки прямые, у сгиба лист изгибается и строка круто уходит вверх или вниз. Поэтому буквы
    перебираются по убыванию расстояния от сгиба (не дальше ``REACH``): строка начинается там, где она прямая,
    а к сгибу продолжается по наклону последних букв. Буква присоединяется к строке, если отклонение её центра
    от продолжения меньше ``LINK_DY`` высоты буквы, а зазор по горизонтали меньше ``LINK_GAP`` высот.
    Межколонник (зазор больше) рвёт строку: у двухколонной полосы к сгибу выходит строка ближней колонки.

    Args:
        g: Буквы кадра.
        fold: Прямая сгиба.
        side: ``L`` или ``R``.

    Returns:
        Строки (буквы — от сгиба наружу) не короче ``MIN_LINE_GLYPHS`` букв, чья ближняя к сгибу буква не дальше
        ``MAX_START``; ``at_edge`` — по робастному краю колонки.
    """
    near, far = fold_distance(g, fold, side)
    hgt = g.glyph_h
    mine = np.nonzero((far > -0.5 * hgt) & (near < REACH))[0]
    order = mine[np.argsort(-far[mine])]
    chains: list[list[int]] = []
    # Состояние строки: (cy последней буквы, ближний край последней буквы, наклон dy/dx в сторону сгиба).
    states: list[tuple[float, float, float]] = []
    closed: list[list[int]] = []
    for idx in order:
        d_far, cy = far[idx], g.cy[idx]
        live = [k for k, st in enumerate(states) if st[1] - d_far < LINK_GAP * hgt]
        live_set = set(live)
        closed.extend(chains[k] for k in range(len(chains)) if k not in live_set)
        chains, states = [chains[k] for k in live], [states[k] for k in live]
        best, best_dy = -1, LINK_DY * hgt
        for k, (tail_cy, tail_near, slope) in enumerate(states):
            step = tail_near - near[idx] if near[idx] < tail_near else 0.0
            predicted = tail_cy + slope * step
            dy = abs(cy - predicted)
            if dy < best_dy and d_far < tail_near + 0.5 * hgt:
                best, best_dy = k, dy
        if best >= 0:
            tail_cy, tail_near, slope = states[best]
            step = max(tail_near - near[idx], 1.0)
            # Наклон — сглаженный по последним буквам; знак «вверх/вниз» к сгибу.
            new_slope = (cy - tail_cy) / step if len(chains[best]) >= 3 else slope
            slope = 0.6 * slope + 0.4 * float(np.clip(new_slope, -1.0, 1.0))
            chains[best].append(idx)
            states[best] = (cy, min(tail_near, near[idx]), slope)
        else:
            chains.append([idx])
            states.append((cy, near[idx], 0.0))
    closed.extend(chains)
    lines = []
    for chain in closed:
        if len(chain) < MIN_LINE_GLYPHS:
            continue
        ordered = np.array(sorted(chain, key=lambda i: near[i]))
        if near[ordered[0]] < MAX_START:
            lines.append(ordered)
    if not lines:
        return []
    edges = np.array([near[chain[0]] for chain in lines])
    # Край колонки — нижний квартиль концов строк (большинство строк набора доходит до края).
    column_edge = float(np.percentile(edges, 25))
    return [
        Line(chain, float(g.cy[chain[0]]), float(edge), bool(edge <= column_edge + EDGE_SLACK * hgt))
        for chain, edge in zip(lines, edges)
    ]
