"""Качество черт букв у сгиба против опорных букв той же строки: сжатие, бледность, потеря ядра штриха, распад, резкость.

Признаки — из литературы по качеству документных изображений, перенесённые на уровень одной буквы:
- бледность ядра — «выцветание» штриха (ink fading);
- доля ядра штриха среди всей краски буквы — обратная к transient pixels density (Alaei, Doermann et al., 2023);
  расплывшаяся перемычка даёт много полутона при малой глубине (fattened/faded stroke, Obafemi-Ajayi & Agam 2012);
- число кусков буквы на уровне половины глубины краски — broken character factor (Blando et al. 1995);
- резкость края: нормированный градиент, лапласиан, Tenengrad, DOM (Kumar–Chen–Doermann, как в
  ``ocr_utils.defocus_detection``) — для сравнения метрик расфокуса.

Все уровни — в долях отражения: яркость пикселя к уровню бумаги рядом с буквой. Так тень у сгиба (она гасит и
бумагу, и краску) сокращается, а бледнеющая краска — нет.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np

from ocr_utils.defocus_detection.metrics.dom import _region_sharpness as dom_region
from research.spread_defects.lines import END_GLYPHS, Glyphs, Line

# Поле вокруг буквы для уровня бумаги, в долях высоты буквы (не меньше 3 px).
PAD = 0.3
# Ядро краски буквы — этот процентиль яркости внутри рамки.
CORE_PCT = 5
# Пороги в долях глубины краски опорных букв строки: «краска вообще», «половина глубины», «ядро».
INK_LEVEL, HALF_LEVEL, CORE_LEVEL = 0.2, 0.5, 0.75
# Кусок буквы на уровне половины глубины учитывается, если он не меньше стольких px.
PIECE_MIN = 4

# Имена признаков буквы (порядок колонок таблиц).
FEATURES = ("aspect", "depth", "core_share", "pieces", "grad", "lap", "tenengrad", "dom", "midtone", "pitch")
# Признаки, считаемые по буквам (``pitch`` — по профилю строки, отдельно).
GLYPH_FEATURES = FEATURES[:-1]


@dataclass(frozen=True)
class GlyphMeasure:
    """Признаки одной буквы.

    Attributes:
        aspect: Ширина/высота рамки.
        depth: Глубина ядра краски: 1 − ядро/бумага (0 — краски нет, 1 — чёрная).
        paper: Уровень бумаги рядом с буквой (p90 рамки с полем), 0..255.
        core_share: Доля пикселей глубже ``CORE_LEVEL`` опорной глубины среди пикселей глубже ``INK_LEVEL``.
        pieces: Кусков на уровне ``HALF_LEVEL`` опорной глубины.
        grad: p95 модуля градиента к перепаду бумага − ядро (резкость края, 1/px).
        lap: Дисперсия лапласиана к квадрату перепада.
        tenengrad: Средний квадрат собелевского градиента к квадрату перепада.
        dom: DOM по рамке (порог края — по кадру).
        midtone: Доля пикселей между ``INK_LEVEL`` и ``HALF_LEVEL`` среди краски.
    """

    aspect: float
    depth: float
    paper: float
    core_share: float
    pieces: float
    grad: float
    lap: float
    tenengrad: float
    dom: float
    midtone: float


def glyph_crop(gray: np.ndarray, g: Glyphs, i: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Рамка буквы с полем и положение самой буквы внутри неё.

    Args:
        gray: Полутоновый кадр float32.
        g: Буквы кадра.
        i: Индекс буквы.

    Returns:
        (кроп с полем, (x0, y0, x1, y1) буквы в координатах кропа).
    """
    pad = max(3, int(PAD * g.h[i]))
    y0, x0 = max(0, g.y0[i] - pad), max(0, g.x0[i] - pad)
    y1, x1 = min(gray.shape[0], g.y0[i] + g.h[i] + pad), min(gray.shape[1], g.x0[i] + g.w[i] + pad)
    crop = gray[y0:y1, x0:x1]
    return crop, (g.x0[i] - x0, g.y0[i] - y0, g.x0[i] - x0 + g.w[i], g.y0[i] - y0 + g.h[i])


def measure_glyph(gray: np.ndarray, g: Glyphs, i: int, ref_depth: float, edge_threshold: float) -> GlyphMeasure:
    """Признаки одной буквы.

    Args:
        gray: Полутоновый кадр float32.
        g: Буквы кадра.
        i: Индекс буквы.
        ref_depth: Глубина ядра опорных букв строки (пороги уровней краски — в её долях; NaN — по своей глубине).
        edge_threshold: Порог края для DOM (``dom.gradient_threshold`` по кадру).

    Returns:
        ``GlyphMeasure``.
    """
    crop, (bx0, by0, bx1, by1) = glyph_crop(gray, g, i)
    paper = max(float(np.percentile(crop, 90)), 1.0)
    box = crop[by0:by1, bx0:bx1]
    refl = box / paper
    core = float(np.percentile(box, CORE_PCT))
    depth = 1.0 - core / paper
    base = ref_depth if np.isfinite(ref_depth) and ref_depth > 0.05 else max(depth, 0.05)
    ink = refl < 1.0 - INK_LEVEL * base
    half = refl < 1.0 - HALF_LEVEL * base
    strong = refl < 1.0 - CORE_LEVEL * base
    n_ink = max(int(ink.sum()), 1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(half.astype(np.uint8), connectivity=8)
    pieces = float((stats[1:, cv2.CC_STAT_AREA] >= PIECE_MIN).sum()) if count > 1 else 0.0
    span = max(paper - core, 1.0)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    lap = cv2.Laplacian(crop, cv2.CV_32F, ksize=3)
    dom, dom_weight = dom_region(crop.astype(np.uint8), edge_threshold) if min(crop.shape) >= 5 else (np.nan, 0)
    return GlyphMeasure(
        aspect=float(g.w[i]) / float(g.h[i]),
        depth=depth,
        paper=paper,
        core_share=float(strong.sum()) / n_ink,
        pieces=pieces,
        grad=float(np.percentile(magnitude, 95)) / span,
        lap=float(lap.var()) / span**2,
        tenengrad=float((magnitude**2).mean()) / span**2,
        dom=float(dom) if dom_weight else float("nan"),
        midtone=float((ink & ~half).sum()) / n_ink,
    )


@dataclass(frozen=True)
class LineMeasure:
    """Сводка одной строки у сгиба: концевые буквы против опорных.

    Attributes:
        side: ``L``/``R``.
        cy: Высота концевой буквы, px.
        edge: Расстояние концевой буквы от сгиба, px.
        at_edge: Строка доходит до края колонки.
        glyph_h: Медианная высота опорных букв строки, px.
        ref_depth: Глубина ядра опорных букв (пересвет — по ней).
        ref_paper: Уровень бумаги у опорных букв.
        ratio: Отношения медиан «концевые / опорные» по ``FEATURES`` (словарь имя → значение).
        tabular: Строка кончается у сгиба отдельной группой (номер страницы за отточием или широким пробелом).
        absolute: Медианы признаков концевых знаков без деления на опору (для табличных строк).
    """

    side: str
    cy: float
    edge: float
    at_edge: bool
    glyph_h: float
    ref_depth: float
    ref_paper: float
    ratio: dict
    tabular: bool = False
    absolute: dict | None = None


# Табличная строка: концевая группа (не больше ``TAB_MAX`` знаков) отделена от остального текста строки пробелом
# шире ``TAB_GAP`` высот буквы (у отточия точки мельче буквы и в строку не входят — остаётся широкий пробел).
TAB_GAP, TAB_MAX = 1.3, 4
# Допуск высоты концевого знака к медиане опорных строчных.
SAME_HEIGHT = 0.2


def tab_group(g: Glyphs, line: Line, side: str) -> int:
    """Размер концевой группы табличной строки (номер страницы у сгиба) или 0, если строка обычная.

    Args:
        g: Буквы кадра.
        line: Строка (буквы от сгиба наружу).
        side: ``L``/``R``.

    Returns:
        Число знаков группы (1..``TAB_MAX``) или 0.
    """
    idx = line.glyphs
    hgt = float(np.median(g.h[idx]))
    for k in range(min(TAB_MAX, len(idx) - 1)):
        near, far = idx[k], idx[k + 1]
        if side == "L":
            gap = g.x0[near] - (g.x0[far] + g.w[far])
        else:
            gap = g.x0[far] - (g.x0[near] + g.w[near])
        if gap > TAB_GAP * hgt:
            return k + 1
    return 0


def measure_line(gray: np.ndarray, g: Glyphs, line: Line, side: str, edge_threshold: float) -> LineMeasure | None:
    """Концевые буквы строки против опорных.

    Args:
        gray: Полутоновый кадр float32.
        g: Буквы кадра.
        line: Строка у сгиба.
        side: Сторона.
        edge_threshold: Порог края DOM по кадру.

    Returns:
        ``LineMeasure`` или None, если опорных букв меньше трёх.
    """
    refs = line.refs
    if len(refs) < 3:
        return None
    group = tab_group(g, line, side)
    first = [measure_glyph(gray, g, i, np.nan, edge_threshold) for i in refs]
    ref_depth = float(np.median([m.depth for m in first]))
    ref = [measure_glyph(gray, g, i, ref_depth, edge_threshold) for i in refs]
    # Группа у сгиба, отделённая от строки широким пробелом (номер страницы, обрывок тени или края листа за
    # сгибом), — не буквы строки: она пропускается. Концевые — следующие знаки той же высоты, что строчные
    # опорные (±``SAME_HEIGHT``): заглавные, инициалы и цифры — другой набор (1983/12 IMG_0063; обрывки за сгибом
    # давали «порчу» 0.000 на 1979/07 IMG_0017 и др., отзыв пользователя 2026-10-04).
    ref_h = float(np.median(g.h[refs]))
    candidates = line.glyphs[group : group + END_GLYPHS + 3]
    end_ids = np.array([i for i in candidates if abs(g.h[i] / ref_h - 1.0) <= SAME_HEIGHT][:END_GLYPHS], int)
    if len(end_ids) < 2:
        return None
    ends = [measure_glyph(gray, g, i, ref_depth, edge_threshold) for i in end_ids]
    ratio = {"pitch": pitch_ratio(gray, g, line, side)}
    absolute = {}
    for name in GLYPH_FEATURES:
        r = np.nanmedian([getattr(m, name) for m in ref])
        e = np.nanmedian([getattr(m, name) for m in ends])
        ratio[name] = float(e / r) if np.isfinite(r) and r > 0 else float("nan")
        absolute[name] = float(e)
    return LineMeasure(
        side=side,
        cy=line.cy,
        edge=line.edge,
        at_edge=line.at_edge,
        glyph_h=float(np.median(g.h[refs])),
        ref_depth=ref_depth,
        ref_paper=float(np.median([m.paper for m in ref])),
        ratio=ratio,
        tabular=bool(group),
        absolute=absolute,
    )


def lines_table(measures: list[LineMeasure]) -> dict[str, np.ndarray]:
    """Таблица строк кадра в виде колонок (для npz).

    Args:
        measures: Сводки строк.

    Returns:
        Колонки: side (U1), cy, edge, at_edge, glyph_h, ref_depth, ref_paper и по колонке на признак ``FEATURES``.
    """
    table = {
        "side": np.array([m.side for m in measures], dtype="U1"),
        **{
            f.name: np.array([getattr(m, f.name) for m in measures], float)
            for f in fields(LineMeasure)
            if f.name not in ("side", "ratio", "at_edge", "tabular", "absolute")
        },
        "at_edge": np.array([m.at_edge for m in measures], bool),
    }
    for name in FEATURES:
        table[name] = np.array([m.ratio[name] for m in measures], float)
    table["tabular"] = np.array([m.tabular for m in measures], bool)
    for name in GLYPH_FEATURES:
        table[f"abs_{name}"] = np.array([(m.absolute or {}).get(name, np.nan) for m in measures], float)
    return table


# --- Сжатие по шагу штрихов строки --------------------------------------------------------------------------
# Форма концевых компонент (``aspect``) ненадёжна: у корешка тонкие соединения букв «и», «н», «п», «м» рвутся, буква
# распадается на палочки с ширина/высота ≈ 0.3 — мера принимала распад за сжатие (1990/01 IMG_0037, 1977/07
# IMG_0085, отзыв пользователя 2026-10-04). Шаг вертикальных штрихов (первый пик автокорреляции профиля краски
# вдоль строки) от слипания и распада букв не зависит: он меняется только от горизонтального сжатия.

# Окно у конца строки и опорное окно, в высотах буквы; поиск пика автокорреляции, в высотах буквы.
END_SPAN, REF_SPAN = 3.0, 8.0
LAG_LO, LAG_HI = 0.15, 0.9


def stroke_pitch(profile: np.ndarray, glyph_h: float) -> float:
    """Шаг штрихов по профилю краски вдоль строки: первый локальный максимум автокорреляции.

    Args:
        profile: Тёмность по столбцам (бумага − серый), усреднённая по высоте строки.
        glyph_h: Высота буквы, px.

    Returns:
        Шаг, px (NaN — профиль короткий или пика нет).
    """
    lo, hi = max(2, int(LAG_LO * glyph_h)), int(LAG_HI * glyph_h)
    if len(profile) < 2 * hi:
        return float("nan")
    p = profile - profile.mean()
    acf = np.correlate(p, p, mode="full")[len(p) - 1 :]
    if acf[0] <= 0:
        return float("nan")
    acf = acf / acf[0]
    peaks = [k for k in range(lo, hi) if acf[k] > acf[k - 1] and acf[k] >= acf[k + 1] and acf[k] > 0.05]
    if not peaks:
        return float("nan")
    k = peaks[0]
    a, b, c = acf[k - 1], acf[k], acf[k + 1]
    denom = a - 2 * b + c
    return float(k + (0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0))


def line_profile(gray: np.ndarray, g: Glyphs, idx: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Профиль тёмности по столбцам ``x0..x1`` в полосе высот букв ``idx``.

    Args:
        gray: Полутоновый кадр float32.
        g: Буквы кадра.
        idx: Буквы, задающие полосу высот (центральная часть букв, без выносных).
        x0, x1: Границы по x.

    Returns:
        Профиль (пустой, если окно вне кадра).
    """
    hgt = float(np.median(g.h[idx]))
    cy = float(np.median(g.y0[idx] + g.h[idx] / 2))
    y0, y1 = int(max(0, cy - 0.35 * hgt)), int(min(gray.shape[0], cy + 0.35 * hgt))
    x0, x1 = max(0, x0), min(gray.shape[1], x1)
    if x1 - x0 < 4 or y1 - y0 < 3:
        return np.zeros(0)
    window = gray[y0:y1, x0:x1]
    paper = float(np.percentile(window, 95))
    return (paper - window).mean(axis=0)


def pitch_ratio(gray: np.ndarray, g: Glyphs, line: Line, side: str) -> float:
    """Сжатие у конца строки: шаг штрихов в ``END_SPAN`` высотах у сгиба к шагу в опорной части той же строки.

    Args:
        gray: Полутоновый кадр float32.
        g: Буквы кадра.
        line: Строка у сгиба.
        side: ``L``/``R``.

    Returns:
        Отношение шагов (1 — как в середине строки; NaN — не измерено).
    """
    if len(line.refs) < 3:
        return float("nan")
    hgt = float(np.median(g.h[line.glyphs]))
    end = line.glyphs[0]
    ends = line.glyphs[: max(3, END_GLYPHS)]
    if side == "L":
        e1 = int(g.x0[end] + g.w[end])
        e0 = int(e1 - END_SPAN * hgt)
    else:
        e0 = int(g.x0[end])
        e1 = int(e0 + END_SPAN * hgt)
    refs = line.refs
    r0, r1 = int(g.x0[refs].min()), int((g.x0[refs] + g.w[refs]).max())
    r1 = min(r1, int(r0 + REF_SPAN * hgt)) if side == "R" else r1
    r0 = max(r0, int(r1 - REF_SPAN * hgt)) if side == "L" else r0
    end_pitch = stroke_pitch(line_profile(gray, g, ends, e0, e1), hgt)
    ref_pitch = stroke_pitch(line_profile(gray, g, refs, r0, r1), hgt)
    return end_pitch / ref_pitch if ref_pitch > 0 else float("nan")
