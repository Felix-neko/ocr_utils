"""Полоса-вклейка у корешка разворота: лента ±25 мм вдоль разделителя, обнаружение по щели между экземплярами SAM 3, края полосы по строкам (ступень яркости или складка), вспомогательные линии по краям и линия корешка (без полосы — разделитель v2, с полосой — средний наклон краёв через её середину); стенд — ``research/gutter_strip``."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line, fit_line, mm_px

# Полуширина ленты поперёк корешка, мм бумаги (25 мм = 172 px сегментации): полоса 4–20 мм с запасом на наклон и сдвиг
# разделителя.
BAND_MM = 25.0
# Средняя доля высоты страниц, по которой строится лента (углы и закругления листа у верха и низа отброшены).
ROWS_SHARE = 0.8
# Окно «яркости бумаги без букв» вдоль корешка, мм: максимум яркости по вертикали снимает буквы и тонкие горизонтальные
# линейки, но не трогает вертикальные кромки полосы (3 мм = 21 px).
PAPER_WINDOW_MM = 3.0
# Щель между экземплярами SAM 3 шире этого — строка «полоса не взята», мм; метки страниц берутся у краёв ленты, дальше
# этого от корешка, мм.
GAP_MIN_MM = 3.0
LABEL_FROM_MM = 15.0
# Полоса есть, если медианная ширина щели между экземплярами SAM 3 у корешка (без оторванных островков страниц на
# полосе) не меньше этого, мм. Стенд ``research/gutter_strip``, разметка ПХ 1975 (253 кадра, 37 полос), 2026-10-23:
# 32 полосы из 37, ложных нет (прежняя щель до первого пикселя страницы при 7.8 — те же 32 без 03/IMG_0004, где клочки
# бумаги на полосе сужали щель с 12.2 до 3.6 мм). Запас мал: 05/IMG_0020 «нет» — 8.12, 03/IMG_0021 «полоса» — 8.26.
# Плато тона, ступени, складки, смена бумаги у сгиба, ширина без текста — хуже. Тёмная щель стыка тетрадей с ниткой
# уже: 2–6 мм.
STRIP_GAP_MM = 8.2
# Окно поиска края полосы по строке вокруг медианной кромки щели SAM 3, мм: язык экземпляра у рваной полосы уходит на
# 2–4 мм, кромка полосы — рядом.
EDGE_SEARCH_MM = 4.0
# Отклик края в строке слабее этого (уровни L: перепад на 2 px или глубина складки) — строка в прямую не идёт.
EDGE_MIN_RESPONSE = 4.0
# Строки сглаживаются вдоль корешка перед поиском края, мм.
EDGE_ALONG_MM = 3.0
# Складка — темнее бумаги на смещении ±0.7 мм с обеих сторон (5 px сегментации).
CREASE_OFFSET_MM = 0.7
# Запас маски полосы по сторонам при добавлении в бумагу кадра, мм (``strip_paper``).
STRIP_PAPER_MARGIN_MM = 2.0
# Края полосы «согласны», если их наклоны различаются не больше, градусы: иначе край цеплялся за складку или языки SAM 3
# (на 6 из 36 полос ПХ 1975 — 1.7–2.1°, 02/IMG_0067), и наклон корешка — по разделителю v2.
EDGE_AGREE_DEG = 1.0
# Язык кромки страницы у корешка: строка, где кромка отходит от сплошной кромки страницы (квантиль ``SOLID_EDGE_Q`` к
# странице) дальше этого, мм. Только признак «подозрительная лента» стенда ``research/gutter_strip`` (ПХ 1975/04
# IMG_0066, 0068, 0082: полоса под языками одной страницы), на решение о полосе не влияет.
TONGUE_MM = 2.0
SOLID_EDGE_Q = 0.9


@dataclass(frozen=True)
class Estimate:
    """Оценка способа обнаружения по кадру.

    Поля:
        score: Балл «полоса есть» (больше — увереннее), NaN — способ не смог.
        left_u: Левый край полосы (сдвиг от корешка ленты), px, или NaN.
        right_u: Правый край, px, или NaN.
        extra: Прочие признаки способа.
    """

    score: float
    left_u: float = float("nan")
    right_u: float = float("nan")
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StripBand:
    """Лента ±``BAND_MM`` у корешка, выпрямленная по прямой ``spine``: столбец — сдвиг поперёк от корешка.

    Поля:
        spine: Прямая, вдоль которой выпрямлена лента.
        ys: Строки кадра (сегментация) по строкам ленты.
        us: Сдвиги столбцов от корешка, px сегментации (минус — влево).
        light: Яркость L бумаги без букв (максимум по вертикали в ``PAPER_WINDOW_MM``), float.
        inst: Метки экземпляров SAM 3.
    """

    spine: Line
    ys: np.ndarray
    us: np.ndarray
    light: np.ndarray
    inst: np.ndarray

    def x_of(self, u: np.ndarray | float, rows: np.ndarray | None = None) -> np.ndarray:
        """Абсциссы кадра для сдвигов ``u`` в строках ленты ``rows`` (все строки — None)."""
        ys = self.ys if rows is None else self.ys[rows]
        return self.spine.x_at(ys) + np.asarray(u, np.float64)

    def mm(self, px: np.ndarray | float) -> np.ndarray | float:
        """px сегментации → мм бумаги."""
        return np.asarray(px, np.float64) / SEG_PX_PER_MM


@dataclass(frozen=True)
class Strip:
    """Полоса-вклейка кадра.

    Поля:
        left: Левая вспомогательная линия (край полосы со стороны левой страницы).
        right: Правая вспомогательная линия.
        mask: Маска полосы на масштабе сегментации (между линиями, в книге).
        width_mm: Ширина полосы посередине высоты, мм.
        left_points: Точки левого края по строкам (N, 2) ``x, y``, вошедшие в прямую.
        right_points: То же справа.
    """

    left: Line
    right: Line
    mask: np.ndarray
    width_mm: float
    left_points: np.ndarray
    right_points: np.ndarray


def rows_of(spine: Line, shape: tuple[int, int], share: float = ROWS_SHARE) -> np.ndarray:
    """Строки ленты: средняя доля ``share`` пролёта страниц, в пределах кадра."""
    pad = (1 - share) / 2 * (spine.y_bot - spine.y_top)
    return np.arange(max(int(spine.y_top + pad), 0), min(int(spine.y_bot - pad), shape[0] - 1) + 1)


def remap_band(img: np.ndarray, spine: Line, ys: np.ndarray, us: np.ndarray, nearest: bool) -> np.ndarray:
    """Выборка картинки вдоль прямой: строка ленты — строка кадра, столбец — сдвиг ``us`` от прямой.

    Args:
        img: Картинка масштаба сегментации (BGR, метки или bool).
        spine: Прямая.
        ys: Строки кадра.
        us: Сдвиги, px.
        nearest: Ближайший сосед (метки, маски) или билинейная (кадр).

    Returns:
        Лента ``(len(ys), len(us), …)`` того же типа (bool — как bool).
    """
    map_x = (spine.x_at(ys.astype(np.float64))[:, None] + us[None, :]).astype(np.float32)
    map_y = np.repeat(ys.astype(np.float32)[:, None], len(us), axis=1)
    src = img.astype(np.uint8) if img.dtype == bool else img
    out = cv2.remap(src, map_x, map_y, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # fmt: skip
    return out > 0 if img.dtype == bool else out


def band_axes(spine: Line, shape: tuple[int, int], band_mm: float = BAND_MM) -> tuple[np.ndarray, np.ndarray]:
    """Строки кадра и сдвиги столбцов ленты у корешка."""
    half = int(round(mm_px(band_mm)))
    return rows_of(spine, shape), np.arange(-half, half + 1, dtype=np.float64)


def paper_light(bgr_band: np.ndarray) -> np.ndarray:
    """Яркость бумаги без букв по ленте: максимум L по вертикали в окне ``PAPER_WINDOW_MM`` (float)."""
    lab = cv2.cvtColor(bgr_band, cv2.COLOR_BGR2LAB)
    win = max(3, int(round(mm_px(PAPER_WINDOW_MM))) | 1)
    return cv2.dilate(lab[..., 0], np.ones((win, 1), np.uint8)).astype(np.float32)


def build_strip_band(img: np.ndarray, inst: np.ndarray, spine: Line) -> StripBand:
    """Лента у корешка кадра для обнаружения и сегментации полосы.

    Args:
        img: Кадр масштаба сегментации, BGR.
        inst: Экземпляры-страницы SAM 3 того же масштаба.
        spine: Разделитель v2 (``auto``).

    Returns:
        Лента.
    """
    ys, us = band_axes(spine, img.shape[:2])
    light = paper_light(remap_band(img, spine, ys, us, nearest=False))
    return StripBand(spine, ys, us, light, remap_band(inst, spine, ys, us, nearest=True))


def label_near(inst: np.ndarray, cols: np.ndarray) -> int:
    """Самая частая ненулевая метка SAM 3 в столбцах ``cols`` ленты (0 — нет)."""
    v = inst[:, cols].ravel()
    v = v[v > 0]
    return int(np.bincount(v).argmax()) if len(v) else 0


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Наибольшая связная (8-связность) часть маски; пустая маска — как есть."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def detect_sam3(band: StripBand) -> Estimate:
    """Щель SAM 3 у корешка: щель между левой и правой страницей и рваность их кромок.

    Белую полосу SAM 3 не берёт никто (щель во всю ширину полосы), полосу с тенью рвёт один экземпляр языками. Балл —
    доля строк со щелью ≥ ``GAP_MIN_MM`` плюс рваность (наибольший межквартильный размах кромки, мм) / 5; признак
    «полоса есть» — медианная ширина щели ``extra["gap_mm"]`` ≥ ``STRIP_GAP_MM``.

    Args:
        band: Лента у корешка.

    Returns:
        Оценка с медианными кромками щели (``left_u``, ``right_u``) по строкам со щелью.
    """
    us = band.us
    far = mm_px(LABEL_FROM_MM)
    lab_l = label_near(band.inst, np.flatnonzero(us <= -far))
    lab_r = label_near(band.inst, np.flatnonzero(us >= far))
    if not lab_l or not lab_r or lab_l == lab_r:
        return Estimate(0.0, extra={"gap_share": 0.0, "gap_mm": 0.0, "ragged_mm": 0.0,
                                    "one_instance": int(lab_l == lab_r)})  # fmt: skip
    w = len(us)
    # Страница — её связная часть в ленте: оторванные островки экземпляра на самой полосе (клочки бумаги, пятна —
    # ПХ 1975/03 IMG_0004: в 54 % строк, медиана щели 3.6 мм вместо 12.2) щель не сужают; языки рваной кромки
    # связаны со страницей и остаются.
    left = largest_component(band.inst == lab_l)
    right = largest_component(band.inst == lab_r)
    has = left.any(axis=1) & right.any(axis=1)
    l_end = np.where(has, w - 1 - np.argmax(left[:, ::-1], axis=1), 0)[has]
    r_start = np.where(has, np.argmax(right, axis=1), 0)[has]
    if len(l_end) < 20:
        return Estimate(float("nan"), extra={"gap_mm": 0.0})
    gap = band.mm(r_start - l_end)
    gap_share = float((gap >= GAP_MIN_MM).mean())
    ragged = max(float(np.subtract(*np.percentile(band.mm(v), [75, 25]))) for v in (l_end, r_start))
    gap_rows = gap >= GAP_MIN_MM
    lu = float(np.median(us[l_end[gap_rows]])) if gap_rows.any() else float("nan")
    ru = float(np.median(us[r_start[gap_rows]])) if gap_rows.any() else float("nan")
    extra = {"gap_share": gap_share, "gap_mm": float(np.median(gap)), "ragged_mm": ragged, "one_instance": 0}
    extra.update(tongue_measures(band, l_end, r_start))
    return Estimate(gap_share + ragged / 5, lu, ru, extra)


def tongue_measures(band: StripBand, l_end: np.ndarray, r_start: np.ndarray) -> dict[str, float]:
    """Языки кромок страниц у корешка: насколько кромка каждой страницы отходит от её сплошной части на ленту.

    Сплошная кромка — квантиль ``SOLID_EDGE_Q`` положения кромки к своей странице (у левой — меньшие столбцы, у правой
    — бо́льшие); язык — строка, где кромка отошла от неё к корешку дальше ``TONGUE_MM``.

    Args:
        band: Лента у корешка.
        l_end: Последний столбец левой страницы по строкам с обеими страницами.
        r_start: Первый столбец правой страницы по тем же строкам.

    Returns:
        ``tongue_{left,right}_share`` — доля строк-языков, ``tongue_{left,right}_mm`` — 90-й перцентиль длины языка, мм,
        ``ragged_{left,right}_mm`` — межквартильный размах кромки, мм.
    """
    out = {}
    for name, edge, toward_page in (("left", l_end, -1), ("right", r_start, 1)):
        q = SOLID_EDGE_Q if toward_page > 0 else 1.0 - SOLID_EDGE_Q
        solid = float(np.quantile(edge, q))
        # Длина языка — отход кромки от сплошной части к корешку (у левой страницы — вправо, у правой — влево).
        tongue = band.mm(np.maximum(0.0, (solid - edge) * toward_page))
        out[f"tongue_{name}_share"] = float((tongue >= TONGUE_MM).mean())
        out[f"tongue_{name}_mm"] = float(np.percentile(tongue, 90))
        out[f"ragged_{name}_mm"] = float(np.subtract(*np.percentile(band.mm(edge), [75, 25])))
    return out


def edge_response(band: StripBand) -> np.ndarray:
    """Отклик края полосы по ленте: наибольшее из ступени яркости бумаги (|перепад на 2 px|) и глубины тонкой складки."""
    k = max(3, int(round(mm_px(EDGE_ALONG_MM))) | 1)
    light = cv2.blur(band.light, (1, k))
    step = np.zeros_like(light)
    step[:, 1:-1] = np.abs(light[:, 2:] - light[:, :-2])
    d = max(1, int(round(mm_px(CREASE_OFFSET_MM))))
    crease = np.minimum(np.roll(light, d, axis=1), np.roll(light, -d, axis=1)) - light
    crease[:, :d] = 0
    crease[:, -d:] = 0
    return np.maximum(step, crease)


def row_edges(band: StripBand, response: np.ndarray, centre_u: float) -> tuple[np.ndarray, np.ndarray]:
    """Край полосы по строкам: столбец наибольшего отклика в окне ``±EDGE_SEARCH_MM`` вокруг ``centre_u``.

    Args:
        band: Лента.
        response: Отклик края (``edge_response``).
        centre_u: Медианная кромка щели SAM 3 с этой стороны, px от корешка ленты.

    Returns:
        ``(индексы строк ленты, сдвиги края u)`` — строки с откликом не слабее ``EDGE_MIN_RESPONSE``.
    """
    c = int(np.argmin(np.abs(band.us - centre_u)))
    r = int(round(mm_px(EDGE_SEARCH_MM)))
    lo, hi = max(c - r, 0), min(c + r + 1, len(band.us))
    win = response[:, lo:hi]
    j = np.argmax(win, axis=1)
    ok = win[np.arange(len(win)), j] >= EDGE_MIN_RESPONSE
    return np.flatnonzero(ok), band.us[lo + j[ok]]


def segment_strip(band: StripBand, sam3: Estimate, book: np.ndarray) -> Strip | None:
    """Полоса по ленте и оценке SAM 3; None — полосы нет (щель уже ``STRIP_GAP_MM``) или края не построились.

    Args:
        band: Лента у корешка.
        sam3: Оценка ``detect_sam3`` (медианные кромки щели и её ширина).
        book: Грубая маска книги масштаба сегментации (маска полосы — в её пределах).

    Returns:
        Полоса или None.
    """
    gap = sam3.extra.get("gap_mm", 0.0)
    if not np.isfinite(sam3.left_u) or not np.isfinite(sam3.right_u) or gap < STRIP_GAP_MM:
        return None
    response = edge_response(band)
    lines, points = [], []
    for centre in (sam3.left_u, sam3.right_u):
        rows, us = row_edges(band, response, centre)
        xs = band.x_of(us, rows)
        ys = band.ys[rows].astype(np.float64)
        line = fit_line(xs, ys, band.spine.y_top, band.spine.y_bot)
        if line is None:
            return None
        lines.append(line)
        points.append(np.stack([xs, ys], axis=1))
    left, right = lines
    h, w = book.shape
    yy = np.arange(h, dtype=np.float64)
    xl, xr = left.x_at(yy), right.x_at(yy)
    cols = np.arange(w)[None, :]
    mask = (cols >= xl[:, None]) & (cols <= xr[:, None]) & book
    mid = (band.spine.y_top + band.spine.y_bot) / 2
    width = float((right.x_at(mid) - left.x_at(mid)) / mm_px(1.0))
    return Strip(left, right, mask, width, points[0], points[1])


def spine_line(strip: Strip | None, divider: Line, method: str) -> tuple[Line, str]:
    """Линия корешка кадра.

    Полосы нет — разделитель v2 (``auto``) как есть: маски у корешка целые, и он держит стык. Полоса есть — наклон —
    средний наклон двух её краёв, если они согласны (разница не больше ``EDGE_AGREE_DEG``), иначе наклон разделителя;
    положение — середина полосы (решение 2026-10-21). Наклон по краям книги и по внешним сторонам страниц отвергнут:
    крышка переплёта бывает не параллельна страницам (ПХ 1975/03 IMG_0064), у левой страницы внешняя сторона — срез
    толстого блока (+2…+4°, 04/IMG_0067); на 36 полосах ПХ 1975 против краёв полосы — медиана 0.45° и 0.28°, разделитель
    v2 — 0.11°.

    Args:
        strip: Полоса или None.
        divider: Разделитель v2 (``auto``).
        method: Способ, стоящий за разделителем (для подписи).

    Returns:
        ``(линия, откуда она)``.
    """
    if strip is None:
        return divider, f"разделитель v2 {method}"
    diff = abs(strip.left.tilt_deg - strip.right.tilt_deg)
    if diff <= EDGE_AGREE_DEG:
        k = float(np.tan((np.arctan(strip.left.k) + np.arctan(strip.right.k)) / 2))
        source = "средний наклон краёв полосы"
    else:
        k = divider.k
        source = f"наклон {method} (края полосы расходятся на {diff:.1f}°)"
    y0 = (divider.y_top + divider.y_bot) / 2
    x0 = float((strip.left.x_at(y0) + strip.right.x_at(y0)) / 2)
    return Line(x0, y0, k, divider.y_top, divider.y_bot), source + ", середина полосы"


def find_strip(img: np.ndarray, inst: np.ndarray, book: np.ndarray, divider: Line) -> tuple[Strip | None, Estimate]:
    """Полоса кадра: лента у разделителя, щель SAM 3, сегментация.

    Args:
        img: Кадр масштаба сегментации, BGR.
        inst: Экземпляры-страницы SAM 3.
        book: Грубая маска книги.
        divider: Разделитель v2 (``auto``).

    Returns:
        ``(полоса или None, оценка щели SAM 3)``.
    """
    band = build_strip_band(img, inst, divider)
    sam3 = detect_sam3(band)
    return segment_strip(band, sam3, book), sam3


def strip_paper(img: np.ndarray, inst: np.ndarray, book: np.ndarray, pages: list, paper: np.ndarray) -> np.ndarray:
    """Что полоса-вклейка добавляет к бумаге кадра: маска полосы в строках, где у корешка есть обе страницы, вне бумаги.

    Склейка страниц (``glue``) закрывает щель только до ``glue.W_MAX`` (≈ 11.6 мм) — у полосы 12–20 мм белая часть
    оставалась не бумагой и закрашивалась до прямоугольника (ПХ 1975/05 IMG_0021: щель 19 мм, склейка взяла 28 px из
    131). Полоса — настоящая бумага разворота.

    Args:
        img: Кадр масштаба сегментации, BGR.
        inst: Экземпляры-страницы SAM 3.
        book: Грубая маска книги.
        pages: Страницы кадра (``page_pose.PageQuad``) слева направо.
        paper: Текущая маска бумаги.

    Returns:
        Булева маска добавки (пустая — полосы нет или страниц с трапецией не две).
    """
    from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod, spine_candidates

    empty = np.zeros(paper.shape, bool)
    two = [p for p in pages if p.quad is not None]
    if len(two) != 2:
        return empty
    divider = spine_candidates(two)[SpineMethod.AUTO]
    strip, _ = find_strip(img, inst, book, divider)
    if strip is None:
        return empty
    # Полоса с запасом STRIP_PAPER_MARGIN_MM по сторонам: край полосы — по складке, а рваная маска страницы кончается
    # раньше, между ними оставалась белая щёлка 1–2 мм (05/IMG_0021); запас — тоже бумага (страницы или полосы).
    h, w = paper.shape
    yy = np.arange(h, dtype=np.float64)
    pad = mm_px(STRIP_PAPER_MARGIN_MM)
    cols = np.arange(w)[None, :]
    wide = (cols >= strip.left.x_at(yy)[:, None] - pad) & (cols <= strip.right.x_at(yy)[:, None] + pad) & book
    # Только строки, где есть обе страницы: выше и ниже полоса заходит на ткань переплёта и торцы.
    rows = two[0].mask.any(axis=1) & two[1].mask.any(axis=1)
    return wide & rows[:, None] & ~paper
