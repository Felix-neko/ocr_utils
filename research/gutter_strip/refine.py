"""Уточняющая сегментация межстраничной ленты у корешка в полном разрешении: набором способов (SAM 2.1 по точкам страниц и полосы, SAM 3 по тексту, прямые ступени и складки без сети) по строкам ленты — правый край левой страницы и левый край правой; приёмка краёв, выбор согласного и ближнего к своей странице, решение «полоса есть» по ширине между ними."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from ocr_utils.scan_cropping.v2.edge_refine import (
    CHUNK_MM,
    END_SHARE,
    INK_CHROMA,
    INK_DROP,
    MIN_COVER,
    OVERLAP_MM,
    RAGGED_OK_MM,
    SAM3_PROMPTS,
    STEP_MAX_MM,
    clahe,
    roughness_mm,
    step_mm,
)
from ocr_utils.scan_cropping.v2.gutter_strip import GAP_MIN_MM, Estimate, detect_sam3
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line
from research.gutter_strip.band import Band, StripFrame, build_band
from research.gutter_strip.detect import detect_creases, detect_edges

# Полуширина ленты от разделителя, мм (как ``gutter_strip.BAND_MM``): полоса ПХ 1975 — 8–15 мм, страница — дальше.
BAND_HALF_MM = 25.0
# Лента продлевается за верх и низ пролёта страниц, мм: у концов — фон за краем листа и задиры полосы.
END_PAD_MM = 12.0
# Положительные точки страницы — на этом расстоянии от разделителя, мм (страница там почти всегда, полоса — ближе).
PAGE_POS_MM = 15.0
# Подсказки SAM 3 для самой полосы (полный набор).
STRIP_PROMPTS = ("paper strip", "strip of paper", "tape")
# Прямизна края полосы: p90 остатка строк на прямой к ней, без ``STRAIGHT_END_MM`` у концов пролёта, мм (стартовое
# значение; ПХ 1975/04 IMG_0066: край правой страницы, согласный у всех способов, — 0.59–0.65 мм).
STRAIGHT_OK_MM = 0.8
STRAIGHT_END_MM = 15.0
# Прямая края ищется RANSAC: строка на прямой, если ближе ``LINE_TOL_MM``; край принят, если на прямой не меньше
# ``LINE_MIN_SHARE`` годных строк (кусок, ушедший на кончики языков — ПХ 1975/04 IMG_0082, правый край в средней трети
# на +0.5…+1.4 мм вместо +7…+10, — выброс, а не отказ способа). Выбросы — пропуски края.
LINE_TOL_MM = 1.0
LINE_MIN_SHARE = 0.6
RANSAC_TRIALS = 300
# Край — гладкие куски (``edge_pieces``): полоса может менять ширину и идти по части линии разреза, её кромки — любые
# гладкие кривые (указание пользователя 2026-10-23). Кусок рвётся на скачке больше ``JUMP_MM`` на ``JUMP_SPAN_MM``
# или на пропуске длиннее ``JUMP_SPAN_MM``; кусок короче ``PIECE_MIN_MM`` — выброс. Объезд — кусок, оба соседа которого
# ближе к своей странице дальше ``DETOUR_MM`` и сходятся между собой через него в ``DETOUR_MM`` (кусок способа на
# кончиках языков: ПХ 1975/04 IMG_0082, правый край в средней трети на +0.5…+1.4 мм между +7 и +10).
JUMP_MM = 1.5
JUMP_SPAN_MM = 3.0
PIECE_MIN_MM = 20.0
DETOUR_MM = 2.0
# Края не должны заходить друг за друга глубже этого, мм (в доле строк ``ORDER_SHARE``).
ORDER_TOL_MM = 1.0
ORDER_SHARE = 0.95
# Краска своей страницы за новым краем (на ленту), мм²; ближе ``INK_FROM_EDGE_MM`` к краю не в счёт — там обрывки
# тёмной щели сгиба (ПХ 1975/04 IMG_0066: 11 мм² «краски» в 0.1–2.0 мм за краем левой страницы, ни одной буквы).
# Порог 10: пометки на самой полосе под языками страницы — до 8 мм² (05/IMG_0021), край, режущий колонку текста, —
# десятки мм² (буква 9 pt ~0.5 мм², 3 буквы на строку в 5 мм, 30 строк).
INK_OUT_MM2 = 10.0
INK_FROM_EDGE_MM = 2.5
# Глифы краски — компоненты не крупнее, мм / не мельче, мм².
GLYPH_MAX_H_MM = 8.0
GLYPH_MAX_W_MM = 10.0
GLYPH_MIN_MM2 = 0.2
# Согласие способов: медиана расхождения с общим краем не больше, мм.
AGREE_MM = 2.0
# Способ без сети (прямые ступени и складки) годен, если его пара прямых видна хотя бы в этой доле строк ленты.
LINES_MIN_SHARE = 0.4
# Полоса есть, если медиана ширины между выбранными краями не меньше, мм. Множества 2026-10-23: полосы ПХ 1975 — от
# 5.1 мм (кроме промахов 1.6 и 2.4), без полосы ПХ и ХиП (60 кадров) — до 4.4 мм (01/IMG_0022, 03/IMG_0022).
STRIP_WIDTH_MM = 5.0
# Отклик линии на снимке (ступень светлоты бумаги без букв или тонкая складка), уровни L 0–255: ступень — перепад на
# ``RESP_STEP_MM`` по обе стороны, складка — темнее бумаги на ``RESP_CREASE_MM`` с обеих сторон; сглаживание вдоль
# корешка — ``RESP_ALONG_MM``.
RESP_STEP_MM = 0.3
RESP_CREASE_MM = 0.7
RESP_ALONG_MM = 3.0
# Опора края — медиана по годным строкам наибольшего отклика в ``SUPPORT_HALF_MM`` от края. Край ближе к своей
# странице, чем выбранный по согласию, вытесняет его, если отстоит дальше ``AGREE_MM`` и опора у него не слабее
# ``SUPPORT_STRONG_L``: кромка листа за белой полосой, которую сети считают той же бумагой (ПХ 1975/04 IMG_0083: линия
# на +16 мм — 13.7; ложная кромка 10/IMG_0067 на +4 мм — 6.4; настоящие слабые кромки полос 0082, 03/IMG_0004 — 3–5,
# их берут сети). Стартовое значение, калибруется по множеству.
SUPPORT_HALF_MM = 0.5
SUPPORT_STRONG_L = 10.0
# Вытесняющий край отдаёт страницу полосе — краски своей страницы за ним должно быть почти нет, мм² (ПХ 1975/04
# IMG_0083 — 3; граница тени загиба ХиП 1991/01 IMG_1045 через буквы у корешка — 8).
OVERRIDE_INK_MM2 = 5.0
# Способ ``solid``: сплошная кромка страницы грубого SAM 3 — прямая по строкам, где кромка ближе к своей странице
# (доля ``SOLID_SHARE`` строк), притянутая к наибольшему отклику линии в ``SOLID_SNAP_MM`` от неё. ПХ 1975/04
# IMG_0083: языки правой страницы — до +5 мм, кромка листа — тонкая линия на +16 мм, сети берут сгиб на +3.3 мм.
SOLID_SHARE = 0.4
SOLID_SNAP_MM = 2.5
# Поиск наклонной линии кромки (``best_line``): наклоны к разделителю, градусы; окно — от сплошной кромки SAM 3
# ``SOLID_BACK_MM`` к корешку и ``SOLID_AHEAD_MM`` к странице (на 0083 кромка листа — на 4.5 мм дальше к странице).
LINE_TILTS_DEG = np.arange(-1.5, 1.51, 0.1)
SOLID_BACK_MM = 2.5
SOLID_AHEAD_MM = 6.0
# Ядро пальца ближе этого к разделителю (по горизонтали) делает строку негодной, мм.
FINGER_NEAR_MM = 20.0


class Engine(StrEnum):
    """Источник краёв способа."""

    SAM2 = "sam2"  # страницы точками: свои положительные, чужие и щель — отрицательные
    SAM2_STRIP = "sam2strip"  # сама полоса точками: положительные в середине щели грубого SAM 3
    SAM3 = "sam3"  # страницы текстом «book page» / «paper page»
    SAM3_STRIP = "sam3strip"  # полоса текстом «paper strip» / «tape»
    EDGES = "edges"  # пара прямых ступеней яркости (``detect.detect_edges``), без сети
    CREASES = "creases"  # пара прямых складок (``detect.detect_creases``), без сети
    SOLID = "solid"  # сплошная кромка страниц грубого SAM 3, притянутая к линии на снимке, без сети


class Prep(StrEnum):
    """Подготовка куска перед сетью."""

    NONE = "none"
    LEVELS = "lvL"  # растяжение светлоты по бумаге куска
    CLAHE = "clahe"


@dataclass(frozen=True)
class Method:
    """Способ: источник, сжатие куска вдоль корешка, подготовка."""

    engine: Engine
    squeeze: int = 1
    prep: Prep = Prep.NONE

    @property
    def name(self) -> str:
        """Имя для сводки: ``sam2_sq4_lvL``, ``edges``."""
        if self.engine in (Engine.EDGES, Engine.CREASES, Engine.SOLID):
            return self.engine.value
        tail = "" if self.prep is Prep.NONE else f"_{self.prep.value}"
        return f"{self.engine.value}_sq{self.squeeze}{tail}"

    @property
    def neural(self) -> bool:
        """Способ — сеть (без сети край выбирается только при согласии с сетью)."""
        return self.engine not in (Engine.EDGES, Engine.CREASES, Engine.SOLID)


REDUCED = (
    Method(Engine.SAM2, 4),
    Method(Engine.SAM2, 4, Prep.LEVELS),
    Method(Engine.SAM3, 4),
    Method(Engine.EDGES),
    Method(Engine.SOLID),
)
FULL = REDUCED + (
    Method(Engine.SAM2_STRIP, 4),
    Method(Engine.SAM2, 1),
    Method(Engine.SAM3, 1),
    Method(Engine.SAM2, 4, Prep.CLAHE),
    Method(Engine.SAM3, 4, Prep.LEVELS),
    Method(Engine.SAM3_STRIP, 4),
    Method(Engine.CREASES),
)


class SpineBand:
    """Лента ±``BAND_HALF_MM`` вдоль разделителя в полном разрешении: строка — строка кадра, столбец — горизонтальный
    сдвиг от разделителя (как ``gutter_strip.remap_band``, только в полном разрешении)."""

    def __init__(self, spine: Line, y_top: float, y_bot: float, ratio: float) -> None:
        """Геометрия ленты.

        Args:
            spine: Разделитель на масштабе сегментации.
            y_top: Верх пролёта страниц, px сегментации (лента начинается на ``END_PAD_MM`` выше).
            y_bot: Низ пролёта, px сегментации.
            ratio: Полное разрешение / масштаб сегментации.
        """
        self.spine = spine
        self.ratio = ratio
        self.ppm = ratio * SEG_PX_PER_MM
        pad = END_PAD_MM * SEG_PX_PER_MM
        self.y0 = (y_top - pad) * ratio
        self.length = int(round((y_bot - y_top + 2 * pad) * ratio))
        self.half = int(round(BAND_HALF_MM * self.ppm))
        self.width = 2 * self.half + 1
        # Строки ленты в полном кадре и в масштабе сегментации.
        self.ys_full = self.y0 + np.arange(self.length, dtype=np.float64)
        self.ys_seg = self.ys_full / ratio
        # Пролёт страниц в строках ленты (вне него — продление за край листа).
        self.page_rows = (int(round(pad * ratio)), int(round((y_bot - y_top + pad) * ratio)))

    def x_full(self, cols: np.ndarray | float, rows: np.ndarray | None = None) -> np.ndarray:
        """Абсциссы полного кадра для столбцов ленты ``cols`` в строках ``rows`` (все строки — None)."""
        ys = self.ys_seg if rows is None else self.ys_seg[rows]
        return self.spine.x_at(ys) * self.ratio + (np.asarray(cols, np.float64) - self.half)

    def warp(self, img: np.ndarray, seg: bool, nearest: bool = False) -> np.ndarray:
        """Лента из кадра (``seg`` — картинка масштаба сегментации, иначе полного разрешения; bool — как bool)."""
        cols = np.arange(self.width, dtype=np.float64)
        map_x = self.spine.x_at(self.ys_seg)[:, None] * self.ratio + (cols[None, :] - self.half)
        map_y = np.repeat(self.ys_full[:, None], self.width, axis=1)
        if seg:
            map_x, map_y = map_x / self.ratio, map_y / self.ratio
        src = img.astype(np.uint8) if img.dtype == bool else img
        out = cv2.remap(src, map_x.astype(np.float32), map_y.astype(np.float32),
                        cv2.INTER_NEAREST if nearest or img.dtype == bool else cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # fmt: skip
        return out > 0 if img.dtype == bool else out

    def col(self, u_mm: float) -> int:
        """Столбец ленты на ``u_mm`` от разделителя (минус — влево)."""
        return int(round(self.half + u_mm * self.ppm))

    def u_mm(self, cols: np.ndarray | float) -> np.ndarray | float:
        """Сдвиг столбцов от разделителя, мм."""
        return (np.asarray(cols, np.float64) - self.half) / self.ppm

    def seg_points(self, edge: np.ndarray) -> np.ndarray:
        """Точки края (столбцы по строкам, NaN — пропуск) в кадре сегментации ``(N, 2)`` ``x, y``."""
        ok = np.isfinite(edge)
        rows = np.flatnonzero(ok)
        xs = self.x_full(edge[ok], rows) / self.ratio
        return np.stack([xs, self.ys_seg[rows]], axis=1)


@dataclass
class EdgeMeasures:
    """Меры одного края способа.

    Поля:
        cover: Доля годных строк с краем.
        rough_mm: p90 отклонения от скользящей медианы, мм.
        step_mm: Наибольшая ступень, мм.
        straight_mm: Не меряется (NaN): кромки полосы — любые гладкие кривые.
        ink_out_mm2: Краска своей страницы за краем, мм².
        median_mm: Медиана положения края от разделителя, мм.
        inlier_share: Доля строк с краем, попавших в гладкие куски без объездов.
        support_l: Опора края на снимке — отклик линии вдоль него, уровни L (``support_of``).
        breaks: Разрывов между гладкими кусками края (``edge_pieces``).
        accepted: Прошёл приёмку.
    """

    cover: float = 0.0
    rough_mm: float = float("nan")
    step_mm: float = float("nan")
    straight_mm: float = float("nan")
    ink_out_mm2: float = float("nan")
    median_mm: float = float("nan")
    inlier_share: float = 0.0
    support_l: float = float("nan")
    breaks: int = 0
    accepted: bool = False


@dataclass
class MethodResult:
    """Края способа по строкам ленты (столбцы, NaN — нет) и их меры.

    Поля:
        method: Способ.
        left: Правый край левой страницы по строкам (первый столбец вне неё).
        right: Левый край правой страницы (первый её столбец).
        left_m: Меры левого края.
        left_clean: Левый край без выбросов от его прямой (NaN), для выбора; None — не мерился.
        right_clean: То же правый.
        left_line: Левый край на всю высоту ленты (``complete``): пропуски и концы для итога.
        right_line: То же правый.
        right_m: Меры правого.
        missed_chunks: Кусков с промахом.
        order_ok: Края не заходят друг за друга (``ORDER_TOL_MM`` в ``ORDER_SHARE`` строк).
        seconds: Время способа.
        note: Почему способ неприменим (пусто — применим).
    """

    method: Method
    left: np.ndarray
    right: np.ndarray
    left_m: EdgeMeasures = field(default_factory=EdgeMeasures)
    left_clean: np.ndarray | None = None
    right_clean: np.ndarray | None = None
    left_line: np.ndarray | None = None
    right_line: np.ndarray | None = None
    right_m: EdgeMeasures = field(default_factory=EdgeMeasures)
    missed_chunks: int = 0
    order_ok: bool = True
    seconds: float = 0.0
    note: str = ""


@dataclass
class SpineRefine:
    """Итог уточнения ленты кадра.

    Поля:
        band: Лента.
        coarse: Оценка щели грубого SAM 3 (``detect_sam3``).
        results: Имя способа → итог.
        left: Выбранный край левой страницы (столбцы ленты по строкам, сглажен) или None.
        right: Выбранный край правой.
        left_by: Способ левого края.
        right_by: Способ правого.
        full_set: Понадобился полный набор.
        width_mm: Медиана ширины между выбранными краями по годным строкам, мм (NaN — края не выбраны).
        strip: Решение «полоса есть».
        status: Почему края не выбраны.
    """

    band: SpineBand
    coarse: Estimate
    results: dict[str, MethodResult]
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    left_by: str = ""
    right_by: str = ""
    full_set: bool = False
    width_mm: float = float("nan")
    strip: bool = False
    status: str = ""


@dataclass(frozen=True)
class BandContext:
    """Всё о ленте кадра, что нужно способам и приёмке.

    Поля:
        band: Геометрия.
        bgr: Лента кадра полного разрешения.
        coarse_left: Грубая страница слева (итоговые страницы v2) по ленте.
        coarse_right: То же справа.
        ink: Глифы краски по ленте.
        valid: Годные строки для мер (обе грубые страницы есть, без концов и пальца).
        gap_cols: Середина щели грубого SAM 3 по строкам ленты (столбцы, NaN — щели нет).
        seg_band: Лента стенда масштаба сегментации (для способов без сети).
        raw_left: Экземпляр грубого SAM 3 левой страницы по ленте (с языками, без склейки v2).
        raw_right: То же правой.
        response: Отклик линии на снимке по ленте, уровни L (``line_response``).
    """

    band: SpineBand
    bgr: np.ndarray
    coarse_left: np.ndarray
    coarse_right: np.ndarray
    ink: np.ndarray
    valid: np.ndarray
    gap_cols: np.ndarray
    seg_band: Band
    raw_left: np.ndarray
    raw_right: np.ndarray
    response: np.ndarray


def line_response(bgr: np.ndarray, ppm: float) -> np.ndarray:
    """Отклик вертикальной линии по ленте: наибольшее из ступени светлоты бумаги без букв и глубины тонкой складки.

    Ступень — модуль разности через ``RESP_STEP_MM`` вправо и влево светлоты бумаги без букв (максимум L по вертикали в
    3 мм); складка — насколько средняя по вертикали в 3 мм светлота темнее обеих сторон на ``RESP_CREASE_MM``; затем
    сглаживание вдоль корешка.

    Args:
        bgr: Лента полного разрешения.
        ppm: px/мм.

    Returns:
        Отклик, float, уровни L 0–255.
    """
    raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
    win = max(3, int(round(3.0 * ppm)) | 1)
    light = cv2.dilate(raw, np.ones((win, 1), np.uint8)).astype(np.float32)
    d = max(1, int(round(RESP_STEP_MM * ppm)))
    step = np.zeros_like(light)
    step[:, d:-d] = np.abs(light[:, 2 * d :] - light[:, : -2 * d])
    # Складка — по среднему вдоль корешка: максимум стёр бы пунктирную кромку листа (ПХ 1975/04 IMG_0083, +16 мм).
    mean = cv2.blur(raw.astype(np.float32), (1, win))
    c = max(1, int(round(RESP_CREASE_MM * ppm)))
    crease = np.minimum(np.roll(mean, c, axis=1), np.roll(mean, -c, axis=1)) - mean
    crease[:, :c] = 0
    crease[:, -c:] = 0
    k = max(3, int(round(RESP_ALONG_MM * ppm)) | 1)
    return cv2.blur(np.maximum(step, np.maximum(crease, 0)), (1, k))


def support_of(edge: np.ndarray, ctx: "BandContext") -> float:
    """Опора края на снимке: медиана по годным строкам наибольшего отклика линии в ``SUPPORT_HALF_MM`` от края."""
    band = ctx.band
    rows = np.flatnonzero(ctx.valid & np.isfinite(edge))
    if len(rows) < 50:
        return float("nan")
    r = max(1, int(round(SUPPORT_HALF_MM * band.ppm)))
    cols = np.clip(np.round(edge[rows]).astype(int)[:, None] + np.arange(-r, r + 1)[None, :], 0, band.width - 1)
    return float(np.median(ctx.response[rows[:, None], cols].max(axis=1)))


def stretch_band_levels(piece: np.ndarray) -> np.ndarray:
    """Растяжение светлоты куска по бумаге (краска отсечена Оцу; 1-й и 99-й процентили бумаги ± 8 уровней)."""
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = gray[gray > thr]
    if paper.size < 100:
        return piece
    lo, hi = np.percentile(paper, (1.0, 99.0))
    white = min(255.0, float(hi) + 8.0)
    black = max(0.0, min(float(lo) - 8.0, white - 255.0 / 4.0))
    lut = np.clip((np.arange(256) - black) * 255.0 / (white - black), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run_end(mask: np.ndarray, start: int, step: int) -> np.ndarray:
    """Конец сплошного участка маски по строкам от столбца ``start`` в сторону ``step`` (+1 — вправо, −1 — влево).

    Args:
        mask: Маска ленты.
        start: Столбец, где маска должна быть (иначе NaN).
        step: Направление.

    Returns:
        По строкам: первый столбец вне маски при движении вправо (у левой страницы — её правый край) или последний
        столбец маски при движении влево (у правой — её левый край); NaN — в ``start`` не маска.
    """
    h, w = mask.shape
    out = np.full(h, np.nan)
    inside = mask[:, start]
    seg = mask[:, start:] if step > 0 else mask[:, : start + 1][:, ::-1]
    first_out = np.argmax(~seg, axis=1)
    has_out = (~seg).any(axis=1)
    n = seg.shape[1]
    run = np.where(has_out, first_out, n)
    if step > 0:
        out[inside] = (start + run)[inside]
    else:
        out[inside] = (start - run + 1)[inside]
    return out


def band_ink(bgr: np.ndarray, ppm: float) -> np.ndarray:
    """Глифы краски по ленте: темнее бумаги на ``INK_DROP`` уровней L, серые, размером с букву."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[..., 0] * (100.0 / 255.0)
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    paper = float(np.median(light[light > np.percentile(light, 30)]))
    dark = (light < paper - INK_DROP * 100.0 / 255.0) & (chroma < INK_CHROMA)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(dark.shape, bool)
    h_mm, w_mm = stats[:, cv2.CC_STAT_HEIGHT] / ppm, stats[:, cv2.CC_STAT_WIDTH] / ppm
    area = stats[:, cv2.CC_STAT_AREA] / ppm**2
    keep = (h_mm <= GLYPH_MAX_H_MM) & (w_mm <= GLYPH_MAX_W_MM) & (area >= GLYPH_MIN_MM2)
    keep[0] = False
    return keep[lab_cc]


def band_context(sf: StripFrame, full: np.ndarray) -> tuple[BandContext, Estimate]:
    """Лента кадра и всё для способов: грубые страницы, краска, годные строки, щель грубого SAM 3.

    Args:
        sf: Кадр стенда (кэш v2).
        full: Исходный кадр полного разрешения, BGR.

    Returns:
        ``(контекст, оценка щели грубого SAM 3)``.
    """
    img = sf.frame.img
    ratio = full.shape[1] / img.shape[1]
    spine = sf.spine0
    pages = sf.frame.pages
    y_top = float(min(p.quad[:, 1].min() for p in pages))
    y_bot = float(max(p.quad[:, 1].max() for p in pages))
    band = SpineBand(spine, y_top, y_bot, ratio)
    bgr = band.warp(full, seg=False)
    left = band.warp(pages[0].mask, seg=True)
    right = band.warp(pages[1].mask, seg=True)
    core = band.warp(sf.core, seg=True)
    near = np.zeros(band.width, bool)
    near[max(band.col(-FINGER_NEAR_MM), 0) : band.col(FINGER_NEAR_MM) + 1] = True
    finger = (core & near[None, :]).any(axis=1)
    valid = left.any(axis=1) & right.any(axis=1) & ~finger
    lo, hi = band.page_rows
    span = hi - lo
    valid[: lo + int(END_SHARE * span)] = False
    valid[hi - int(END_SHARE * span) :] = False
    # Щель грубого SAM 3 и её середина по строкам ленты стенда (масштаб сегментации) → строки этой ленты.
    seg_band = build_band(sf)
    coarse = detect_sam3(seg_band)
    gap_cols = np.full(band.length, np.nan)
    if np.isfinite(coarse.left_u) and np.isfinite(coarse.right_u) and coarse.extra.get("gap_mm", 0.0) >= GAP_MIN_MM:
        mid_u_mm = (coarse.left_u + coarse.right_u) / 2 / SEG_PX_PER_MM
        gap_cols[:] = band.col(mid_u_mm)
    ink = band_ink(bgr, band.ppm)
    # Экземпляры грубого SAM 3 страниц — самые частые метки у точек страниц (столбцы ±``PAGE_POS_MM``).
    inst = band.warp(sf.inst.astype(np.int32).astype(np.float32), seg=True, nearest=True).astype(np.int32)
    raws = []
    for c in (band.col(-PAGE_POS_MM), band.col(PAGE_POS_MM)):
        v = inst[:, c][inst[:, c] > 0]
        lab = int(np.bincount(v).argmax()) if len(v) else -1
        raws.append(inst == lab)
    response = line_response(bgr, band.ppm)
    return BandContext(band, bgr, left, right, ink, valid, gap_cols, seg_band, raws[0], raws[1], response), coarse


class SpineRefiner:
    """Сети уточнения ленты: предиктор SAM 3 и SAM 2.1 — те же, что у прогона v2 (грузятся один раз)."""

    def __init__(self, sam3_predictor, sam2) -> None:
        """Args:
        sam3_predictor: ``SAM3SemanticPredictor`` (``paper.load_sam3``).
        sam2: ``bg_veto.Sam2Background`` (метод ``masks_scored``).
        """
        self.sam3 = sam3_predictor
        self.sam2 = sam2

    def sam3_instances(self, bgr: np.ndarray, prompts: tuple[str, ...]) -> list[np.ndarray]:
        """Все экземпляры SAM 3 по подсказкам ``prompts`` (маски размера куска)."""
        import torch

        h, w = bgr.shape[:2]
        out = []
        with torch.inference_mode():
            self.sam3.set_image(bgr)
            for prompt in prompts:
                res = self.sam3(text=[prompt])
                if not res or res[0].masks is None:
                    continue
                for m in res[0].masks.data.cpu().numpy() > 0.5:
                    out.append(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
        if hasattr(self.sam3, "reset_image"):
            self.sam3.reset_image()
        return out

    def sam2_mask(self, bgr: np.ndarray, pos: list[int], neg: list[int]) -> np.ndarray:
        """Маска SAM 2.1 по точкам: столбцы ``pos`` положительные, ``neg`` отрицательные, по три строки на столбец.

        Из трёх масок — с лучшей оценкой, накрывающая все положительные и ни одной отрицательной (пусто — ни одна).
        """
        h = bgr.shape[0]
        rows = [h * f for f in (0.15, 0.5, 0.85)]
        pts = [[float(c), y] for c in pos for y in rows] + [[float(c), y] for c in neg for y in rows]
        labels = [1] * (len(pos) * 3) + [0] * (len(neg) * 3)
        masks, scores = self.sam2.masks_scored(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), pts, labels)
        best, best_s = np.zeros(bgr.shape[:2], bool), -1.0
        for m, s in zip(masks, scores):
            good = all(m[int(y), c] for c in pos for y in rows) and not any(m[int(y), c] for c in neg for y in rows)
            if good and s > best_s:
                best, best_s = m, float(s)
        return best

    def chunk_edges(
        self, piece: np.ndarray, method: Method, cols: dict[str, int], gap: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Края куска: правый край левой страницы и левый край правой по строкам куска (NaN — нет).

        Args:
            piece: Кусок ленты BGR (уже сжатый и подготовленный).
            method: Способ.
            cols: Столбцы ``left`` / ``right`` (точки страниц) и ``mid`` (разделитель).
            gap: Столбец середины щели грубого SAM 3 или None.

        Returns:
            ``(левый край, правый край)``.
        """
        lp, rp = cols["left"], cols["right"]
        if method.engine is Engine.SAM2:
            neg_l = [rp] + ([gap] if gap is not None else [])
            neg_r = [lp] + ([gap] if gap is not None else [])
            ml = ndi.binary_fill_holes(self.sam2_mask(piece, [lp], neg_l))
            mr = ndi.binary_fill_holes(self.sam2_mask(piece, [rp], neg_r))
            return run_end(ml, lp, 1), run_end(mr, rp, -1)
        if method.engine is Engine.SAM3:
            inst = self.sam3_instances(piece, SAM3_PROMPTS)
            if not inst:
                return np.full(len(piece), np.nan), np.full(len(piece), np.nan)
            cov_l = [float(m[:, lp].mean()) for m in inst]
            cov_r = [float(m[:, rp].mean()) for m in inst]
            il, ir = int(np.argmax(cov_l)), int(np.argmax(cov_r))
            if il == ir or cov_l[il] < 0.5 or cov_r[ir] < 0.5:
                # Один экземпляр на обе страницы — разреза нет.
                return np.full(len(piece), np.nan), np.full(len(piece), np.nan)
            ml, mr = (ndi.binary_fill_holes(inst[i] & ~inst[j]) for i, j in ((il, ir), (ir, il)))
            return run_end(ml, lp, 1), run_end(mr, rp, -1)
        # Полоса как класс: её левая и правая граница — края страниц.
        if gap is None:
            return np.full(len(piece), np.nan), np.full(len(piece), np.nan)
        if method.engine is Engine.SAM2_STRIP:
            ms = self.sam2_mask(piece, [gap], [lp, rp])
        else:
            cands = [
                m for m in self.sam3_instances(piece, STRIP_PROMPTS) if m[:, lp].mean() < 0.5 and m[:, rp].mean() < 0.5
            ]
            ms = max(cands, key=lambda m: m[:, gap].mean()) if cands else np.zeros(piece.shape[:2], bool)
        ms = ndi.binary_fill_holes(ms)
        # От середины щели влево до конца полосы — край левой страницы; вправо — край правой.
        return run_end(ms, gap, -1), run_end(ms, gap, 1)

    def method_edges(self, ctx: BandContext, method: Method) -> tuple[np.ndarray, np.ndarray, int]:
        """Края способа по всей ленте: куски с перекрытием, в перекрытии — кусок с ближним центром.

        Промах куска (его строки без края): обе маски пусты или края нет больше чем в половине строк куска.

        Args:
            ctx: Лента кадра.
            method: Способ.

        Returns:
            ``(левый край, правый край, кусков с промахом)``.
        """
        band = ctx.band
        if method.engine is Engine.SOLID:
            return solid_edges(ctx) + (0,)
        if not method.neural:
            return self.line_edges(ctx, method) + (0,)
        h = band.length
        size = int(CHUNK_MM * method.squeeze * band.ppm)
        step = int((CHUNK_MM - OVERLAP_MM) * method.squeeze * band.ppm)
        starts = list(range(0, max(h - size, 0) + 1, max(step, 1)))
        if starts[-1] + size < h:
            starts.append(max(h - size, 0))
        cols = {"left": band.col(-PAGE_POS_MM), "right": band.col(PAGE_POS_MM), "mid": band.col(0.0)}
        src = clahe(ctx.bgr, band.ppm) if method.prep is Prep.CLAHE else ctx.bgr
        left, right = np.full(h, np.nan), np.full(h, np.nan)
        dist, missed = np.full(h, np.inf), 0
        for y0 in starts:
            y1 = min(y0 + size, h)
            rows = np.arange(y0, y1)
            closer = np.abs(rows - (y0 + y1) / 2) < dist[rows]
            piece = np.ascontiguousarray(src[y0:y1])
            if method.prep is Prep.LEVELS:
                piece = stretch_band_levels(piece)
            ph, pw = piece.shape[:2]
            if method.squeeze > 1:
                piece = cv2.resize(piece, (pw, max(ph // method.squeeze, 8)), interpolation=cv2.INTER_AREA)
            gaps = ctx.gap_cols[y0:y1]
            gap = int(np.nanmedian(gaps)) if np.isfinite(gaps).any() else None
            el, er = self.chunk_edges(piece, method, cols, gap)
            if method.squeeze > 1:
                # Края по строкам сжатого куска → строки куска полного размера.
                src_rows = np.minimum(np.arange(ph) // method.squeeze, len(el) - 1)
                el, er = el[src_rows], er[src_rows]
            if np.isnan(el).mean() > 0.5 and np.isnan(er).mean() > 0.5:
                missed += 1
            left[rows[closer]] = el[closer]
            right[rows[closer]] = er[closer]
            dist[rows[closer]] = np.abs(rows[closer] - (y0 + y1) / 2)
        return left, right, missed

    def line_edges(self, ctx: BandContext, method: Method) -> tuple[np.ndarray, np.ndarray]:
        """Края способа без сети: пара прямых ступеней (``detect_edges``) или складок (``detect_creases``) по ленте
        масштаба сегментации → прямые по строкам этой ленты; пара видна меньше чем в ``LINES_MIN_SHARE`` строк — NaN."""
        band, sb = ctx.band, ctx.seg_band
        est = detect_edges(sb) if method.engine is Engine.EDGES else detect_creases(sb)
        nan = np.full(band.length, np.nan)
        if not np.isfinite(est.score) or est.score < LINES_MIN_SHARE:
            return nan, nan.copy()
        tan = np.tan(np.radians(est.extra.get("shear_deg", 0.0)))
        # Строка ленты стенда для каждой строки этой ленты; прямая u(r) = u_центра + tan·(r − h/2), px сегментации.
        r = np.interp(band.ys_seg, sb.ys, np.arange(len(sb.ys)))
        inside = (band.ys_seg >= sb.ys[0]) & (band.ys_seg <= sb.ys[-1])
        out = []
        for u0 in (est.left_u, est.right_u):
            u_seg = u0 + tan * (r - len(sb.ys) / 2)
            cols = band.half + u_seg * band.ratio
            out.append(np.where(inside, cols, np.nan))
        return out[0], out[1]


def best_line(ctx: BandContext, lo: int, hi: int) -> tuple[np.ndarray, float]:
    """Сильнейшая наклонная прямая линия на снимке между столбцами ``lo`` и ``hi``: перебор наклона (``LINE_TILTS_DEG``) и
    сдвига, балл — медиана отклика линии вдоль прямой по годным строкам.

    Args:
        ctx: Лента кадра.
        lo: Левый столбец окна (на середине годных строк).
        hi: Правый.

    Returns:
        ``(столбцы прямой по всем строкам ленты, балл — уровни L)``; окно пусто — NaN и NaN.
    """
    band = ctx.band
    rows = np.flatnonzero(ctx.valid)
    lo, hi = max(lo, 0), min(hi, band.width - 1)
    if len(rows) < 50 or hi <= lo:
        return np.full(band.length, np.nan), float("nan")
    mid = float(np.median(rows))
    sub = rows[:: max(1, len(rows) // 600)]
    best, best_v = (lo, 0.0), -1.0
    for deg in LINE_TILTS_DEG:
        k = np.tan(np.radians(deg))
        shift = np.round(k * (sub - mid)).astype(int)
        for c in range(lo, hi + 1):
            cols = np.clip(c + shift, 0, band.width - 1)
            v = float(np.median(ctx.response[sub, cols]))
            if v > best_v:
                best, best_v = (c, k), v
    c, k = best
    return c + k * (np.arange(band.length) - mid), best_v


def solid_edges(ctx: BandContext) -> tuple[np.ndarray, np.ndarray]:
    """Способ ``solid``: сплошные кромки страниц грубого SAM 3, притянутые к линиям на снимке.

    Кромка страницы по строкам — конец её экземпляра от точки страницы к корешку (с языками). Сплошная часть — строки,
    где кромка ближе всего к своей странице (доля ``SOLID_SHARE``): по ним прямая RANSAC (``line_fit``); затем сдвиг
    прямой в ``±SOLID_SNAP_MM`` к наибольшему среднему отклику линии вдоль неё.

    Args:
        ctx: Лента кадра.

    Returns:
        ``(левый край, правый край)`` — прямые по всем строкам ленты (NaN — кромка не построилась).
    """
    band = ctx.band
    out = []
    rows_all = np.arange(band.length)
    for raw, page_left in ((ctx.raw_left, True), (ctx.raw_right, False)):
        pos = band.col(-PAGE_POS_MM if page_left else PAGE_POS_MM)
        e = run_end(raw, pos, 1 if page_left else -1)
        e = np.where(ctx.valid, e, np.nan)
        nan = np.full(band.length, np.nan)
        if np.isfinite(e).sum() < 50:
            out.append(nan)
            continue
        # Сплошная часть: у левой — наименьшие кромки (ближе к своей странице), у правой — наибольшие.
        q = np.nanquantile(e, SOLID_SHARE if page_left else 1.0 - SOLID_SHARE)
        solid = (e <= q) if page_left else (e >= q)
        on, share, _ = line_fit(np.where(solid, e, np.nan), ctx.valid, band)
        if not on.any():
            out.append(nan)
            continue
        coef = np.polyfit(rows_all[on], e[on], 1)
        centre = float(np.polyval(coef, np.median(np.flatnonzero(ctx.valid))))
        back, ahead = SOLID_BACK_MM * band.ppm, SOLID_AHEAD_MM * band.ppm
        # Окно поиска линии: к корешку — ``SOLID_BACK_MM``, к своей странице — ``SOLID_AHEAD_MM``.
        lo, hi = (centre - ahead, centre + back) if page_left else (centre - back, centre + ahead)
        line, _ = best_line(ctx, int(lo), int(hi))
        out.append(line)
    return out[0], out[1]


def line_fit(edge: np.ndarray, rows_ok: np.ndarray, band: SpineBand) -> tuple[np.ndarray, float, float]:
    """Прямая края RANSAC по годным строкам без ``STRAIGHT_END_MM`` у концов пролёта.

    Пары строк выбираются случайно (``RANSAC_TRIALS``, затравка 0); лучшая — с наибольшим числом строк ближе
    ``LINE_TOL_MM``; по ним — МНК.

    Args:
        edge: Край по строкам (столбцы ленты, NaN — нет).
        rows_ok: Годные строки.
        band: Лента.

    Returns:
        ``(строки на прямой — bool по всем строкам, доля их среди годных строк с краем, p90 их остатка, мм)``;
        строк мало — пустая маска, 0 и NaN.
    """
    lo, hi = band.page_rows
    cut = int(STRAIGHT_END_MM * band.ppm)
    rows = np.flatnonzero(rows_ok & np.isfinite(edge))
    rows = rows[(rows >= lo + cut) & (rows <= hi - cut)]
    on = np.zeros(len(edge), bool)
    if len(rows) < 50:
        return on, 0.0, float("nan")
    tol = LINE_TOL_MM * band.ppm
    x = edge[rows]
    rng = np.random.default_rng(0)
    best, best_n = None, -1
    for _ in range(RANSAC_TRIALS):
        i, j = rng.choice(len(rows), 2, replace=False)
        if rows[i] == rows[j]:
            continue
        k = (x[j] - x[i]) / (rows[j] - rows[i])
        n = int((np.abs(x - (x[i] + k * (rows - rows[i]))) <= tol).sum())
        if n > best_n:
            best, best_n = (x[i], k, rows[i]), n
    x0, k, r0 = best
    near = np.abs(x - (x0 + k * (rows - r0))) <= tol
    coef = np.polyfit(rows[near], x[near], 1)
    resid = np.abs(x - np.polyval(coef, rows))
    near = resid <= tol
    on[rows[near]] = True
    return on, float(near.mean()), float(np.percentile(resid[near], 90) / band.ppm)


def edge_pieces(edge: np.ndarray, ppm: float) -> list[tuple[int, int]]:
    """Гладкие куски края: рвётся на скачке больше ``JUMP_MM`` и на пропуске длиннее ``JUMP_SPAN_MM``; куски короче
    ``PIECE_MIN_MM`` отбрасываются.

    Скачок в строке — разность медиан окна ``JUMP_SPAN_MM`` после и до неё (ступеньки сжатого куска в 1–3 px через
    4 строки — не скачок); разрез — в локальном максимуме скачка. Короткие пропуски остаются внутри куска.

    Args:
        edge: Край по строкам (столбцы, NaN — нет).
        ppm: px/мм.

    Returns:
        Куски — пары строк ``(начало, конец)``, конец не включается; в куске могут быть короткие пропуски.
    """
    ok = np.isfinite(edge)
    if not ok.any():
        return []
    n = len(edge)
    span = max(2, int(round(JUMP_SPAN_MM * ppm)))
    idx = np.arange(n)
    u = np.interp(idx, idx[ok], edge[ok])
    # Медиана окна до строки (строки i − span … i − 1) и после (i … i + span − 1).
    pad = np.pad(u, span, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, span)
    med = np.median(windows, axis=1)
    before, after = med[:n], med[span : span + n]
    jump = np.abs(after - before)
    peak = (jump > JUMP_MM * ppm) & (jump == ndi.maximum_filter(jump, size=2 * span + 1, mode="nearest"))
    # Длинный пропуск — разрыв: строки пропуска длиннее span выпадают из кусков.
    gap_len = np.zeros(n, int)
    run = 0
    for i in range(n):
        run = run + 1 if not ok[i] else 0
        gap_len[i] = run
    long_gap = np.zeros(n, bool)
    for i in np.flatnonzero((gap_len > 0) & np.r_[gap_len[1:] == 0, True]):
        if gap_len[i] > span:
            long_gap[i - gap_len[i] + 1 : i + 1] = True
    pieces, start = [], None
    for i in range(n + 1):
        usable = i < n and not long_gap[i]
        if start is None:
            start = i if usable and ok[i] else None
        elif not usable or peak[i]:
            pieces.append((start, i))
            start = i if usable and ok[i] else None
    # Конец куска — последняя строка с краем.
    out = []
    for a, b in pieces:
        rows = np.flatnonzero(ok[a:b])
        if len(rows) and rows[-1] + 1 - rows[0] >= PIECE_MIN_MM * ppm:
            out.append((a + int(rows[0]), a + int(rows[-1]) + 1))
    return out


def drop_detours(edge: np.ndarray, pieces: list[tuple[int, int]], page_left: bool, ppm: float) -> list[tuple[int, int]]:
    """Куски без объездов: кусок, оба соседа которого ближе к своей странице дальше ``DETOUR_MM``, а сами соседи
    сходятся между собой (прямая между их ближними концами проходит в ``DETOUR_MM`` от обоих концов)."""
    sign = 1.0 if page_left else -1.0  # ближе к своей странице: у левой — меньше, у правой — больше
    keep = list(pieces)
    changed = True
    while changed and len(keep) >= 3:
        changed = False
        for k in range(1, len(keep) - 1):
            (a0, b0), (a1, b1), (a2, b2) = keep[k - 1], keep[k], keep[k + 1]
            left_end = float(np.nanmedian(edge[max(a0, b0 - int(5 * ppm)) : b0]))
            right_start = float(np.nanmedian(edge[a2 : min(b2, a2 + int(5 * ppm))]))
            mid = float(np.nanmedian(edge[a1:b1]))
            bridge = left_end + (right_start - left_end) * 0.5
            ahead = sign * (mid - bridge) > DETOUR_MM * ppm  # средний — дальше от своей страницы (к корешку)
            if ahead and abs(left_end - right_start) <= DETOUR_MM * ppm + abs(a2 - b0) * 0.05:
                del keep[k]
                changed = True
                break
    return keep


def complete(edge: np.ndarray, length: int) -> np.ndarray:
    """Край на всю высоту ленты: пропуски — линейной интерполяцией между строками с краем, концы — крайним значением."""
    ok = np.isfinite(edge)
    idx = np.arange(length)
    return np.interp(idx, idx[ok], edge[ok]) if ok.any() else np.full(length, np.nan)


def judge_edge(
    edge: np.ndarray, ctx: BandContext, own: np.ndarray, page_left: bool
) -> tuple[EdgeMeasures, np.ndarray, np.ndarray]:
    """Меры и приёмка края одной страницы.

    Args:
        edge: Край по строкам (столбцы ленты).
        ctx: Лента кадра.
        own: Грубая маска своей страницы по ленте (для краски за краем).
        page_left: Край левой страницы (её бумага — левее края).

    Край режется на гладкие куски (``edge_pieces``), объезды выбрасываются (``drop_detours``); строки вне кусков —
    выбросы (пропуски). Приёмка — доля строк в кусках, рваность внутри кусков, краска за краем; скачок между кусками
    разрешён (с него начинается полоса на части длины), прямизна не требуется — кромки полосы любые гладкие.

    Returns:
        ``(меры с решением о приёмке, край без выбросов, он же на всю высоту ленты — пропуски интерполяцией)``.
    """
    band, valid = ctx.band, ctx.valid
    e = np.where(valid, edge, np.nan)
    m = EdgeMeasures(cover=float(np.isfinite(e[valid]).mean()) if valid.any() else 0.0)
    nan_line = np.full(len(e), np.nan)
    if m.cover < 0.5:
        return m, e, nan_line
    # Гладкие куски без объездов; остальные строки — выбросы (пропуски).
    pieces = drop_detours(e, edge_pieces(e, band.ppm), page_left, band.ppm)
    keep = np.zeros(len(e), bool)
    for a, b in pieces:
        keep[a:b] = True
    total = max(int(np.isfinite(e).sum()), 1)
    e = np.where(keep, e, np.nan)
    m.inlier_share, m.breaks = float(np.isfinite(e).sum() / total), max(len(pieces) - 1, 0)
    rough = [roughness_mm(np.where((np.arange(len(e)) >= a) & (np.arange(len(e)) < b), e, np.nan), band.ppm)
             for a, b in pieces if b - a >= 50]  # fmt: skip
    m.rough_mm = float(np.nanmax(rough)) if rough else float("nan")
    # Ступень внутри кусков по построению не больше JUMP_MM; меряется для сводки.
    m.step_mm = float(np.nanmax([step_mm(np.where((np.arange(len(e)) >= a) & (np.arange(len(e)) < b), e, np.nan),
                                         band.ppm) for a, b in pieces if b - a >= 50] or [np.nan]))  # fmt: skip
    m.straight_mm = float("nan")
    m.median_mm = float(band.u_mm(np.nanmedian(e)))
    m.support_l = support_of(e, ctx)
    cols = np.arange(band.width)[None, :]
    # Краска своей страницы за краем: у левой — правее края, у правой — левее (в пределах грубой страницы).
    far = INK_FROM_EDGE_MM * band.ppm
    beyond = cols >= e[:, None] + far if page_left else cols < e[:, None] - far
    beyond &= np.isfinite(e)[:, None]
    m.ink_out_mm2 = float((ctx.ink & own & beyond).sum() / band.ppm**2)
    m.accepted = bool(
        m.cover >= MIN_COVER
        and m.inlier_share >= LINE_MIN_SHARE
        and np.isfinite(m.rough_mm)
        and m.rough_mm <= RAGGED_OK_MM
        and m.ink_out_mm2 <= INK_OUT_MM2
    )
    return m, e, complete(e, len(e))


def judge(res: MethodResult, ctx: BandContext) -> MethodResult:
    """Приёмка обоих краёв способа и их порядка (левый край не правее правого глубже ``ORDER_TOL_MM``)."""
    res.left_m, res.left_clean, res.left_line = judge_edge(res.left, ctx, ctx.coarse_left, page_left=True)
    res.right_m, res.right_clean, res.right_line = judge_edge(res.right, ctx, ctx.coarse_right, page_left=False)
    lc, rc = res.left_clean, res.right_clean
    both = ctx.valid & np.isfinite(lc) & np.isfinite(rc)
    if both.any():
        res.order_ok = bool(((rc[both] - lc[both]) >= -ORDER_TOL_MM * ctx.band.ppm).mean() >= ORDER_SHARE)
    if not res.order_ok:
        res.left_m.accepted = res.right_m.accepted = False
    return res


def choose_edge(results: list[MethodResult], page_left: bool, ppm: float) -> MethodResult | None:
    """Выбор края одной страницы среди принятых способов.

    Общий край — медиана принятых сетей по строкам; согласные — не дальше ``AGREE_MM`` от него; среди согласных —
    наименьшая рваность (+0.05 мм), из них — ближний к своей странице (кромка листа, а не кончик языка). Затем выбор
    вытесняет любой принятый край (и без сети) ближе к своей странице дальше ``AGREE_MM`` с опорой не слабее
    ``SUPPORT_STRONG_L`` и краской за ним не больше ``OVERRIDE_INK_MM2`` — ближний из таких. Без принятых сетей — край с
    сильнейшей опорой из таких.

    Args:
        results: Итоги способов.
        page_left: Край левой страницы.
        ppm: px/мм ленты.

    Returns:
        Способ края или None.
    """
    ok = [r for r in results if side_of(r, page_left)[1].accepted]
    sign = 1.0 if page_left else -1.0
    pos = {id(r): sign * float(np.nanmedian(side_of(r, page_left)[0])) for r in ok}
    # Выбор по согласию — только среди сетей: прямая без сети всегда «ровнее» и выиграла бы у любой сети.
    nets = [r for r in ok if r.method.neural]
    chosen = None
    if nets:
        consensus = np.nanmedian(np.stack([side_of(r, page_left)[0] for r in nets]), axis=0)
        agree = [
            r for r in nets if np.nanmedian(np.abs(side_of(r, page_left)[0] - consensus)) / ppm <= AGREE_MM
        ] or nets
        best = min(side_of(r, page_left)[1].rough_mm for r in agree)
        tied = [r for r in agree if side_of(r, page_left)[1].rough_mm <= best + 0.05]
        # Ближний к своей странице: у левой — наименьший медианный край, у правой — наибольший.
        chosen = min(tied, key=lambda r: pos[id(r)])
    # Сильная линия ближе к своей странице вытесняет выбор по согласию (кромка листа за белой полосой); без сетей —
    # только она.
    strong = [r for r in ok if side_of(r, page_left)[1].support_l >= SUPPORT_STRONG_L]
    if chosen is None:
        # Без принятых сетей — сильнейшая линия (не ближняя: граница тени загиба ближе к странице, чем сгиб).
        return max(strong, key=lambda r: side_of(r, page_left)[1].support_l) if strong else None
    ahead = [r for r in strong if pos[id(r)] < pos[id(chosen)] - AGREE_MM * ppm
             and side_of(r, page_left)[1].ink_out_mm2 <= OVERRIDE_INK_MM2]  # fmt: skip
    return min(ahead, key=lambda r: pos[id(r)]) if ahead else chosen


def side_of(res: MethodResult, page_left: bool) -> tuple[np.ndarray, EdgeMeasures]:
    """Край без выбросов и меры способа для левой (``page_left``) или правой страницы."""
    return (res.left_clean, res.left_m) if page_left else (res.right_clean, res.right_m)


def fill(edge: np.ndarray, line: np.ndarray | None, ppm: float) -> np.ndarray:
    """Край на всю высоту ленты: свои строки — как есть, пропуски и концы — по ``line`` (``complete``; нет его —
    линейной интерполяцией и продолжением крайней точки); медиана 1 мм."""
    ok = np.isfinite(edge)
    idx = np.arange(len(edge))
    if line is not None and np.isfinite(line).all():
        u = np.where(ok, edge, line)
    else:
        u = np.interp(idx, idx[ok], edge[ok])
    return ndi.median_filter(u, size=max(3, int(round(ppm)) | 1), mode="nearest")


def decide(out: SpineRefine, ctx: BandContext, full_set: bool) -> bool:
    """Выбор краёв по готовым итогам способов (``choose_edge``) и решение о полосе; True — оба края выбраны.

    Args:
        out: Итог кадра (поля краёв и решения заполняются здесь же — это его сборка, а не побочный выход).
        ctx: Лента кадра.
        full_set: Итоги включают полный набор.

    Returns:
        Выбраны ли оба края.
    """
    done = list(out.results.values())
    lb, rb = choose_edge(done, True, ctx.band.ppm), choose_edge(done, False, ctx.band.ppm)
    if lb is None or rb is None:
        return False
    out.left = fill(lb.left_clean, lb.left_line, ctx.band.ppm)
    out.right = fill(rb.right_clean, rb.right_line, ctx.band.ppm)
    out.left_by, out.right_by, out.full_set = lb.method.name, rb.method.name, full_set
    w = (out.right - out.left)[ctx.valid] / ctx.band.ppm
    out.width_mm = float(np.median(w)) if len(w) else float("nan")
    out.strip = bool(np.isfinite(out.width_mm) and out.width_mm >= STRIP_WIDTH_MM)
    out.status = ""
    return True


NO_EDGES = "края не выбраны: ни один способ не принят хотя бы для одной страницы"


def refine_spine(
    refiner: SpineRefiner, sf: StripFrame, full: np.ndarray, reduced_only: bool = False
) -> tuple[SpineRefine, BandContext]:
    """Уточнение ленты у корешка кадра: сокращённый набор, при неудаче — полный; выбор краёв и решение о полосе.

    Args:
        refiner: Сети.
        sf: Кадр стенда (кэш v2).
        full: Исходный кадр полного разрешения, BGR.
        reduced_only: Не идти в полный набор.

    Returns:
        ``(итог с краями всех способов, лента кадра)``.
    """
    ctx, coarse = band_context(sf, full)
    out = SpineRefine(ctx.band, coarse, {}, status=NO_EDGES)
    passes = [(False, REDUCED)] if reduced_only else [(False, REDUCED), (True, FULL)]
    for full_set, methods in passes:
        for method in methods:
            if method.name in out.results:
                continue
            t0 = time.monotonic()
            left, right, missed = refiner.method_edges(ctx, method)
            res = judge(MethodResult(method, left, right, missed_chunks=missed), ctx)
            res.seconds = time.monotonic() - t0
            out.results[method.name] = res
        if decide(out, ctx, full_set):
            break
    return out, ctx


def rechoose(sf: StripFrame, full: np.ndarray, edges: dict[str, np.ndarray]) -> tuple[SpineRefine, BandContext]:
    """Повторная приёмка и выбор по сохранённым краям способов (``npz`` команды ``refine``) — без сетей.

    Порядок наборов — как в ``refine_spine``: сокращённый, затем полный из того, что было посчитано.

    Args:
        sf: Кадр стенда.
        full: Исходный кадр полного разрешения, BGR.
        edges: ``{"<способ>_left": …, "<способ>_right": …}``.

    Returns:
        ``(итог, лента кадра)``.
    """
    ctx, coarse = band_context(sf, full)
    out = SpineRefine(ctx.band, coarse, {}, status=NO_EDGES)
    for full_set, methods in ((False, REDUCED), (True, FULL)):
        for method in methods:
            key = f"{method.name}_left"
            if method.name in out.results or key not in edges:
                continue
            res = MethodResult(method, edges[key].astype(np.float64), edges[f"{method.name}_right"].astype(np.float64))
            out.results[method.name] = judge(res, ctx)
        if decide(out, ctx, full_set):
            break
    return out, ctx


def summary(res: SpineRefine) -> dict:
    """Сводка итога для ``refine_spine.json``: решение, ширина, щель грубого SAM 3, меры всех способов."""
    return {
        "strip": res.strip,
        "width_mm": round(res.width_mm, 2) if np.isfinite(res.width_mm) else None,
        "left_by": res.left_by,
        "right_by": res.right_by,
        "full_set": res.full_set,
        "status": res.status,
        "coarse": {k: round(float(v), 3) for k, v in res.coarse.extra.items()},
        "methods": {
            name: {"left": measures_dict(r.left_m), "right": measures_dict(r.right_m), "order_ok": r.order_ok,
                   "missed_chunks": r.missed_chunks, "seconds": round(r.seconds, 2)}  # fmt: skip
            for name, r in res.results.items()
        },
    }


def measures_dict(m: EdgeMeasures) -> dict:
    """Меры края для JSON (числа округлены до 0.001)."""
    return {k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else (None if isinstance(v, float) else v))
            for k, v in m.__dict__.items()}  # fmt: skip


def source_path(src_root: Path, item: str) -> Path:
    """Исходный кадр пункта множества: ``01__IMG_0021`` → ``src_root/01/IMG_0021.jpg``, ``<выпуск>/IMG_0370`` →
    ``src_root/<выпуск>/IMG_0370.jpg`` (расширение — первое найденное из ``.jpg``, ``.JPG``, ``.jpeg``)."""
    rel = Path(item.replace("__", "/"))
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG"):
        path = src_root / rel.with_suffix(ext)
        if path.exists():
            return path
    raise FileNotFoundError(f"нет исходного кадра для {item} в {src_root}")
