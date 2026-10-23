"""Тонкое уточнение рваной внешней стороны страницы (``--refine-ragged``): лента вдоль дальней стороны трапеции в полном
разрешении, сегментация кусков ленты несколькими способами (SAM 3 по тексту, SAM 2.1 по точкам; сжатие куска по
вертикали, растяжение уровней светлоты, CLAHE), отбор способов с ровным краем, не режущим печать, и замена грубой маски
бумаги в ленте.

Стенд — ai_slop/stack_edges/refine.py (190 рваных и 20 ровных сторон ПХ 1975 и ХиП-1991, листы и закраски масок):
лучший — SAM 3 со сжатием ×4 без растяжения; SAM 2 со сжатием ×4 и растяжением по L — запасной (на форзаце с рисунком
SAM 3 не находит лист); SAM 2 со сжатием иногда уходит внутрь на кромку текста (ХиП IMG_0286 L) — от этого защита
печатью; CLAHE и растяжение трёх каналов хуже. Край страницы на веере листов — край верхнего листа (решение
пользователя 2026-10-22): из согласных способов берётся ближний к странице.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
import scipy.ndimage as ndi

from ocr_utils.scan_cropping.v2.page_sides import Side, inward_normal, quad_sides

# Лента: столько наружу и внутрь от дальней стороны трапеции, мм бумаги.
BAND_OUT_MM = 18.0
BAND_IN_MM = 25.0
# Кусок ленты по длине (до сжатия) и перекрытие соседних кусков, мм; со сжатием кусок длиннее во столько же раз —
# на входе сети та же форма около 50 × 43 мм.
CHUNK_MM = 50.0
OVERLAP_MM = 12.0
# Положительные точки SAM 2 и начало поиска края — на столько внутрь от стороны; отрицательные — наружу, мм.
POS_MM = 18.0
NEG_MM = -14.0
# Подсказки SAM 3 (как у грубой сегментации ``paper.SAM3_PROMPTS``).
SAM3_PROMPTS = ("book page", "paper page")
# CLAHE: клетка, мм, и предел.
CLAHE_TILE_MM = 20.0
CLAHE_CLIP = 3.0
# Растяжение уровней светлоты куска по бумаге страницы (столбцы от POS внутрь, краска — Оцу): запас от процентилей,
# процентили и предел усиления.
LEVEL_MARGIN = 8
PAPER_PCT = (1.0, 99.0)
MAX_GAIN = 4.0
# Приёмка способа: покрытие длины, наибольшая рваность (p90 отклонения края от медианы окна MED_MM), краска своей
# страницы за краем, мм². Краска — тёмные бесцветные пиксели ленты (темнее бумаги страницы на INK_DROP, цветность
# Lab меньше INK_CHROMA) внутри выпуклой оболочки своей печати; одна оболочка не годится — посторонняя метка у края
# тянет её треугольником до кромки листа (ПХ 1975/01 IMG_0029 L), цветной рисунок обложки — не краска текста.
INK_DROP = 70
INK_CHROMA = 12.0
# Пятно краски — знак, если его рамка не выше GLYPH_MAX_H_MM и не шире GLYPH_MAX_W_MM, а площадь не меньше
# GLYPH_MIN_MM2: щели между листами стопки и тени у кромки тоже тёмные и бесцветные, но тянутся вдоль стороны.
GLYPH_MAX_H_MM = 8.0
GLYPH_MAX_W_MM = 10.0
GLYPH_MIN_MM2 = 0.2
# Знаки считаются только не ближе этого внутрь от стороны трапеции, мм: за стороной и у неё — клочки торцов и рваной
# бумаги обложки размером со знак (ПХ 1975/03 IMG_0058 L — до 5 мм за стороной), а текст страницы стоит глубже.
INK_FROM_SIDE_MM = 3.0
MIN_COVER = 0.9
RAGGED_OK_MM = 0.3
MED_MM = 30.0
INK_OUT_MM2 = 5.0
# Ступень края: перепад медианы окна STEP_WIN_MM по обе стороны строки на расстоянии STEP_GAP_MM; больше STEP_MAX_MM —
# способ не принят (соседние куски ленты нашли края разных листов веера, ПХ 1975/09 IMG_0022 L, 01/IMG_0029 L —
# рваность по медиане 30 мм одну ступень не видит).
STEP_WIN_MM = 4.0
STEP_GAP_MM = 3.0
STEP_MAX_MM = 1.0
# Согласие: способ годен для выбора, если его край в пределах AGREE_MM от медианы принятых (по медиане строк).
AGREE_MM = 2.0
# Промах способа целиком: покрытие длины меньше этого.
MISS_COVER = 0.5
# Ядро пальца с запасом, мм: строки, где оно лежит у грубого края (в пределах FINGER_EDGE_MM поперёк), — без края
# (остальная зона закраса — в деле). Ладонь, держащая книгу снаружи, лежит в ленте за краем, кромку не закрывает и строк
# не выбрасывает (ХиП 1991/07 IMG_0133 R: ядро где угодно в ленте — 134 мм стороны, у края — 38 мм).
FINGER_CORE_GROW_MM = 3.0
FINGER_EDGE_MM = 3.0
# Сглаживание выбранного края по строкам, мм.
SMOOTH_MM = 1.0
# Доли длины стороны у каждого угла, где маска не заменяется.
END_SHARE = 0.05


class Engine(StrEnum):
    """Сеть способа."""

    SAM3 = "sam3"
    SAM2 = "sam2"


class Prep(StrEnum):
    """Подготовка куска перед сетью."""

    NONE = "none"
    LEVELS = "lvL"  # растяжение уровней светлоты по бумаге
    CLAHE = "clahe"


@dataclass(frozen=True)
class Method:
    """Способ уточнения: сеть, сжатие куска по вертикали, подготовка."""

    engine: Engine
    squeeze: int
    prep: Prep = Prep.NONE

    @property
    def name(self) -> str:
        """Имя для логов и сводки: ``sam3_sq4``, ``sam2_sq4_lvL``."""
        tail = "" if self.prep is Prep.NONE else f"_{self.prep.value}"
        return f"{self.engine.value}_sq{self.squeeze}{tail}"


# Сокращённый набор — способы, чаще других делавшие рваный край ровным на стенде; полный — плюс остальные.
REDUCED = (Method(Engine.SAM3, 4), Method(Engine.SAM2, 4, Prep.LEVELS), Method(Engine.SAM3, 1), Method(Engine.SAM2, 4))
FULL = REDUCED + (
    Method(Engine.SAM3, 4, Prep.LEVELS),
    Method(Engine.SAM2, 1),
    Method(Engine.SAM3, 4, Prep.CLAHE),
    Method(Engine.SAM2, 4, Prep.CLAHE),
)


class Band:
    """Лента вдоль дальней стороны страницы: столбцы поперёк (слева наружу, справа внутрь), строки — вдоль стороны."""

    def __init__(self, quad: np.ndarray, spine_right: bool, seg_px_per_mm: float, ratio: float) -> None:
        """Геометрия ленты.

        Args:
            quad: Трапеция страницы (4, 2) на масштабе сегментации.
            spine_right: Корешок справа от страницы (левая страница) — True.
            seg_px_per_mm: Масштаб сегментации, px/мм.
            ratio: Полное разрешение / масштаб сегментации.
        """
        a, b = (np.asarray(p, np.float64) for p in quad_sides(quad, spine_right)[Side.FAR])
        n = inward_normal(a, b, np.asarray(quad, np.float64).mean(axis=0))
        e = (b - a) / np.linalg.norm(b - a)
        self.ratio = ratio
        self.ppm = ratio * seg_px_per_mm
        self.length = int(np.linalg.norm(b - a) * ratio)
        self.width = int((BAND_OUT_MM + BAND_IN_MM) * self.ppm)
        off = ratio * a - BAND_OUT_MM * self.ppm * n
        # Строка-столбец ленты → точка полного кадра: p = off + col·n + row·e.
        self.m_full = np.array([[n[0], e[0], off[0]], [n[1], e[1], off[1]]], np.float64)
        self.m_seg = self.m_full / ratio

    def warp(self, img: np.ndarray, seg: bool, nearest: bool = False) -> np.ndarray:
        """Лента из кадра (``seg`` — кадр масштаба сегментации, иначе полного разрешения)."""
        m = self.m_seg if seg else self.m_full
        flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
        return cv2.warpAffine(img, m, (self.width, self.length), flags=flags, borderValue=0)

    def unwarp(self, band_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Маска ленты (полное разрешение ленты) → маска кадра сегментации формы ``shape`` (bool)."""
        out = cv2.warpAffine(
            band_mask.astype(np.uint8), self.m_seg, shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0
        )
        return out > 0

    def col(self, u_mm: float) -> int:
        """Столбец ленты на ``u_mm`` внутрь от стороны трапеции."""
        return int(round((u_mm + BAND_OUT_MM) * self.ppm))


@dataclass
class MethodResult:
    """Край одного способа по строкам ленты и его меры.

    Поля:
        method: Способ.
        edge: Столбец края по строкам (NaN — нет края: ядро пальца, промах куска).
        cover: Доля строк с краем (вне ядра пальца).
        rough_mm: p90 отклонения края от медианы окна :data:`MED_MM`, мм.
        step_mm: Наибольшая ступень края (:func:`step_mm`), мм.
        ink_out_mm2: Краска своей страницы за краем (снаружи новой маски), мм².
        missed_chunks: Кусков с промахом (пустая маска, маска без столбца страницы, маска «всё» по ленте).
        miss: Промах способа целиком (покрытие меньше :data:`MISS_COVER`).
        accepted: Прошёл приёмку.
    """

    method: Method
    edge: np.ndarray
    cover: float = 0.0
    rough_mm: float = float("nan")
    step_mm: float = float("nan")
    ink_out_mm2: float = float("nan")
    missed_chunks: int = 0
    miss: bool = False
    accepted: bool = False


@dataclass
class SideRefine:
    """Итог уточнения стороны.

    Поля:
        band: Лента.
        edge: Выбранный край по строкам (сглажен, без NaN) или None — сторона остаётся грубой.
        chosen: Имя выбранного способа или пусто.
        full_set: Пришлось ли идти в полный набор.
        results: Способ → итог.
        status: Причина, если край не выбран.
        finger_share: Доля строк стороны, где палец закрывает край (:func:`finger_rows_of`).
    """

    band: Band
    edge: np.ndarray | None
    chosen: str = ""
    full_set: bool = False
    results: dict[str, MethodResult] = field(default_factory=dict)
    status: str = ""
    finger_share: float = 0.0


def clahe(bgr: np.ndarray, ppm: float) -> np.ndarray:
    """CLAHE по светлоте L (клетка :data:`CLAHE_TILE_MM`, предел :data:`CLAHE_CLIP`)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    tiles = max(1, int(round(max(bgr.shape[:2]) / (CLAHE_TILE_MM * ppm))))
    lab[..., 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def stretch_levels(piece: np.ndarray, pos_col: int) -> np.ndarray:
    """Растяжение светлоты куска по бумаге страницы (столбцы от ``pos_col`` внутрь, краска отсечена Оцу).

    Чёрная точка — ``PAPER_PCT[0]``-й процентиль бумаги минус :data:`LEVEL_MARGIN`, белая — ``PAPER_PCT[1]``-й плюс
    запас; усиление не больше :data:`MAX_GAIN`; цветность не трогается. Бумаги мало — кусок как есть.

    Args:
        piece: Кусок ленты BGR.
        pos_col: Столбец, с которого внутрь лежит страница.

    Returns:
        Кусок BGR.
    """
    gray = cv2.cvtColor(piece[:, pos_col:], cv2.COLOR_BGR2GRAY)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = gray[gray > thr]
    if paper.size < 100:
        return piece
    lo, hi = np.percentile(paper, PAPER_PCT)
    white = min(255.0, float(hi) + LEVEL_MARGIN)
    black = max(0.0, min(float(lo) - LEVEL_MARGIN, white - 255.0 / MAX_GAIN))
    lut = np.clip((np.arange(256) - black) * 255.0 / (white - black), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def edge_of(mask: np.ndarray, start: int) -> np.ndarray:
    """Край по строкам: первый выход из маски при движении от столбца ``start`` наружу (влево); NaN — в ``start`` не
    маска; 0 — маска до самого наружного края ленты."""
    out = np.full(mask.shape[0], np.nan)
    inside = mask[:, start]
    left = mask[:, : start + 1][:, ::-1]
    first_false = np.argmax(~left, axis=1)
    has_false = (~left).any(axis=1)
    out[inside] = np.where(has_false, start - first_false + 1, 0)[inside]
    return out


def roughness_mm(edge: np.ndarray, ppm: float) -> float:
    """p90 отклонения края от скользящей медианы окна :data:`MED_MM`, мм (пропуски — интерполяцией)."""
    ok = np.isfinite(edge)
    if ok.sum() < 50:
        return float("nan")
    idx = np.arange(len(edge))
    u = np.interp(idx, idx[ok], edge[ok])
    med = ndi.median_filter(u, size=max(3, int(round(MED_MM * ppm)) | 1), mode="nearest")
    return float(np.percentile(np.abs(u - med)[ok], 90) / ppm)


def step_mm(edge: np.ndarray, ppm: float) -> float:
    """Наибольшая ступень края, мм: модуль разности медиан окна :data:`STEP_WIN_MM` до и после строки через
    :data:`STEP_GAP_MM` (пропуски — интерполяцией)."""
    ok = np.isfinite(edge)
    if ok.sum() < 50:
        return float("nan")
    idx = np.arange(len(edge))
    u = np.interp(idx, idx[ok], edge[ok])
    med = ndi.median_filter(u, size=max(3, int(round(STEP_WIN_MM * ppm)) | 1), mode="nearest")
    gap = max(1, int(round(STEP_GAP_MM * ppm)))
    return float(np.abs(med[2 * gap :] - med[: -2 * gap]).max() / ppm) if len(med) > 2 * gap else 0.0


class Refiner:
    """Сети уточнения края: предиктор SAM 3 и модель SAM 2.1 — те же, что у прогона v2 (грузятся один раз)."""

    def __init__(self, sam3_predictor, sam2) -> None:
        """Args:
        sam3_predictor: ``SAM3SemanticPredictor`` (``paper.load_sam3``).
        sam2: ``bg_veto.Sam2Background`` (метод ``masks_scored``).
        """
        self.sam3 = sam3_predictor
        self.sam2 = sam2

    def sam3_mask(self, bgr: np.ndarray, pos_col: int) -> np.ndarray:
        """Экземпляр SAM 3, лучше всех накрывающий столбец страницы ``pos_col`` (пусто — ничего не нашлось)."""
        import torch

        h, w = bgr.shape[:2]
        best, best_cov = np.zeros((h, w), bool), 0.0
        with torch.inference_mode():
            self.sam3.set_image(bgr)
            for prompt in SAM3_PROMPTS:
                res = self.sam3(text=[prompt])
                if not res or res[0].masks is None:
                    continue
                for m in res[0].masks.data.cpu().numpy() > 0.5:
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    cov = float(m[:, pos_col].mean())
                    if cov > best_cov:
                        best, best_cov = m, cov
        if hasattr(self.sam3, "reset_image"):
            self.sam3.reset_image()
        return best

    def sam2_mask(self, bgr: np.ndarray, pos_col: int, neg_col: int) -> np.ndarray:
        """Маска SAM 2.1 по трём точкам на странице и трём за краем: из трёх — с лучшей оценкой, накрывающая все
        положительные и ни одной отрицательной (пусто — ни одна не годна)."""
        h = bgr.shape[0]
        rows = [h * f for f in (0.15, 0.5, 0.85)]
        pts = [[float(pos_col), y] for y in rows] + [[float(neg_col), y] for y in rows]
        masks, scores = self.sam2.masks_scored(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), pts, [1, 1, 1, 0, 0, 0])
        best, best_s = np.zeros(bgr.shape[:2], bool), -1.0
        for m, s in zip(masks, scores):
            ok = sum(m[int(y), pos_col] for y in rows) - sum(m[int(y), neg_col] for y in rows)
            if ok >= 3 and s > best_s:
                best, best_s = m, float(s)
        return best

    def method_edge(self, band_img: np.ndarray, band: Band, method: Method) -> tuple[np.ndarray, int]:
        """Край способа по всей ленте: куски с перекрытием, в перекрытии — кусок с ближним центром; промахи кусков.

        Промах куска (его строки без края): маска пуста, маска не накрывает столбец страницы хотя бы в половине строк,
        маска до наружного края ленты в половине строк («всё»).

        Args:
            band_img: Лента BGR полного разрешения.
            band: Её геометрия.
            method: Способ.

        Returns:
            ``(край по строкам, кусков с промахом)``.
        """
        h = band_img.shape[0]
        step = int((CHUNK_MM - OVERLAP_MM) * method.squeeze * band.ppm)
        size = int(CHUNK_MM * method.squeeze * band.ppm)
        starts = list(range(0, max(h - size, 0) + 1, max(step, 1)))
        if starts[-1] + size < h:
            starts.append(max(h - size, 0))
        pos, neg = band.col(POS_MM), band.col(NEG_MM)
        src = clahe(band_img, band.ppm) if method.prep is Prep.CLAHE else band_img
        edge, dist, missed = np.full(h, np.nan), np.full(h, np.inf), 0
        for y0 in starts:
            y1 = min(y0 + size, h)
            rows = np.arange(y0, y1)
            closer = np.abs(rows - (y0 + y1) / 2) < dist[rows]
            piece = np.ascontiguousarray(src[y0:y1])
            if method.prep is Prep.LEVELS:
                piece = stretch_levels(piece, pos)
            ph, pw = piece.shape[:2]
            if method.squeeze > 1:
                piece = cv2.resize(piece, (pw, max(ph // method.squeeze, 8)), interpolation=cv2.INTER_AREA)
            if method.engine is Engine.SAM3:
                m = self.sam3_mask(piece, pos)
            else:
                m = self.sam2_mask(piece, pos, neg)
            if method.squeeze > 1:
                m = cv2.resize(m.astype(np.float32), (pw, ph), interpolation=cv2.INTER_LINEAR) > 0.5
            m = ndi.binary_fill_holes(m)
            e = edge_of(m, pos)
            # Промахи куска: пусто, мимо страницы, «всё» до наружного края ленты.
            if not m.any() or np.isnan(e).mean() > 0.5 or (e == 0).mean() > 0.5:
                missed += 1
                e = np.full(ph, np.nan)
            edge[rows[closer]] = e[closer]
            dist[rows[closer]] = np.abs(rows[closer] - (y0 + y1) / 2)
        return edge, missed


def ink_in_band(band_img: np.ndarray, hulls_band: np.ndarray, pos: int, ppm: float) -> np.ndarray:
    """Знаки текста своей страницы в ленте: пятна темнее бумаги страницы на :data:`INK_DROP`, с цветностью меньше
    :data:`INK_CHROMA`, внутри оболочки своей печати, размером со знак (``GLYPH_*``).

    Args:
        band_img: Лента BGR.
        hulls_band: Оболочки своей печати в ленте (bool).
        pos: Столбец, с которого внутрь лежит страница (по нему — уровень бумаги).
        ppm: Масштаб ленты, px/мм.

    Returns:
        Маска знаков (bool).
    """
    lab = cv2.cvtColor(band_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[..., 0]
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    page = light[:, pos:]
    paper = float(np.median(page[page > np.percentile(page, 30)])) if page.size else 255.0
    ink = ((light < paper - INK_DROP) & (chroma < INK_CHROMA) & hulls_band).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = ((stats[1:, cv2.CC_STAT_HEIGHT] <= GLYPH_MAX_H_MM * ppm)
                & (stats[1:, cv2.CC_STAT_WIDTH] <= GLYPH_MAX_W_MM * ppm)
                & (stats[1:, cv2.CC_STAT_AREA] >= GLYPH_MIN_MM2 * ppm**2))  # fmt: skip
    return keep[labels]


def judge(res: MethodResult, finger_rows: np.ndarray, ink_band: np.ndarray, ppm: float) -> MethodResult:
    """Меры и приёмка способа.

    Args:
        res: Итог способа с краем.
        finger_rows: Строки ленты у ядра пальца (край там не нужен).
        ink_band: Краска своей страницы в ленте (печать внутри её оболочки, bool).
        ppm: Масштаб ленты, px/мм.

    Returns:
        Тот же итог с мерами, ``miss`` и ``accepted``.
    """
    e = res.edge.copy()
    e[finger_rows] = np.nan
    res.edge = e
    # Меры — без строк у углов стороны (END_SHARE): там край уходит за угол листа на верх и низ, а маска в них не
    # заменяется (``apply_edge``).
    cut = int(END_SHARE * len(e))
    e = e.copy()
    e[:cut] = np.nan
    e[len(e) - cut :] = np.nan
    valid = ~finger_rows
    valid[:cut] = False
    valid[len(e) - cut :] = False
    res.cover = float(np.isfinite(e[valid]).mean()) if valid.any() else 0.0
    res.miss = res.cover < MISS_COVER
    if res.miss:
        return res
    res.rough_mm = roughness_mm(e, ppm)
    res.step_mm = step_mm(e, ppm)
    # Краска своей страницы снаружи края (столбцы левее края в строке).
    cols = np.arange(ink_band.shape[1])[None, :]
    outside = cols < np.where(np.isfinite(e), e, -1)[:, None]
    res.ink_out_mm2 = float((ink_band & outside).sum()) / ppm**2
    res.accepted = (res.cover >= MIN_COVER and res.rough_mm <= RAGGED_OK_MM and res.step_mm <= STEP_MAX_MM
                    and res.ink_out_mm2 <= INK_OUT_MM2)  # fmt: skip
    return res


def choose(results: list[MethodResult], ppm: float) -> MethodResult | None:
    """Лучший из принятых: согласный с медианой принятых (в пределах :data:`AGREE_MM` по медиане строк), наименьшая
    рваность; при равенстве в пределах 0.05 мм — ближний к странице (край верхнего листа на веере).

    Args:
        results: Итоги способов.
        ppm: Масштаб ленты, px/мм.

    Returns:
        Выбранный итог или None — принятых нет.
    """
    ok = [r for r in results if r.accepted]
    if not ok:
        return None
    consensus = np.nanmedian(np.stack([r.edge for r in ok]), axis=0) if len(ok) > 1 else ok[0].edge
    agree = [r for r in ok if np.nanmedian(np.abs(r.edge - consensus)) / ppm <= AGREE_MM] or ok
    best = min(r.rough_mm for r in agree)
    near = [r for r in agree if r.rough_mm <= best + 0.05]
    return max(near, key=lambda r: float(np.nanmedian(r.edge)))


def finger_rows_of(core_band: np.ndarray, page_band: np.ndarray, band: Band) -> np.ndarray:
    """Строки ленты, где палец закрывает край: ядро пальца (с запасом) в пределах :data:`FINGER_EDGE_MM` поперёк от
    грубого края строки; в строках без грубого края — от стороны трапеции.

    Ядро где-то ещё в ленте (ладонь снаружи книги, палец глубоко на странице) край не закрывает: там край ищется.

    Args:
        core_band: Ядро пальца с запасом в ленте (bool, полное разрешение ленты).
        page_band: Грубая маска страницы в ленте (bool).
        band: Лента.

    Returns:
        Маска строк (bool) длины ``band.length``.
    """
    rows, width = core_band.shape
    coarse = edge_of(page_band, band.col(POS_MM))
    # Столбец края по строкам: грубый край, где он есть, иначе сторона трапеции.
    centre = np.where(np.isfinite(coarse), coarse, float(band.col(0.0)))
    half = FINGER_EDGE_MM * band.ppm
    cols = np.arange(width)[None, :]
    near = np.abs(cols - centre[:, None]) <= half
    return (core_band & near).any(axis=1)


def refine_side(
    refiner: Refiner,
    clean: np.ndarray,
    quad: np.ndarray,
    spine_right: bool,
    page_seg: np.ndarray,
    hulls_seg: np.ndarray,
    core_seg: np.ndarray,
    seg_px_per_mm: float,
) -> SideRefine:
    """Уточнение дальней стороны страницы: сокращённый набор способов, при неудаче — полный.

    Строки без выбранного края (ядро пальца, промахи кусков) — интерполяцией выбранного края (:func:`fill_edge`).

    Args:
        refiner: Сети.
        clean: Кадр без пальцев полного разрешения (BGR).
        quad: Трапеция страницы на масштабе сегментации.
        spine_right: Корешок справа от страницы (левая страница).
        page_seg: Грубая маска этой страницы на масштабе сегментации.
        hulls_seg: Оболочки своей печати кадра, масштаб сегментации.
        core_seg: Ядро пальцев на масштабе сегментации.
        seg_px_per_mm: Масштаб сегментации, px/мм.

    Returns:
        Итог стороны.
    """
    band = Band(quad, spine_right, seg_px_per_mm, clean.shape[1] / page_seg.shape[1])
    band_img = band.warp(clean, seg=False)
    pos = band.col(POS_MM)
    page_band = band.warp(page_seg.astype(np.uint8), seg=True, nearest=True) > 0
    grow = int(round(FINGER_CORE_GROW_MM * seg_px_per_mm))
    core = cv2.dilate(core_seg.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) if grow else core_seg
    core_band = band.warp(core.astype(np.uint8), seg=True, nearest=True) > 0
    finger_rows = finger_rows_of(core_band, page_band, band)
    ink_band = ink_in_band(band_img, band.warp(hulls_seg.astype(np.uint8), seg=True, nearest=True) > 0, pos, band.ppm)
    ink_band[:, : band.col(INK_FROM_SIDE_MM)] = False
    out = SideRefine(band, None, finger_share=float(finger_rows.mean()))
    for full_set, methods in ((False, REDUCED), (True, FULL)):
        for method in methods:
            if method.name in out.results:
                continue
            edge, missed = refiner.method_edge(band_img, band, method)
            res = judge(MethodResult(method, edge, missed_chunks=missed), finger_rows, ink_band, band.ppm)
            out.results[method.name] = res
        best = choose(list(out.results.values()), band.ppm)
        if best is not None:
            out.full_set = full_set
            out.chosen = best.method.name
            out.edge = fill_edge(best, page_band, pos, band.ppm)
            return out
    out.status = "ни один способ не принят"
    return out


def fill_edge(best: MethodResult, page_band: np.ndarray, pos: int, ppm: float) -> np.ndarray:
    """Выбранный край без пропусков (палец у края, промахи кусков): линейная интерполяция по самому выбранному краю,
    у концов стороны — продолжение крайней точки; край без единой точки — по грубой маске. Сглаживание медианой
    :data:`SMOOTH_MM`.

    Грубая маска в пропуске не годится: под пальцем она огибает зону закраса и тень пальца выемкой внутрь (ПХ 1975/04
    IMG_0042 L — до 460 мм² на стороне, проверка 2026-10-23), а подстановка других способов дала бы ступень на стыке.
    Длинная прямая через пропуск была следствием лишних строк пальца (ладонь снаружи книги), их отсекает
    :func:`finger_rows_of`.

    Args:
        best: Выбранный способ.
        page_band: Грубая маска страницы в ленте.
        pos: Столбец страницы (начало поиска края).
        ppm: Масштаб ленты, px/мм.

    Returns:
        Край по строкам (столбцы ленты).
    """
    e = best.edge.copy()
    ok = np.isfinite(e)
    if not ok.any():
        e = edge_of(page_band, pos)
        ok = np.isfinite(e)
        if not ok.any():
            return np.full_like(e, float(pos))
    idx = np.arange(len(e))
    e = np.interp(idx, idx[ok], e[ok])
    return ndi.median_filter(e, size=max(3, int(round(SMOOTH_MM * ppm)) | 1), mode="nearest")


def apply_edge(paper: np.ndarray, band: Band, edge: np.ndarray) -> np.ndarray:
    """Бумага кадра сегментации с заменой в ленте: от края внутрь — бумага, снаружи — нет.

    Args:
        paper: Маска бумаги кадра (bool), масштаб сегментации.
        band: Лента стороны.
        edge: Край по строкам ленты (столбцы полного разрешения ленты).

    Returns:
        Новая маска бумаги.
    """
    cols = np.arange(band.width)[None, :]
    inside = band.unwarp(cols >= edge[:, None], paper.shape)
    # Средние строки стороны: от края внутрь — бумага, снаружи — нет (ямы залиты, языки срезаны). Строки у углов
    # (END_SHARE): трапеция и угол листа расходятся, поэтому там только срез языков — бумага не прирастает за верх и
    # низ листа, а ступени на границе средних строк нет.
    rows = np.zeros(band.length, bool)
    cut = int(END_SHARE * band.length)
    rows[cut : band.length - cut] = True
    middle = band.unwarp(np.repeat(rows[:, None], band.width, axis=1), paper.shape)
    whole = band.unwarp(np.ones((band.length, band.width), bool), paper.shape)
    ends = whole & ~middle
    return (paper & ~whole) | (middle & inside) | (ends & paper & inside)
