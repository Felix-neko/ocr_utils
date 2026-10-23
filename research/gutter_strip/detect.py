"""Способы обнаружения полосы-вклейки у корешка по ленте: щель и рваность экземпляров SAM 3 (A), светлое плато тона (B), пара прямых ступеней (C), смена бумаги у сгиба по строкам (D), ширина ленты без текста (E), пара тонких складок по кромкам (F)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import mm_px
from ocr_utils.scan_cropping.v2.gutter_strip import CREASE_OFFSET_MM, Estimate, detect_sam3
from research.gutter_strip.band import Band, StripFrame
from research.page_cutting.fold import find_fold, page_spine_edge

# Ширина полосы-вклейки, мм: на ПХ 1975 — 8–15 мм (набор gutter_strip); с запасом в обе стороны.
STRIP_MIN_MM = 4.0
STRIP_MAX_MM = 20.0
# Середина полосы не дальше этого от предварительного корешка, мм.
CENTRE_MAX_MM = 12.0
# B: края плато не берутся в уровень полосы (тёмная линия кромки), мм; окно бумаги страницы за краем — от/до, мм.
PLATEAU_EDGE_MM = 1.0
PAPER_NEAR_MM = 1.5
PAPER_FAR_MM = 6.0
# C: ступень яркости бумаги — перепад больше этого на 1 px сегментации (уровни L 0–255); наклоны сдвига ленты, градусы.
STEP_L_PER_PX = 2.5
SHEAR_DEG = np.arange(-1.5, 1.51, 0.25)
# Сглаживание ленты вдоль корешка перед ступенями, мм.
ALONG_SMOOTH_MM = 3.0
# F: складка — глубже этого по сравнению с бумагой на смещении ``CREASE_OFFSET_MM``, уровни L.
CREASE_L = 6.0
# D: край страницы дальше этого от сгиба — «чужая бумага у сгиба», мм.
EDGE_FAR_MM = 2.0
# E: пиксель — текст, если темнее бумаги строки на столько уровней L.
TEXT_DARK_L = 40.0


def centre_ok(i: np.ndarray, j: np.ndarray, us: np.ndarray) -> np.ndarray:
    """Середина интервала столбцов ``[i, j]`` не дальше ``CENTRE_MAX_MM`` от корешка ленты."""
    return np.abs((us[i] + us[j]) / 2) <= mm_px(CENTRE_MAX_MM)


def candidate_pairs(us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Все пары столбцов ``i < j`` с шириной ``STRIP_MIN_MM … STRIP_MAX_MM`` и серединой у корешка."""
    n = len(us)
    lo, hi = int(round(mm_px(STRIP_MIN_MM))), int(round(mm_px(STRIP_MAX_MM)))
    i, w = np.meshgrid(np.arange(n), np.arange(lo, hi + 1), indexing="ij")
    i, j = i.ravel(), (i + w).ravel()
    ok = j < n
    i, j = i[ok], j[ok]
    ok = centre_ok(i, j, us)
    return i[ok], j[ok]


def window_mean(csum: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Средние профиля на отрезках ``[a, b)`` по кумулятивной сумме (границы обрезаются по профилю)."""
    n = len(csum) - 1
    a, b = np.clip(a, 0, n), np.clip(b, 0, n)
    return np.where(b > a, (csum[b] - csum[a]) / np.maximum(b - a, 1), np.nan)


def detect_plateau(band: Band) -> Estimate:
    """B. Светлое плато: участок 4–20 мм у корешка, медианная яркость бумаги которого выше бумаги страниц по обе стороны.

    Полоса-вклейка белее страниц (ПХ 1975: L 246–250 против 233–243), тень и щель у корешка темнее — у них контраст
    отрицательный. Уровень полосы — медиана (складка-сгиб посередине полосы её не роняет). Балл — наименьший из двух
    перепадов «полоса − бумага слева/справа», уровни L.
    """
    prof = np.median(band.light, axis=0).astype(np.float64)
    n = len(prof)
    e = int(round(mm_px(PLATEAU_EDGE_MM)))
    near, far = int(round(mm_px(PAPER_NEAR_MM))), int(round(mm_px(PAPER_FAR_MM)))
    # Бумага страницы за краем — медиана окна [near, far) от края наружу (по столбцу края).
    side = far - near
    paper = median_filter(prof, size=side, mode="nearest")  # медиана окна с центром в столбце
    best = (-np.inf, 0, 0)
    lo, hi = int(round(mm_px(STRIP_MIN_MM))), int(round(mm_px(STRIP_MAX_MM)))
    for w in range(lo, hi + 1):
        inner = w - 2 * e
        if inner < 3:
            continue
        inside = median_filter(prof, size=inner, mode="nearest")  # центр окна — середина полосы
        i = np.arange(n)
        j = i + w
        ok = (j < n) & centre_ok(i, np.minimum(j, n - 1), band.us)
        mid = np.clip(i + w // 2, 0, n - 1)
        left_c = np.clip(i - near - side // 2, 0, n - 1)
        right_c = np.clip(j + near + side // 2, 0, n - 1)
        contrast = np.where(ok, np.fmin(inside[mid] - paper[left_c], inside[mid] - paper[right_c]), -np.inf)
        k = int(np.argmax(contrast))
        if contrast[k] > best[0]:
            best = (float(contrast[k]), k, k + w)
    score, bi, bj = best
    if not np.isfinite(score):
        return Estimate(float("nan"))
    tone = np.median(band.tone, axis=0)
    inside_tone = np.median(tone[bi + e : bj - e + 1])
    outside_tone = np.median(np.r_[tone[max(bi - far, 0) : max(bi - near, 0)], tone[bj + near + 1 : bj + far + 1]])
    return Estimate(score, float(band.us[bi]), float(band.us[bj]),
                    {"width_mm": float(band.mm(bj - bi)), "tone_diff": float(inside_tone - outside_tone)})  # fmt: skip


def crease_map(band: Band) -> np.ndarray:
    """F. Тонкие тёмные складки: пиксель темнее бумаги по обе стороны на ``CREASE_OFFSET_MM`` больше ``CREASE_L``."""
    k = max(3, int(round(mm_px(ALONG_SMOOTH_MM))) | 1)
    light = cv2.blur(band.light, (1, k))
    d = max(1, int(round(mm_px(CREASE_OFFSET_MM))))
    left = np.roll(light, d, axis=1)
    right = np.roll(light, -d, axis=1)
    v = np.minimum(left, right) - light
    v[:, :d] = 0
    v[:, -d:] = 0
    return v > CREASE_L


def detect_creases(band: Band) -> Estimate:
    """F. Пара прямых тонких складок по кромкам полосы на расстоянии 4–20 мм (сгиб посередине полосы — третья).

    Для каждого наклона ``SHEAR_DEG`` — доля строк со складкой в столбце; балл — лучшая пара ``min(доля_i, доля_j)``.
    Широкая тёмная щель стыка тетрадей складкой не считается: у неё темно и на смещении ``CREASE_OFFSET_MM``.
    """
    crease = crease_map(band)
    i, j = candidate_pairs(band.us)
    best = (-1.0, 0, 0, 0.0)
    for deg in SHEAR_DEG:
        c = sheared_share(crease, deg)
        c = np.maximum.reduce([c, np.roll(c, 1), np.roll(c, -1)])
        s = np.minimum(c[i], c[j])
        k = int(np.argmax(s))
        if s[k] > best[0]:
            best = (float(s[k]), int(i[k]), int(j[k]), float(deg))
    score, bi, bj, deg = best
    return Estimate(score, float(band.us[bi]), float(band.us[bj]),
                    {"width_mm": float(band.mm(band.us[bj] - band.us[bi])), "shear_deg": deg})  # fmt: skip


def step_maps(band: Band) -> tuple[np.ndarray, np.ndarray]:
    """C. Ступени яркости бумаги поперёк корешка: подъём и спад больше ``STEP_L_PER_PX`` (bool по ленте)."""
    k = max(3, int(round(mm_px(ALONG_SMOOTH_MM))) | 1)
    light = cv2.blur(band.light, (1, k))
    g = np.zeros_like(light)
    g[:, 1:-1] = (light[:, 2:] - light[:, :-2]) / 2
    return g > STEP_L_PER_PX, g < -STEP_L_PER_PX


def sheared_share(mask: np.ndarray, deg: float) -> np.ndarray:
    """Доля строк со ступенью по столбцам после сдвига строк на наклон ``deg`` (прямая кромка под углом к корешку)."""
    h, w = mask.shape
    shift = np.round(np.tan(np.radians(deg)) * (np.arange(h) - h / 2)).astype(int)
    ys, xs = np.nonzero(mask)
    xs = xs - shift[ys]
    ok = (xs >= 0) & (xs < w)
    return np.bincount(xs[ok], minlength=w)[:w] / max(h, 1)


def detect_edges(band: Band) -> Estimate:
    """C. Пара длинных прямых ступеней: подъём слева (вход на белую полосу), спад справа, 4–20 мм между ними.

    Для каждого наклона ``SHEAR_DEG`` — доля строк со ступенью в столбце; балл — лучшая пара ``min(подъём_i, спад_j)``
    (доля высоты ленты, на которой обе кромки видны).
    """
    rise, fall = step_maps(band)
    i, j = candidate_pairs(band.us)
    best = (-1.0, 0, 0, 0.0)
    for deg in SHEAR_DEG:
        r, f = sheared_share(rise, deg), sheared_share(fall, deg)
        # Ступень кромки размыта на 1–2 px: максимум в окне ±1 px.
        r = np.maximum.reduce([r, np.roll(r, 1), np.roll(r, -1)])
        f = np.maximum.reduce([f, np.roll(f, 1), np.roll(f, -1)])
        s = np.minimum(r[i], f[j])
        k = int(np.argmax(s))
        if s[k] > best[0]:
            best = (float(s[k]), int(i[k]), int(j[k]), float(deg))
    score, bi, bj, deg = best
    return Estimate(score, float(band.us[bi]), float(band.us[bj]),
                    {"width_mm": float(band.mm(band.us[bj] - band.us[bi])), "shear_deg": deg})  # fmt: skip


def detect_paper_change(sf: StripFrame) -> Estimate:
    """D. Смена бумаги у сгиба (``page_cutting.fold.page_spine_edge``): край страницы по цвету дальше ``EDGE_FAR_MM`` от сгиба.

    Балл — наибольшая по страницам доля строк, где бумага страницы кончается до сгиба (между ними чужая бумага).
    """
    img = sf.frame.img
    pages = sf.frame.pages
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fold = find_fold(gray, pages[0].mask, pages[1].mask, SEG_PX_PER_MM)
    if fold is None:
        return Estimate(float("nan"))
    shares, widths = [], []
    for page, spine_right in ((pages[0], True), (pages[1], False)):
        edge = page_spine_edge(img, fold, page.mask, spine_right, SEG_PX_PER_MM)
        d = np.abs(edge[:, 0] - fold.x_at(edge[:, 1])) / SEG_PX_PER_MM
        shares.append(float((d >= EDGE_FAR_MM).mean()) if len(d) else 0.0)
        widths.append(float(np.median(d)) if len(d) else 0.0)
    return Estimate(max(shares), extra={"share_left": shares[0], "share_right": shares[1],
                                        "width_left_mm": widths[0], "width_right_mm": widths[1]})  # fmt: skip


def detect_textfree(band: Band) -> Estimate:
    """E. Ширина ленты без текста вокруг корешка (медиана по строкам), мм: на полосе-вклейке букв нет — слабый признак."""
    gray = cv2.cvtColor(band.bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    text = gray < band.light - TEXT_DARK_L
    # Тёмная щель у корешка — не текст: столбцы, тёмные больше чем в половине строк, не в счёт.
    text &= ~(text.mean(axis=0) > 0.5)[None, :]
    c = int(np.argmin(np.abs(band.us)))
    widths = []
    for row in text[:: max(1, len(text) // 300)]:
        left = np.flatnonzero(row[:c])
        right = np.flatnonzero(row[c:])
        a = left[-1] if len(left) else 0
        b = c + right[0] if len(right) else len(row) - 1
        widths.append(b - a)
    return Estimate(float(band.mm(np.median(widths))))


METHODS = ("sam3", "plateau", "edges", "creases", "paper", "textfree")


def detect_all(sf: StripFrame, band: Band) -> dict[str, Estimate]:
    """Все способы по кадру: имя способа → оценка."""
    return {
        "sam3": detect_sam3(band),
        "plateau": detect_plateau(band),
        "edges": detect_edges(band),
        "creases": detect_creases(band),
        "paper": detect_paper_change(sf),
        "textfree": detect_textfree(band),
    }
