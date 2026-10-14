"""Своя печать кадра по глифам: сгустки глифов цепочкой (строки) с зазором — защита от закраса и затравки сегментации."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.orientation.detectors.ink_axis import LINE_MAX_THICKNESS_PX, RLSA_GAP_PX, glyph_mask

# Зазор вокруг своей печати, px рабочего масштаба ``finger_protect`` (½ входа). Калибровка — превью газет ЭГ
# (~5 px/мм на рабочем масштабе): 10 px ≈ 2 мм бумаги (``research/newspaper_crop``).
GUARD_PX = 10
# Окно и сдвиг адаптивного порога краски на рабочем масштабе (~три строки текста), уровни серого.
INK_BLOCK, INK_OFFSET = 31, 18
# Смыкание своей печати в сплошные колонки перед раздутием, px рабочего масштаба: край зазора идёт гладко по краю
# колонки, а не по каждой букве.
CLOSE_PX = 41
# Наименьшее число глифов в сгустке, который считается печатью: слово из трёх букв — уже печать.
FINGER_MIN_GLYPHS = 3
# Масштаб, на котором считается защита для кадров подшивок: ``finger_protect`` калиброван на превью ЭГ (~190 dpi,
# внутри ещё ½), кадры МТО 1990 ~350 dpi — уменьшаем вдвое, чтобы глифы были того же размера в px.
GLYPH_SCALE = 0.5
# Зазор маски «своей печати» (вход сегментации фона и бумаги), px масштаба ``finger_protect`` (¼ кадра).
CONTENT_GUARD_PX = 3
# Доля кадра, в которой хранится маска своей печати (как кэш стенда ``research/bound_volumes``: ¼ кадра).
CONTENT_SCALE = 0.25


def guard_mask(own: np.ndarray, guard_px: int = GUARD_PX, close_px: int = CLOSE_PX) -> np.ndarray:
    """Зазор: своя печать, сомкнутая в колонки и раздутая на ``guard_px``.

    Args:
        own: Маска своей печати (bool или 0/1).
        guard_px: Ширина зазора, px рабочего масштаба.
        close_px: Окно смыкания (прямоугольник), px; 0 — без смыкания.

    Returns:
        Булева маска зазора.
    """
    m = own.astype(np.uint8)
    if close_px > 0:
        # Смыкание прямоугольником склеивает буквы и строки колонки в один сгусток.
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)))
    # Раздутие эллипсом на ширину зазора.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_px + 1, 2 * guard_px + 1))
    return cv2.dilate(m, k) > 0


def finger_protect(rgb: np.ndarray, guard_px: int = GUARD_PX) -> np.ndarray:
    """Защита печати от закраса пальцев: глифы, стоящие цепочкой, по всему кадру с зазором, в размере кадра.

    Берутся компоненты размера глифа (``ink_axis.glyph_mask`` по адаптивной краске), смыкаются вдоль и поперёк,
    и охраняются тонкие сгустки не меньше :data:`FINGER_MIN_GLYPHS` глифов — одиночные тёмные пятна в тени пальца,
    мрамор торца крышки и доски пола (толстые неровные сгустки) не охраняются.

    Args:
        rgb: Кадр RGB (рабочий масштаб считается как ½ этого кадра).
        guard_px: Зазор у печати, px рабочего масштаба.

    Returns:
        Маска uint8 0/255 размера кадра.
    """
    h, w = rgb.shape[:2]
    shape = (h // 2, w // 2)
    # Адаптивная краска на рабочем масштабе: фон кадра неоднороден, глобальный порог не годится.
    gray = cv2.cvtColor(cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, INK_BLOCK, INK_OFFSET)
    # glyph_mask сам бинаризует Оцу по серому: подаём ему уже чистую краску (инвертированную в «чёрное на белом»).
    glyphs = glyph_mask(255 - ink)
    # Центры глифов: по ним считается, сколько глифов в каждом сгустке.
    _, _, _, centres = cv2.connectedComponentsWithStats(glyphs, connectivity=8)
    cx = np.clip(np.round(centres[1:, 0]).astype(int), 0, shape[1] - 1)
    cy = np.clip(np.round(centres[1:, 1]).astype(int), 0, shape[0] - 1)
    lines = np.zeros(shape, bool)
    # Строки каждой ориентации набора отдельно: смыкание вдоль оси, сгусток — строка, если в нём не меньше
    # FINGER_MIN_GLYPHS глифов и он тонкий по повёрнутому прямоугольнику (наклон строки не мешает).
    for kernel in (np.ones((1, RLSA_GAP_PX), np.uint8), np.ones((RLSA_GAP_PX, 1), np.uint8)):
        smeared = cv2.morphologyEx(glyphs, cv2.MORPH_CLOSE, kernel)
        count, labels = cv2.connectedComponents(smeared, connectivity=8)
        per_cluster = np.bincount(labels[cy, cx], minlength=count)
        # Пиксели сгустков, сгруппированные по метке (одна сортировка вместо маски на каждый сгусток).
        flat = labels.ravel()
        order = np.argsort(flat, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(flat, minlength=count))]
        keep = np.zeros(count, bool)
        for i in np.nonzero(per_cluster >= FINGER_MIN_GLYPHS)[0]:
            if i == 0:
                continue
            pix = order[bounds[i] : bounds[i + 1]]
            pts = np.c_[pix % shape[1], pix // shape[1]].astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            keep[i] = min(rw, rh) <= LINE_MAX_THICKNESS_PX
        lines |= keep[labels]
    own = (glyphs > 0) & lines
    guard = guard_mask(own, guard_px).astype(np.uint8) * 255
    return cv2.resize(guard, (w, h), interpolation=cv2.INTER_NEAREST)


def glyph_protect(rgb: np.ndarray, guard_px: int, scale: float = GLYPH_SCALE) -> np.ndarray:
    """Защита печати по глифам в размере кадра, посчитанная на копии масштаба :data:`GLYPH_SCALE`.

    Args:
        rgb: Кадр RGB в исходном разрешении.
        guard_px: Зазор у печати, px масштаба ``finger_protect`` (¼ кадра).
        scale: Масштаб уменьшенной копии относительно кадра (``Scale.glyph``; для калибровочных кадров — ½).

    Returns:
        Маска uint8 0/255 размера кадра.
    """
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    guard = finger_protect(small, guard_px)
    return cv2.resize(guard, (w, h), interpolation=cv2.INTER_NEAREST)


def content_mask(rgb: np.ndarray, shape: tuple[int, int], scale: float = GLYPH_SCALE) -> np.ndarray:
    """Своя печать кадра (защита глифов с зазором :data:`CONTENT_GUARD_PX`) в размере ``shape``.

    Повторяет кэш ``geometry/content`` стенда подшивок: маска хранилась на ¼ кадра (ближайший сосед) и уже оттуда
    приводилась к масштабу сегментации — тот же путь даёт ту же маску пиксель в пиксель.

    Args:
        rgb: Кадр RGB после закраса пальцев, исходное разрешение.
        shape: ``(H, W)`` масштаба сегментации.
        scale: Масштаб копии защиты глифов (``Scale.glyph``); маска хранится на половине этого масштаба.

    Returns:
        Булева маска.
    """
    h, w = rgb.shape[:2]
    full = glyph_protect(rgb, CONTENT_GUARD_PX, scale)
    # Промежуточная маска на масштабе ``finger_protect`` (¼ калибровочного кадра) — как кэш стенда.
    q = scale * CONTENT_SCALE / GLYPH_SCALE
    quarter = cv2.resize(full, (round(w * q), round(h * q)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(quarter, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
