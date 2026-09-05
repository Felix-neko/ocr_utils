"""Debug-оверлеи: что закрасили и что защитили от размытия. Всегда JPG.

Два файла на полосу:

* ``<путь>.jpg`` — вся полоса ДО закраса с полупрозрачными масками поверх. До
  закраса намеренно: иначе не видно, что именно убрали;
* ``<путь>_inpaint.jpg`` — только для полос с масками: по каждой зоне полоска
  «до | после» в масштабе 1:1. На уменьшенной странице в 6000 px качество заливки
  просто не разглядеть, а судить надо именно о нём.

Цвета и приём наложения взяты у ``background_smoothing.pipeline.draw_overlay``,
чтобы оверлеи двух подсистем читались одинаково.

Подписи — ASCII: у ``cv2.putText`` нет кириллических глифов, и русский текст вышел
бы рядом знаков вопроса. Тащить сюда PIL с TTF ради подписи не стоит.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Заливки масок (BGR) и их прозрачность — как в background_smoothing.
COLOR_DILATED = (0, 255, 255)  # жёлтый — защитная маска (что НЕ размывается)
COLOR_PRIMARY = (0, 0, 255)  # красный — найденный контент
COLOR_INPAINTED = (255, 0, 255)  # пурпурный — закрашенные маски разметки
ALPHA_DILATED = 0.25
ALPHA_PRIMARY = 0.45
ALPHA_INPAINTED = 0.45

# Рамки без заливки: под ними должно быть видно саму картинку.
COLOR_REGION = (0, 255, 0)  # ярко-зелёный — растровые области из базы
COLOR_ROI = (255, 200, 0)  # голубой — ROI зон закраса, по ним видно группировку
OUTLINE_FRAC = 0.0008  # толщина рамки как доля длинной стороны кадра

# Длинная сторона полностраничного оверлея. Полный кадр в 21 Мп в JPEG весит
# мегабайты, а разглядывать на нём всё равно нечего — для этого есть врезки 1:1.
OVERLAY_SIDE = 2400
OVERLAY_QUALITY = 88

# Поле вокруг зоны на врезке «до | после», доля её размера.
CROP_PAD_FRAC = 0.35

# Высота полоски с подписью, пикс.
LABEL_BAR_PX = 34


def _blend(canvas: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    """Подмешивает цвет под маской, на месте. Только под маской — полнокадровая
    заливка на 21 Мп стоила бы лишних 63 МБ на каждый цвет."""
    idx = mask > 0
    if not idx.any():
        return
    canvas[idx] = (canvas[idx].astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def draw_page_overlay(
    bgr: np.ndarray,
    m_primary: "np.ndarray | None",
    m_dilated: "np.ndarray | None",
    regions=(),
    inpaint_masks=None,
    rois=(),
) -> np.ndarray:
    """Полностраничный оверлей поверх ИСХОДНОГО кадра."""
    canvas = bgr.copy()
    if m_dilated is not None:
        _blend(canvas, m_dilated, COLOR_DILATED, ALPHA_DILATED)
    if m_primary is not None:
        _blend(canvas, m_primary, COLOR_PRIMARY, ALPHA_PRIMARY)
    for mask in (inpaint_masks or {}).values():
        _blend(canvas, mask, COLOR_INPAINTED, ALPHA_INPAINTED)

    thickness = max(1, int(round(OUTLINE_FRAC * max(canvas.shape[:2]))))
    for r in regions:
        cv2.rectangle(canvas, (r.x1, r.y1), (r.x2, r.y2), COLOR_REGION, thickness)
        cv2.putText(
            canvas,
            r.kind,
            (r.x1 + 4, max(20, r.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            thickness * 0.5,
            COLOR_REGION,
            thickness,
        )
    for _kind, (x1, y1, x2, y2) in rois:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_ROI, thickness)
    return canvas


def downscale(img: np.ndarray, side: int = OVERLAY_SIDE) -> np.ndarray:
    """Уменьшение до длинной стороны ``side``; картинка меньше — возвращается как есть."""
    scale = side / max(img.shape[:2])
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Подпись ASCII в левом верхнем углу, на тёмной подложке.

    Публичная: ею подписывает врезки и ``compare``. ASCII — потому что у
    ``cv2.putText`` нет кириллических глифов.
    """
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], LABEL_BAR_PX), (0, 0, 0), -1)
    cv2.putText(out, text, (8, LABEL_BAR_PX - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def zone_strips(before: np.ndarray, after: np.ndarray, rois, pad_frac: float = CROP_PAD_FRAC) -> "np.ndarray | None":
    """Врезки «до | после» по каждой зоне, в масштабе 1:1, столбиком.

    ``rois`` — ``[(вид, (x1, y1, x2, y2)), ...]``. Пустой список → ``None``.
    """
    strips = []
    for kind, (x1, y1, x2, y2) in rois:
        px, py = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(before.shape[1], x2 + px), min(before.shape[0], y2 + py)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        left = label(before[cy1:cy2, cx1:cx2], f"before {kind} {cx2 - cx1}x{cy2 - cy1}")
        right = label(after[cy1:cy2, cx1:cx2], "after")
        gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
        strips.append(np.hstack([left, gap, right]))
    if not strips:
        return None

    width = max(s.shape[1] for s in strips)
    padded = [np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0)), constant_values=255) for s in strips]
    separator = np.full((12, width, 3), 255, np.uint8)
    stacked: "list[np.ndarray]" = []
    for s in padded:
        stacked.extend([s, separator])
    return np.vstack(stacked[:-1])


def write_overlays(
    debug_dir: Path,
    rel_path: str,
    before: np.ndarray,
    after: np.ndarray,
    m_primary=None,
    m_dilated=None,
    regions=(),
    inpaint_masks=None,
    rois=(),
) -> None:
    """Пишет оба оверлея, зеркаля структуру подпапок входа."""
    out = (debug_dir / rel_path).with_suffix(".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    page = draw_page_overlay(before, m_primary, m_dilated, regions, inpaint_masks, rois)
    cv2.imwrite(str(out), downscale(page), [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_QUALITY])

    if rois:
        strips = zone_strips(before, after, rois)
        if strips is not None:
            zoom = out.with_name(f"{out.stem}_inpaint.jpg")
            cv2.imwrite(str(zoom), strips, [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_QUALITY])
