"""Стенд «кайма кожи в copy back»: copy back без ядра пальца, раздутого на припуск, — мера розовой каймы у края ядра и вырезки у пальцев «оригинал | закрас | припуск 0 | 4 | 8 px»."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.copy_back import LAYOUT_PAD_PX, copy_back_blocks
from ocr_utils.scan_cropping.v2.pipeline import read_layout, read_layout_labels, to_frame
from ocr_utils.scan_cropping.v2.scale import Scale

# Припуск ядра пальца, px калибровочного кадра (пересчёт через ``Scale.frame_px``, как у ``LAYOUT_PAD_PX``). Замер на
# МТС 1990/04 IMG_1197 (13.8 px/мм): превышение a* над бумагой у края ядра 6 → 5 → 4 на 1, 3, 5 px, 2 на 8–12 px, 0 с
# 16 px; решение пользователя 2026-10-15 — 4 px: кайма почти исчезает, буквы у пальца остаются.
CORE_GROW_PX = 4
# Полоса за краем исходного ядра, где мерится кайма, px кадра (от, до).
RIM_BAND_PX = (1, 3)
# Опора «бумага» для меры каймы — зона закраса дальше этого от ядра, px кадра.
PAPER_FROM_CORE_PX = 40
# Поле вокруг каймы на вырезке, px кадра; ширина ряда вырезок на листе, px.
CROP_PAD = 60
ROW_WIDTH = 1600


@dataclass(frozen=True)
class MarginInputs:
    """Вход copy back одного кадра из кэша v2 с ``--save-fingers``.

    Поля:
        key: Ключ кадра.
        original: Исходный кадр BGR.
        clean: Кадр после закраса пальцев BGR.
        paint: Зона закраса пальцев (bool, размер кадра).
        core: Ядро пальца (bool, размер кадра).
        allowed: Где copy back разрешён — грубая маска книги в размере кадра (bool).
        polygons: Полигоны блоков surya в координатах кадра.
        labels: Типы блоков surya или None.
    """

    key: str
    original: np.ndarray
    clean: np.ndarray
    paint: np.ndarray
    core: np.ndarray
    allowed: np.ndarray
    polygons: list[np.ndarray]
    labels: list[str] | None


def load_margin_inputs(cache_dir: Path, source: Path) -> MarginInputs | None:
    """Вход copy back из папки кадра в кэше (``clean.png``, ``fingers.npz``, ``book.npz``, ``layout.json``).

    Args:
        cache_dir: Папка кадра в ``--cache`` прогона v2 с ``--save-fingers``.
        source: Исходный кадр.

    Returns:
        Вход или None, если нет ``clean.png`` или разметки surya.
    """
    polygons = read_layout(cache_dir / "layout.json")
    if not (cache_dir / "clean.png").exists() or not polygons:
        return None
    clean = cv2.imread(str(cache_dir / "clean.png"))
    with np.load(cache_dir / "fingers.npz") as f:
        paint, core = f["paint"] > 0, f["core"] > 0
    with np.load(cache_dir / "book.npz") as b:
        allowed = to_frame(b["book"], clean.shape[:2])
    return MarginInputs(
        cache_dir.name, cv2.imread(str(source)), clean, paint, core, allowed, polygons,
        read_layout_labels(cache_dir / "layout.json"),
    )  # fmt: skip


def grow_core(core: np.ndarray, px: int) -> np.ndarray:
    """Ядро пальца, раздутое эллипсом на ``px`` (0 — как есть).

    Args:
        core: Ядро (bool).
        px: Припуск, px.

    Returns:
        Раздутое ядро (bool).
    """
    if px <= 0:
        return core
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(core.astype(np.uint8), kernel) > 0


def copy_back_with_margin(inputs: MarginInputs, grow_px: int, scale: Scale) -> tuple[np.ndarray, np.ndarray]:
    """Copy back пакета, которому передано ядро с припуском: кайма кожи у края ядра не возвращается с оригинала.

    Args:
        inputs: Вход кадра.
        grow_px: Припуск ядра, px калибровочного кадра (0 — как сейчас в пакете).
        scale: Масштабы прогона (пересчёт припусков в px кадра).

    Returns:
        ``(итог copy back BGR, маска возвращённого bool)``.
    """
    pad = (scale.frame_px(LAYOUT_PAD_PX[0]), scale.frame_px(LAYOUT_PAD_PX[1]))
    core = grow_core(inputs.core, scale.frame_px(grow_px) if grow_px > 0 else 0)
    out, returned, _ = copy_back_blocks(
        inputs.original, inputs.clean, inputs.polygons, inputs.paint, inputs.allowed, pad, inputs.labels, core
    )
    return out, returned


def rim_redness(
    image: np.ndarray, core: np.ndarray, paint: np.ndarray, where: np.ndarray, band_px: tuple[int, int] = RIM_BAND_PX
) -> float:
    """Кайма кожи: медианное превышение a* над бумагой в полосе ``RIM_BAND_PX`` за краем исходного ядра.

    Args:
        image: Итог BGR.
        core: Исходное ядро (bool).
        paint: Зона закраса (bool).
        where: Где мерить (bool) — возврат copy back без припуска: вне него в полосе закрас LaMa, каймы там нет.
        band_px: Полоса за краем исходного ядра, px кадра (от, до).

    Returns:
        Превышение a* (уровни OpenCV); 0 — каймы нет; NaN — полосы нет.
    """
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    band = paint & where & (dist >= band_px[0]) & (dist <= band_px[1])
    paper = paint & (dist > PAPER_FROM_CORE_PX)
    if band.sum() < 20 or paper.sum() < 20:
        return float("nan")
    a = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)
    return float(np.median(a[band]) - np.median(a[paper]))


def margin_rows(inputs: MarginInputs, grows: list[int], scale: Scale) -> tuple[list[dict], list[np.ndarray]]:
    """Copy back с припусками ``grows``: меры по кадру и вырезки у мест, где припуск что-то меняет.

    Args:
        inputs: Вход кадра.
        grows: Припуски, px калибровочного кадра (первый — опорный, обычно 0).
        scale: Масштабы прогона.

    Returns:
        ``(строки мер по припускам, ряды вырезок «оригинал | закрас | припуски…»)``.
    """
    results = [copy_back_with_margin(inputs, grow, scale) for grow in grows]
    base_returned = results[0][1]
    outs = [out for out, _ in results]
    rows = [
        {
            "key": inputs.key,
            "grow_px": grow,
            "returned_px": int(returned.sum()),
            "removed_px": int((base_returned & ~returned).sum()),
            "rim_a": rim_redness(out, inputs.core, inputs.paint, base_returned),
            # Сразу за припуском: остаток каймы, которую припуск уже не вычитает.
            "rim_next_a": rim_redness(
                out, inputs.core, inputs.paint, base_returned, (scale.frame_px(grow) + 1, scale.frame_px(grow) + 3)
            ),
        }
        for grow, (out, returned) in zip(grows, results)
    ]
    # Вырезки — где самый большой припуск снимает возврат.
    changed = base_returned & ~results[int(np.argmax(grows))][1]
    crops = []
    if changed.sum() >= 50:
        grown = cv2.dilate(changed.astype(np.uint8), np.ones((31, 31), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(grown)
        h, w = changed.shape
        for k in range(1, count):
            x, y, bw, bh, area = stats[k]
            if area < 400:
                continue
            y0, y1 = max(0, y - CROP_PAD), min(h, y + bh + CROP_PAD)
            x0, x1 = max(0, x - CROP_PAD), min(w, x + bw + CROP_PAD)
            tiles = [im[y0:y1, x0:x1] for im in [inputs.original, inputs.clean] + outs]
            row = np.hstack([np.pad(t, ((0, 0), (0, 8), (0, 0)), constant_values=255) for t in tiles])
            k_scale = min(1.0, ROW_WIDTH / row.shape[1])
            crops.append(cv2.resize(row, None, fx=k_scale, fy=k_scale, interpolation=cv2.INTER_AREA))
    return rows, crops
