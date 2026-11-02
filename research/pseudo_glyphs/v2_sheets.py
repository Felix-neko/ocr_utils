"""Листы «было | стало» по отладке v2: картинки шага ``08_rectify`` двух прогонов (код базы и с фильтром псевдоглифов) рядом, шапка и легенда — в полях."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import framed

# Высота склейки, px: листы ужимаются до неё (отсмотр целиком; спорное место — по исходникам отладки).
SHEET_HEIGHT = 2000
# Поле между половинами, px.
GAP_PX = 16


def pairs(before: Path, after: Path, pattern: str = "08_rectify") -> list[tuple[Path, Path]]:
    """Пары одноимённых картинок шага в отладке двух прогонов.

    Args:
        before: Корень отладки «было».
        after: Корень отладки «стало».
        pattern: Часть имени файла шага.

    Returns:
        Пары путей (относительный путь совпадает).
    """
    out = []
    for path in sorted(before.rglob(f"*{pattern}*.jpg")):
        twin = after / path.relative_to(before)
        if twin.exists():
            out.append((path, twin))
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Склейка двух картинок одной высоты (меньшая добивается белым снизу) с полем между ними."""
    height = max(left.shape[0], right.shape[0])
    pad = [np.vstack([im, np.full((height - im.shape[0], im.shape[1], 3), 255, np.uint8)]) for im in (left, right)]
    return np.hstack([pad[0], np.full((height, GAP_PX, 3), 255, np.uint8), pad[1]])


def write_sheets(before: Path, after: Path, out_dir: Path, title: str) -> list[Path]:
    """Листы «было | стало» по всем парам шага ``08_rectify``.

    Args:
        before: Корень отладки «было».
        after: Корень отладки «стало».
        out_dir: Папка листов.
        title: Заголовок (набор).

    Returns:
        Пути листов.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for left_path, right_path in pairs(before, after):
        left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
        body = side_by_side(left, right)
        scale = min(1.0, SHEET_HEIGHT / body.shape[0])
        body = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rel = left_path.relative_to(before)
        picture = framed(body, [f"{title}: {rel}",
                                "слева — было (код базы: серии по подряд идущим строкам, правило 0.3 мм у края), "
                                "справа — стало (серии без отступов, фильтр псевдоглифов + CRAFT)"], [])  # fmt: skip
        target = out_dir / ("__".join(rel.parts))
        cv2.imwrite(str(target), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(target)
    return written


__all__ = ["pairs", "side_by_side", "write_sheets"]
