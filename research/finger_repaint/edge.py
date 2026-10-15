"""Точный стенд осветления тени пальца: вход карты из кэша прогона v2 (``--save-before-shadow``), карта текущим кодом, меры каймы и пересвета по зонам, листы «до | было | стало»."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.finger_shadow import (
    FILL_REACH_MM,
    SHADOW_SCALE,
    FingerShadow,
    disk,
    estimate_finger_shadow,
    lighten_finger_shadow,
)

# Запас вокруг зоны на листе, px полного масштаба; ширина листа, px; зазор между панелями, px.
SHEET_PAD = 60
SHEET_WIDTH = 1500
PANEL_GAP = 12
# Расширение краски, кожи и печати при мере каймы, px полного масштаба: буквы copy back и inkback — не кайма.
INK_GROW_PX = 9


@dataclass(frozen=True)
class EdgeInputs:
    """Вход карты тени одного кадра из кэша прогона v2.

    Поля:
        key: Ключ кадра в кэше.
        image: Итог до осветления, BGR uint8.
        paint: Зона закраса пальца (bool).
        core: Ядро пальца (bool).
        book: Книга — бумага ∪ защита контента (bool).
        original: Пиксели оригинала в итоге (bool): вне зоны закраса и возвращённое copy back.
    """

    key: str
    image: np.ndarray
    paint: np.ndarray
    core: np.ndarray
    book: np.ndarray
    original: np.ndarray


def load_edge_inputs(cache_dir: Path) -> EdgeInputs | None:
    """Вход карты из папки кадра в кэше (``before_shadow.png`` и ``shadow_masks.npz``).

    Args:
        cache_dir: Папка кадра в ``--cache`` прогона v2 с ``--save-before-shadow``.

    Returns:
        Вход или None, если файлов нет (кадр без зоны тени или прогон без опции).
    """
    image_path, masks_path = cache_dir / "before_shadow.png", cache_dir / "shadow_masks.npz"
    if not image_path.exists() or not masks_path.exists():
        return None
    with np.load(masks_path) as m:
        masks = {name: m[name] for name in ("paint", "core", "book", "original")}
    return EdgeInputs(cache_dir.name, cv2.imread(str(image_path)), **masks)


def lightness(img: np.ndarray) -> np.ndarray:
    """Светлота L (0..100) картинки BGR (uint8 или float — с обрезкой до 0..255)."""
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    return lab[..., 0].astype(np.float32) * (100 / 255)


def zone_masks(
    shadow: FingerShadow, inputs: EdgeInputs, zone: np.ndarray, px_per_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Маски меры одной зоны на полном масштабе: окрестность (зона ∪ её полоса закраса) и бумага закраса LaMa в ней.

    Args:
        shadow: Карта (текущий код).
        inputs: Вход кадра.
        zone: Зона на масштабе карты (bool).
        px_per_mm: Плотность итога, px/мм.

    Returns:
        ``(окрестность, закрас LaMa без краски, кожи и печати)`` — bool полного масштаба.
    """
    h, w = inputs.image.shape[:2]
    mm = px_per_mm * SHADOW_SCALE
    fill = shadow.fill if shadow.fill is not None else np.zeros_like(shadow.area)
    near = disk(zone, FILL_REACH_MM * mm) & (zone | fill)
    near_full = cv2.resize(near.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    excluded = shadow.ink | shadow.content | shadow.skin
    excluded_full = cv2.resize(excluded.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    excluded_full = cv2.dilate(excluded_full, np.ones((INK_GROW_PX, INK_GROW_PX), np.uint8)) > 0
    lama = inputs.paint & ~inputs.original & near_full & ~excluded_full
    return near_full, lama


def zone_measures(
    result: np.ndarray, image: np.ndarray, ref_light: np.ndarray, near: np.ndarray, lama: np.ndarray
) -> dict:
    """Меры осветления одной зоны против опоры R.

    Args:
        result: Итог (BGR uint8).
        image: Итог до осветления (BGR uint8).
        ref_light: Светлота опоры R полного масштаба.
        near: Окрестность зоны (bool).
        lama: Бумага закраса LaMa в ней (bool).

    Returns:
        ``rim_p1`` — 1 % остатка L (итог − опора) на закрасе LaMa, кайма (чем ближе к 0, тем лучше, < 0 — темнее);
        ``over_p99`` — 99 % остатка в окрестности, пересвет; ``add_mean`` — средняя прибавка L в окрестности.
    """
    light = lightness(result)
    residual = light - ref_light
    return {
        "rim_p1": float(np.percentile(residual[lama], 1)) if lama.any() else 0.0,
        "over_p99": float(np.percentile(residual[near], 99)) if near.any() else 0.0,
        "add_mean": float((light - lightness(image))[near].mean()) if near.any() else 0.0,
    }


def zone_sheet(panels: list[np.ndarray], near: np.ndarray) -> np.ndarray:
    """Лист зоны: вырезки панелей в ряд (до | было | стало) вокруг окрестности, ширина не больше ``SHEET_WIDTH``.

    Args:
        panels: Картинки полного масштаба одного размера (BGR uint8).
        near: Окрестность зоны (bool).

    Returns:
        Лист BGR uint8.
    """
    h, w = near.shape
    ys, xs = np.nonzero(near)
    y0, y1 = max(ys.min() - SHEET_PAD, 0), min(ys.max() + SHEET_PAD, h)
    x0, x1 = max(xs.min() - SHEET_PAD, 0), min(xs.max() + SHEET_PAD, w)
    tiles = [np.pad(p[y0:y1, x0:x1], ((0, 0), (0, PANEL_GAP), (0, 0)), constant_values=255) for p in panels]
    sheet = np.hstack(tiles)
    scale = min(1.0, SHEET_WIDTH / sheet.shape[1])
    return cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def evaluate_frame(
    inputs: EdgeInputs, px_per_mm: float, baseline: np.ndarray | None, sheets_dir: Path | None
) -> list[dict]:
    """Карта текущим кодом на входе кадра; меры по зонам для «до», «было» (итог прежнего прогона) и «стало».

    Args:
        inputs: Вход кадра.
        px_per_mm: Плотность итога, px/мм.
        baseline: Итог прежнего прогона того же кадра (BGR uint8) или None.
        sheets_dir: Куда писать листы зон или None.

    Returns:
        Строки по зонам: ``key``, ``zone``, меры с префиксами ``before_``, ``baseline_`` (если есть), ``current_``.
    """
    shadow = estimate_finger_shadow(inputs.image, inputs.paint, inputs.core, inputs.book, inputs.original, px_per_mm)
    current = lighten_finger_shadow(inputs.image, shadow)
    if baseline is not None and baseline.shape != inputs.image.shape:
        baseline = None
    h, w = inputs.image.shape[:2]
    # Опора — текущая (с профилем у края листа): одна для всех вариантов.
    ref_light = lightness(cv2.resize(shadow.ref, (w, h), interpolation=cv2.INTER_LINEAR))
    variants = {"before": inputs.image, "current": current}
    if baseline is not None:
        variants["baseline"] = baseline
    count, labels = cv2.connectedComponents(shadow.area.astype(np.uint8))
    rows = []
    for k in range(1, count):
        near, lama = zone_masks(shadow, inputs, labels == k, px_per_mm)
        row: dict = {"key": inputs.key, "zone": k}
        for name, result in variants.items():
            row |= {f"{name}_{m}": v for m, v in zone_measures(result, inputs.image, ref_light, near, lama).items()}
        rows.append(row)
        if sheets_dir is not None:
            panels = [inputs.image] + ([baseline] if baseline is not None else []) + [current]
            cv2.imwrite(
                str(sheets_dir / f"{inputs.key}_z{k}.jpg"), zone_sheet(panels, near), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
    return rows


def baseline_path(output_root: Path, key: str) -> Path | None:
    """Итог прежнего прогона по ключу кадра (``a__b__c`` → ``a/b/c.<ext>``) или None.

    Args:
        output_root: Корень итогов прежнего прогона.
        key: Ключ кадра в кэше.

    Returns:
        Путь к итогу или None.
    """
    rel = Path(*key.split("__"))
    for ext in (".tiff", ".tif", ".png", ".jpg"):
        path = output_root / rel.with_suffix(ext)
        if path.exists():
            return path
    return None
