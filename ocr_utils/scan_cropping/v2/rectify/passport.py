"""Паспорт страницы выхода v2 (``<кадр>_L.json`` рядом с маской ``<кадр>_L.fingers.png`` в папке ``--output-pages-meta``): исходный кадр, часть разворота, флаги, dpi, цветность, зоны закраса пальцев в координатах страницы; чтение паспорта для ``scan_markup``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.units import MM_PER_INCH, mm_to_px_float

# Версия схемы паспорта.
PASSPORT_VERSION = 1
# Суффиксы файлов паспорта и маски закраса к имени страницы без расширения.
PASSPORT_SUFFIX = ".json"
FINGERS_SUFFIX = ".fingers.png"
# Упрощение контура закраса (Дуглас — Пекер), мм: контур для CVAT, а не для пиксельной точности.
CONTOUR_EPSILON_MM = 0.5
# Зоны закраса меньше этой площади не пишутся в контуры, мм² (крошки на краю страницы после переноса).
MIN_ZONE_MM2 = 4.0


@dataclass(frozen=True)
class PagePassport:
    """Паспорт одной страницы выхода.

    Поля:
        version: Версия схемы (:data:`PASSPORT_VERSION`).
        source: Исходный кадр, путь относительно входа v2 (POSIX).
        part: Часть кадра: ``L``, ``R``, ``LR`` или ``P`` (одна страница кадра, без стороны разреза).
        flags: Флаги кадра из сайдкара (строки ``FrameFlag``).
        cover: Страница — обложка (флаг ``cover-*``).
        dpi: Настоящее разрешение страницы (по плотности бумаги), точек на дюйм.
        width: Ширина страницы, px.
        height: Высота страницы, px.
        tint: Доля тонированного фона (``colour_pages.tinted_share``; для ``LR`` — наибольшая из двух страниц).
        plain: Освещение и поле выпрямления применялись (не цветная, не тонированная, не обложка).
        finger_mm2: Площадь закраса пальцев на странице (закрашено и не возвращено copy back), мм².
        finger_zones: Контуры закраса, px страницы: список многоугольников ``[[x, y], ...]``.
    """

    source: str
    part: str
    flags: list[str]
    cover: bool
    dpi: float
    width: int
    height: int
    tint: float
    plain: bool
    finger_mm2: float = 0.0
    finger_zones: list[list[list[int]]] = field(default_factory=list)
    version: int = PASSPORT_VERSION


def zone_contours(mask: np.ndarray, dpi: float) -> list[list[list[int]]]:
    """Контуры зон маски (внешние, упрощённые) без крошек меньше :data:`MIN_ZONE_MM2`.

    Args:
        mask: Маска bool.
        dpi: Разрешение маски.

    Returns:
        Многоугольники ``[[x, y], ...]`` в px маски.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eps = mm_to_px_float(CONTOUR_EPSILON_MM, dpi)
    min_px = MIN_ZONE_MM2 * (dpi / MM_PER_INCH) ** 2
    zones = []
    for contour in contours:
        if cv2.contourArea(contour) < min_px:
            continue
        poly = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(poly) >= 3:
            zones.append(poly.astype(int).tolist())
    return zones


def finger_area_mm2(mask: np.ndarray, dpi: float) -> float:
    """Площадь маски, мм².

    Args:
        mask: Маска bool.
        dpi: Разрешение маски.

    Returns:
        Площадь в мм² бумаги.
    """
    return float(mask.sum()) / (dpi / MM_PER_INCH) ** 2


def write_passport(stem_path: Path, passport: PagePassport, fingers: np.ndarray | None) -> None:
    """Пишет паспорт ``<stem>.json`` и, если закрас есть, маску ``<stem>.fingers.png`` (1 бит: 0 / 255).

    Args:
        stem_path: Путь страницы в папке паспортов без расширения (``meta/1974/01/IMG_0078_L``).
        passport: Паспорт.
        fingers: Маска закраса размера страницы или None.
    """
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem_path.with_name(stem_path.name + PASSPORT_SUFFIX)
    json_path.write_text(json.dumps(asdict(passport), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    mask_path = stem_path.with_name(stem_path.name + FINGERS_SUFFIX)
    if fingers is not None and fingers.any():
        cv2.imwrite(str(mask_path), fingers.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_BILEVEL, 1])
    else:
        # Прежний прогон мог оставить маску: без закраса её быть не должно.
        mask_path.unlink(missing_ok=True)


def read_passport(path: Path) -> PagePassport:
    """Читает паспорт страницы.

    Args:
        path: Файл ``<stem>.json``.

    Returns:
        Паспорт.

    Raises:
        ValueError: версия схемы новее поддерживаемой.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("version", 0)) > PASSPORT_VERSION:
        raise ValueError(f"{path}: паспорт версии {data['version']} новее поддерживаемой {PASSPORT_VERSION}")
    return PagePassport(**data)
