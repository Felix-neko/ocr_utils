"""Признаки кандидата line art для правила «пометка»: что осталось на вырезке после заливки слов — сколько пятен, какие они (тонкие, вытянутые, бледные), где лежат.

Залитая вырезка — та же, что видит второй проход DeepSeek: ``work/crops/pass2/<id>.png``, если второй проход был;
иначе она строится заново тем же ``pass2.fill_words`` по словам первого прохода. Дальше — как у ``verdict_pass2``:
без длинных линеек и крапин (``despeckle(strip_rules(...))``), только внутри рамки кандидата.

Меры компоненты (пятна связной краски):

* **толщина** — ``2 · площадь / периметр`` (для штриха любой формы — его ширина), мм;
* **длина** — ``периметр / 2``, мм;
* **вытянутость** — длина к толщине;
* **заполнение** — площадь к площади её повёрнутой рамки (``cv2.minAreaRect``): у скобки, дуги, галочки — мало;
* **серость** — медианная яркость пикселей компоненты на серой вырезке первого прохода: карандаш бледнее печати.

Пометка — это немного тонких вытянутых штрихов и мало краски; рисунок — много краски, толстые линии или
заполненные пятна; заставка рубрики — одна компонента, но толстая и заполненная (буква с палочками).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.line_art.deepseek.pass2 import despeckle, fill_words, strip_rules
from ocr_utils.page_layout.line_art.deepseek.rules import CROP_DPI, block_class, on_region
from research.margin_marks.sources import Candidate, DeepSeekOutput

MM_PER_PX = 25.4 / CROP_DPI
# Компонента меньше этой площади, мм², в признаки не идёт (крапины и так сняты ``despeckle``).
MIN_COMPONENT_MM2 = 0.5
# Компонента «длинная» — от этой длины, мм: мелкие недозалитые обрывки букв не считаются.
LONG_COMPONENT_MM = 5.0
# Прямые по осям: краска, переживающая открытие ядром такой длины по горизонтали или вертикали, мм.
STRAIGHT_MM = 3.0
# Штрих «линейный»: вытянутость не меньше и толщина не больше.
LINEAR_MIN_ELONGATION = 8.0
LINEAR_MAX_THICKNESS_MM = 0.8


@dataclass(frozen=True)
class MarkFeatures:
    """Признаки кандидата (одна строка ``features.csv``).

    Attributes:
        variant, id, page, outcome, classes, box_source, pass2_used: Кто и каков итог нынешнего решения.
        info_kind, info_sources: Что сказал классический детектор (``штрих``/``пятно``/…; ``ink``, ``surya``, ``tables``).
        pass1_nontext: Классы нетекстовых блоков DeepSeek первого прохода на рамке (через ``|``), пусто — только текст.
        filled_from: Откуда залитая вырезка: ``pass2`` (готовая) или ``refill`` (построена заново).
        components: Компонент краски после заливки слов, снятия линеек и крапин.
        ink_mm2: Краски в рамке, мм².
        ink_share: Доля краски в рамке.
        largest_share: Доля крупнейшей компоненты во всей краске.
        linear_share: Доля краски в «линейных» компонентах.
        thickness_max_mm: Наибольшая толщина компоненты.
        length_max_mm: Наибольшая длина компоненты.
        fill_max: Наибольшее заполнение повёрнутой рамки компонентой.
        gray_median: Медианная яркость краски остатка на серой вырезке первого прохода (0 — чёрный).
        edge_touch: Крупнейшая компонента прижата к левому или правому краю рамки (в пределах 3 мм).
        long_components: Компонент длиной от ``LONG_COMPONENT_MM``.
        long_thickness_max_mm: Наибольшая толщина среди длинных компонент.
        wobble_min_mm, wobble_max_mm: Наименьшая и наибольшая волнистость длинных компонент — отклонение пикселей от прямой,
            подогнанной по компоненте, p95 минус полтолщины, мм: печатная линейка около 0, карандаш — десятые доли мм.
        corners: Длинных компонент с углом (есть и горизонтальный, и вертикальный прямой пробег): рамка, коробка.
        straight_share: Доля краски остатка на прямых по осям (переживает открытие ядром ``STRAIGHT_MM``): печатные
            рамки, линейки, бланки — около 1, карандашная скобка или волнистое подчёркивание — меньше.
        box_w_mm, box_h_mm: Размер рамки кандидата, мм.
    """

    variant: str
    id: str
    page: str
    outcome: str
    classes: str
    box_source: str
    pass2_used: bool
    info_kind: str
    info_sources: str
    pass1_nontext: str
    filled_from: str
    components: int
    ink_mm2: float
    ink_share: float
    largest_share: float
    linear_share: float
    thickness_max_mm: float
    length_max_mm: float
    fill_max: float
    gray_median: float
    edge_touch: bool
    box_w_mm: float
    box_h_mm: float
    long_components: int = 0
    long_thickness_max_mm: float = 0.0
    straight_share: float = 0.0
    wobble_min_mm: float = 0.0
    wobble_max_mm: float = 0.0
    corners: int = 0


def filled_crop(
    candidate: Candidate, gray: np.ndarray, pass2: np.ndarray | None, deepseek: DeepSeekOutput
) -> tuple[np.ndarray, str]:
    """Залитая вырезка кандидата: готовая второго прохода или построенная заново по словам первого.

    Args:
        candidate: Кандидат.
        gray: Серая вырезка первого прохода.
        pass2: Готовая залитая вырезка или ``None``.
        deepseek: Ответы DeepSeek варианта (слова первого прохода).

    Returns:
        ``(вырезка: краска 0, бумага 255; откуда)``.
    """
    if pass2 is not None:
        return pass2, "pass2"
    binary, _ = fill_words(gray, candidate.crop.inner, deepseek.pass1_words.get(candidate.id, []))
    return binary, "refill"


def residue(binary: np.ndarray, inner) -> np.ndarray:
    """Остаток краски в рамке кандидата, как у ``verdict_pass2``: без длинных линеек и крапин; маска ``bool``."""
    stripped = despeckle(strip_rules(binary))
    mask = np.zeros(binary.shape, bool)
    x0, y0, x1, y1 = inner
    mask[y0:y1, x0:x1] = stripped[y0:y1, x0:x1] == 0
    return mask


def component_measures(mask: np.ndarray, gray: np.ndarray) -> list[dict]:
    """Меры компонент остатка (площадь, толщина, длина, вытянутость, заполнение, серость, рамка), мм.

    Args:
        mask: Остаток краски.
        gray: Серая вырезка первого прохода (того же размера).

    Returns:
        По компоненте словарь; компоненты меньше ``MIN_COMPONENT_MM2`` отброшены.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    mm2 = MM_PER_PX**2
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area * mm2 < MIN_COMPONENT_MM2:
            continue
        piece = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = max(1.0, sum(cv2.arcLength(c, True) for c in contours))
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.vstack(contours))
        thickness = 2.0 * area / perimeter
        length = perimeter / 2.0
        # Волнистость: отклонение пикселей от прямой, подогнанной по компоненте (PCA), p95 минус полтолщины.
        ys, xs = np.nonzero(piece)
        points = np.column_stack([xs, ys]).astype(np.float64)
        centred = points - points.mean(axis=0)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        across = np.abs(centred @ axes[1])
        wobble = max(0.0, float(np.percentile(across, 95)) - thickness / 2.0)
        # Угол: у компоненты есть и горизонтальный, и вертикальный прямой пробег от STRAIGHT_MM (угол рамки, коробка).
        run = max(3, int(round(STRAIGHT_MM / MM_PER_PX)))
        has_h = bool(cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)).any())
        has_v = bool(cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8)).any())
        out.append(
            {
                "area_mm2": area * mm2,
                "thickness_mm": thickness * MM_PER_PX,
                "length_mm": length * MM_PER_PX,
                "elongation": length / max(thickness, 1e-6),
                "fill": area / max(1.0, rw * rh),
                "wobble_mm": wobble * MM_PER_PX,
                "corner": has_h and has_v,
                "gray": float(np.median(gray[piece > 0])) if gray.shape == mask.shape else 0.0,
                "box": (
                    int(stats[index, 0]),
                    int(stats[index, 1]),
                    int(stats[index, 0] + stats[index, 2]),
                    int(stats[index, 1] + stats[index, 3]),
                ),
            }
        )
    return out


def straight_share(mask: np.ndarray) -> float:
    """Доля краски остатка на прямых по осям: открытие горизонтальным и вертикальным ядром ``STRAIGHT_MM``."""
    total = int(mask.sum())
    if not total:
        return 0.0
    ink = mask.astype(np.uint8)
    length = max(3, int(round(STRAIGHT_MM / MM_PER_PX)))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return float(np.count_nonzero(horizontal | vertical)) / total


def is_linear(component: dict) -> bool:
    """Компонента — тонкий вытянутый штрих."""
    return component["elongation"] >= LINEAR_MIN_ELONGATION and component["thickness_mm"] <= LINEAR_MAX_THICKNESS_MM


def features_of(
    candidate: Candidate, gray: np.ndarray, pass2: np.ndarray | None, deepseek: DeepSeekOutput
) -> MarkFeatures:
    """Признаки кандидата.

    Args:
        candidate: Кандидат.
        gray: Серая вырезка первого прохода.
        pass2: Готовая залитая вырезка второго прохода или ``None``.
        deepseek: Ответы DeepSeek варианта.

    Returns:
        :class:`MarkFeatures`.
    """
    binary, source = filled_crop(candidate, gray, pass2, deepseek)
    mask = residue(binary, candidate.crop.inner)
    components = component_measures(mask, gray)
    x0, y0, x1, y1 = candidate.crop.inner
    area = max(1, (x1 - x0) * (y1 - y0))
    ink = float(sum(c["area_mm2"] for c in components))
    largest = max(components, key=lambda c: c["area_mm2"], default=None)
    near = 3.0 / MM_PER_PX
    edge = bool(largest) and (largest["box"][0] - x0 <= near or x1 - largest["box"][2] <= near)
    blocks = [b for b in deepseek.pass1_markdown.get(candidate.id, []) if on_region(b, candidate.crop.inner)]
    nontext = sorted({block_class(b).value for b in blocks if block_class(b) is not None})
    return MarkFeatures(
        variant=candidate.variant.value,
        id=candidate.id,
        page=candidate.page,
        outcome=candidate.outcome,
        classes="|".join(sorted(candidate.classes)),
        box_source="|".join(sorted({o.get("box_source", "") for o in candidate.objects})),
        pass2_used=candidate.pass2_used,
        info_kind=str(candidate.info.get("kind", "")),
        info_sources="|".join(candidate.info.get("sources") or []),
        pass1_nontext="|".join(nontext),
        filled_from=source,
        components=len(components),
        ink_mm2=round(ink, 2),
        ink_share=round(ink / (area * MM_PER_PX**2), 4),
        largest_share=round(largest["area_mm2"] / ink, 3) if largest and ink > 0 else 0.0,
        linear_share=round(sum(c["area_mm2"] for c in components if is_linear(c)) / ink, 3) if ink > 0 else 0.0,
        thickness_max_mm=round(max((c["thickness_mm"] for c in components), default=0.0), 3),
        length_max_mm=round(max((c["length_mm"] for c in components), default=0.0), 2),
        fill_max=round(max((c["fill"] for c in components), default=0.0), 3),
        gray_median=round(float(np.median([c["gray"] for c in components])) if components else 0.0, 1),
        edge_touch=edge,
        box_w_mm=round((x1 - x0) * MM_PER_PX, 1),
        box_h_mm=round((y1 - y0) * MM_PER_PX, 1),
        long_components=sum(1 for c in components if c["length_mm"] >= LONG_COMPONENT_MM),
        long_thickness_max_mm=round(max((c["thickness_mm"] for c in components if c["length_mm"] >= LONG_COMPONENT_MM), default=0.0), 3),
        straight_share=round(straight_share(mask), 3),
        wobble_min_mm=round(min((c["wobble_mm"] for c in components if c["length_mm"] >= LONG_COMPONENT_MM), default=0.0), 3),
        wobble_max_mm=round(max((c["wobble_mm"] for c in components if c["length_mm"] >= LONG_COMPONENT_MM), default=0.0), 3),
        corners=sum(1 for c in components if c["length_mm"] >= LONG_COMPONENT_MM and c["corner"]),
    )


def as_row(features: MarkFeatures) -> dict:
    """Строка CSV."""
    return asdict(features)


__all__ = ["LONG_COMPONENT_MM", "MarkFeatures", "straight_share", "as_row", "component_measures", "features_of", "filled_crop", "is_linear", "residue"]
