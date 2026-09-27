"""Оверлеи стенда по трём папкам: формула была и surya нашла, формулы не было и surya нашла, формула была и surya не нашла."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research.surya_equations.evaluate import Box, box_quality, covers, union_box
from research.surya_equations.labels import PageLabels, Verdict
from research.surya_equations.sources import MM_PER_UNIT, EquationBox, load_gray

# Палитра (BGR, как в OpenCV; навык draw-overlay): эталон — «принято» зелёным, верный бокс surya —
# синим главной границы, ложный — «отвергнуто» красным, пропущенный блок — оранжевой заливкой.
COLOR_BLOCK = (0, 150, 0)
COLOR_BLOCK_INLINE = (120, 175, 120)  # строчная двухэтажная формула — справочно, в полноту не входит
COLOR_FOUND = (220, 90, 20)
COLOR_FALSE = (0, 0, 220)
COLOR_MISSED = (0, 165, 255)
COLOR_TEXT = (20, 20, 20)
MISSED_ALPHA = 0.4
FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Полоса целиком — не выше этого (px); вырезка — поле вокруг случая (мм) и потолок ширины (px).
PAGE_HEIGHT_PX = 1500
CROP_PAD_MM = 6.0
CROP_WIDTH_PX = 1400
VERDICT_CAPTION = {Verdict.NOT_FORMULA: "не формула", Verdict.FORMULA_TEXT_LINE: "строка абзаца"}


class Category(str, Enum):
    """Папка оверлея (имя папки — значение)."""

    FOUND = "формула была, surya нашла"
    FALSE = "формулы не было, surya нашла"
    MISSED = "формула была, surya не нашла"


@dataclass(frozen=True)
class Case:
    """Один случай на полосе: что показать крупно и в какую папку."""

    category: Category
    name: str  # суффикс файла вырезки: «E0» — блок эталона, «S2» — бокс surya
    region: Box  # что вырезать (кадр кэша)


def page_cases(labels: PageLabels, equations: tuple[EquationBox, ...]) -> list[Case]:
    """Случаи полосы по категориям: пойманные и пропущенные выносные блоки, ложные боксы surya.

    Args:
        labels: эталон и вердикты полосы.
        equations: боксы surya ``Equation``.

    Returns:
        Список :class:`Case`; пустой — полосе не место ни в одной папке.
    """
    boxes = [e.box for e in equations]
    cases = []
    for index, block in enumerate(labels.blocks):
        if block.inline:
            continue
        hits = [boxes[j] for j in range(len(boxes)) if covers(block.box, boxes[j])]
        if hits:
            cases.append(Case(Category.FOUND, f"E{index}", union_box([block.box, *hits])))
        else:
            cases.append(Case(Category.MISSED, f"E{index}", block.box))
    for j, box in enumerate(boxes):
        if labels.verdicts[j][0] != Verdict.FORMULA:
            cases.append(Case(Category.FALSE, f"S{j}", box))
    return cases


def _on_paper(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет, каким полупрозрачная заливка ложится на белую бумагу (образец в легенде — такой же)."""
    return tuple(int(round(alpha * own + (1.0 - alpha) * 255)) for own in color)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Шрифт с кириллицей (Hershey-шрифты OpenCV кириллицу не рисуют)."""
    return ImageFont.truetype(FONT_FILE, size)


def _texts(canvas: np.ndarray, items: list[tuple[tuple[int, int], str, tuple[int, int, int], int]]) -> np.ndarray:
    """Подписи на белой подложке.

    Args:
        canvas: BGR-картинка.
        items: ``((x, y) левого верхнего угла, текст, цвет BGR, кегль px)``.

    Returns:
        Новая BGR-картинка с подписями.
    """
    image = Image.fromarray(canvas[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    for (x, y), text, color, size in items:
        font = load_font(size)
        # Подпись не должна уходить за край холста: сдвиг внутрь по обеим осям.
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        x = int(min(max(2, x), image.width - (right - left) - 4))
        y = int(min(max(2, y), image.height - (bottom - top) - 4))
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text((x, y), text, font=font, fill=color[::-1])
    return np.asarray(image)[:, :, ::-1].copy()


LEGEND_ROWS = (
    ("эталон: выносная формула", COLOR_BLOCK, 1.0, False),
    ("эталон: строчная двухэтажная (справочно)", COLOR_BLOCK_INLINE, 1.0, False),
    ("surya Equation на формуле", COLOR_FOUND, 1.0, False),
    ("surya Equation мимо (не формула / строка абзаца)", COLOR_FALSE, 1.0, False),
    ("формула, которую surya не нашла", COLOR_MISSED, MISSED_ALPHA, True),
)


def _header(canvas: np.ndarray, title: str) -> np.ndarray:
    """Белая полоса над картинкой: заголовок и легенда (легенда не закрывает текст полосы).

    Образец заливки в легенде — с той же прозрачностью, что на странице (:func:`_on_paper`).

    Args:
        canvas: BGR-картинка с разметкой.
        title: заголовок.

    Returns:
        Новая BGR-картинка: полоса заголовка и легенды сверху, под ней ``canvas``.
    """
    size, step = 14, 20
    band = np.full((30 + step * len(LEGEND_ROWS), canvas.shape[1], 3), 255, np.uint8)
    for index, (_, color, alpha, filled) in enumerate(LEGEND_ROWS):
        y = 30 + step * index + step // 2
        if filled:
            cv2.rectangle(band, (8, y - 6), (32, y + 6), _on_paper(color, alpha), -1)
        else:
            cv2.rectangle(band, (8, y - 6), (32, y + 6), color, 2)
    items = [((8, 6), title, COLOR_TEXT, 16)]
    items += [((40, 30 + step * i + 2), text, COLOR_TEXT, size) for i, (text, *_) in enumerate(LEGEND_ROWS)]
    return np.vstack([_texts(band, items), canvas])


def _to_px(offset: tuple[float, float], k: float, box: Box) -> tuple[tuple[int, int], tuple[int, int]]:
    """Рамка из кадра кэша в пиксели холста.

    Args:
        offset: левый верхний угол вырезки в пикселях холста (уже с учётом зума).
        k: пикселей холста на единицу кадра.
        box: рамка в кадре кэша.

    Returns:
        Углы ``((x0, y0), (x1, y1))`` для ``cv2.rectangle``.
    """
    return (int(box[0] * k - offset[0]), int(box[1] * k - offset[1])), (
        int(box[2] * k - offset[0]),
        int(box[3] * k - offset[1]),
    )


def draw_view(
    gray: np.ndarray,
    frame_scale: float,
    region: Box,
    zoom: float,
    labels: PageLabels,
    equations: tuple[EquationBox, ...],
    title: str,
    line_px: int,
) -> np.ndarray:
    """Нарисовать область полосы со всеми сущностями.

    Args:
        gray: серый JPEG полосы.
        frame_scale: пикселей JPEG на единицу кадра кэша.
        region: вырезаемая область в кадре кэша (вся полоса — ``(0, 0, w, h)``).
        zoom: во сколько раз ужать/растянуть пиксели JPEG.
        labels: эталон и вердикты полосы.
        equations: боксы surya.
        title: заголовок над картинкой.
        line_px: толщина рамок.

    Returns:
        BGR-картинка с легендой и подписями.
    """
    x0, y0, x1, y1 = (int(v * frame_scale) for v in region)
    x0, y0 = max(0, x0), max(0, y0)
    crop = gray[y0 : min(gray.shape[0], y1), x0 : min(gray.shape[1], x1)]
    interpolation = cv2.INTER_AREA if zoom < 1 else cv2.INTER_CUBIC
    canvas = cv2.cvtColor(cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=interpolation), cv2.COLOR_GRAY2BGR)
    k = frame_scale * zoom
    offset = (x0 * zoom, y0 * zoom)  # левый верхний угол вырезки в пикселях холста

    boxes = [e.box for e in equations]
    captions = []
    # Пропущенные блоки — полупрозрачной заливкой под рамками, чтобы буквы читались.
    layer = canvas.copy()
    for block in labels.blocks:
        if not block.inline and not any(covers(block.box, box) for box in boxes):
            cv2.rectangle(layer, *_to_px(offset, k, block.box), COLOR_MISSED, -1)
    cv2.addWeighted(layer, MISSED_ALPHA, canvas, 1 - MISSED_ALPHA, 0, canvas)
    pad = line_px + 2
    for index, block in enumerate(labels.blocks):
        (a, b), (c, d) = _to_px(offset, k, block.box)
        color = COLOR_BLOCK_INLINE if block.inline else COLOR_BLOCK
        cv2.rectangle(canvas, (a - pad, b - pad), (c + pad, d + pad), color, line_px)
        hits = [boxes[j] for j in range(len(boxes)) if covers(block.box, boxes[j])]
        if block.inline:
            text = f"E{index} строчная"
        elif hits:
            quality = box_quality(block.box, union_box(hits))
            text = (
                f"E{index} IoU {quality['iou']:.2f}; запас, мм: "
                f"Л {quality['left_mm']:+.1f} В {quality['top_mm']:+.1f} "
                f"П {quality['right_mm']:+.1f} Н {quality['bottom_mm']:+.1f}"
            )
        else:
            text = f"E{index} пропуск"
        captions.append(((a, d + pad + 2), text, color, 13 if zoom < 1 else 16))
    for j, equation in enumerate(equations):
        verdict, _ = labels.verdicts[j]
        color = COLOR_FOUND if verdict == Verdict.FORMULA else COLOR_FALSE
        (a, b), (c, d) = _to_px(offset, k, equation.box)
        cv2.rectangle(canvas, (a, b), (c, d), color, line_px)
        text = f"S{j} {equation.confidence:.2f}" + (
            f" {VERDICT_CAPTION[verdict]}" if verdict in VERDICT_CAPTION else ""
        )
        captions.append(((a, b - (17 if zoom < 1 else 21)), text, color, 13 if zoom < 1 else 16))
    return _header(_texts(canvas, captions), title)


def write_page(
    sharpened_dir: Path,
    out_dir: Path,
    page: str,
    frame: tuple[int, int],
    labels: PageLabels,
    equations: tuple[EquationBox, ...],
) -> dict[str, int]:
    """Оверлеи одной полосы во все её папки: полоса целиком и вырезка на каждый случай.

    Args:
        sharpened_dir: ``SHARPENED_DIR``.
        out_dir: корень оверлеев; внутри — папки :class:`Category`.
        page: «год/выпуск/полоса».
        frame: ширина и высота кадра кэша.
        labels: эталон и вердикты полосы.
        equations: боксы surya.

    Returns:
        Категория → сколько случаев записано (для сводки у вызывающего).
    """
    cases = page_cases(labels, equations)
    if not cases:
        return {}
    gray = load_gray(sharpened_dir, page)
    frame_scale = gray.shape[1] / frame[0]
    stem = page.replace("/", "_")
    title = f"{page} · заострённый JPEG · surya Equation (кэш sharpened)"
    whole = draw_view(
        gray, frame_scale, (0, 0, frame[0], frame[1]), PAGE_HEIGHT_PX / gray.shape[0], labels, equations, title, 3
    )
    counts: dict[str, int] = {}
    for category in {case.category for case in cases}:
        folder = out_dir / category.value
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{stem}.jpg"), whole, [cv2.IMWRITE_JPEG_QUALITY, 88])
    for case in cases:
        pad = CROP_PAD_MM / MM_PER_UNIT
        region = (case.region[0] - pad, case.region[1] - pad, case.region[2] + pad, case.region[3] + pad)
        width_px = (region[2] - region[0]) * frame_scale
        zoom = min(1.0, CROP_WIDTH_PX / width_px)
        crop = draw_view(gray, frame_scale, region, zoom, labels, equations, f"{page} · {case.name}", 3)
        cv2.imwrite(
            str(out_dir / case.category.value / f"{stem}_{case.name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        counts[case.category.value] = counts.get(case.category.value, 0) + 1
    return counts
