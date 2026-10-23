"""Отладочный оверлей кадра v2: закрас пальцев, зона поиска тени, copy back, блоки surya, защита, книга, бумага, трапеции страниц — слева, итог — справа; отдельная картинка тени пальца: область поиска | карта осветления."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from PIL import Image, ImageDraw

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, _font, framed
from ocr_utils.scan_cropping.finger_removal.text_protection import polygons_to_mask
from ocr_utils.scan_cropping.v2.finger_shadow import REF_RING_MM, SHADOW_REACH_MM, FingerShadow
from ocr_utils.scan_cropping.v2.page_pose import PagePose, PageQuad
from ocr_utils.scan_cropping.v2.spread_lines import EdgeKind, Line, SpreadLines

# Цвета (BGR, палитра проекта ``/draw-overlay``).
COLOUR_PAINT = (0, 165, 255)  # зона закраса пальца — оранжевый
COLOUR_CORE = (180, 105, 255)  # ядро пальца — розовый
COLOUR_BACK = (40, 170, 40)  # возвращено с оригинала (copy back surya) — зелёный
COLOUR_BOOK = (220, 90, 20)  # граница грубой маски книги — синий
COLOUR_PLAIN = (170, 170, 170)  # граница SAM 3 plain — серый, справочная
COLOUR_ADDED = (140, 140, 0)  # добавлено доводкой края и склейкой — бирюзовый
COLOUR_CUT = (0, 0, 220)  # срезано уточнением склейки — красный
COLOUR_PAPER = (0, 150, 0)  # граница итоговой бумаги — зелёный тёмный
COLOUR_BOX = (200, 60, 200)  # прямоугольник вырезки — фиолетовый
COLOUR_SURYA = (200, 140, 60)  # блок surya — вспомогательный (``stages.COLOUR_HINT``)
COLOUR_SURYA_LABEL = (150, 90, 30)  # подпись типа блока surya — тот же тон темнее, чтобы читался на бумаге
COLOUR_PROTECT = (0, 200, 230)  # добавлено защитой контента — жёлтый
COLOUR_SIDE_ADDED = (120, 200, 0)  # залито выравниванием боковой стороны — салатовый
COLOUR_SIDE_CUT = (60, 20, 160)  # срезано выравниванием боковой стороны — тёмно-красный
COLOUR_INK_ADDED = (255, 200, 0)  # добавлено вторым проходом SAM 3 (кадр без крупной краски) — голубой
COLOUR_POCKET = (60, 110, 230)  # добавлено как карман, листок, вклейка на странице — оранжево-красный
COLOUR_PALETTE = (230, 120, 255)  # добавлено у края цветной страницы по её палитре — сиреневый
COLOUR_SHADOW = (120, 20, 120)  # граница зоны поиска тени пальца — тёмно-фиолетовый
COLOUR_SKIN = (0, 0, 160)  # кожа в итоге (тень по ней не оценивается и не осветляется) — тёмно-красный
COLOUR_RING = (0, 220, 255)  # кольцо опоры тени на листе (pull-push R) — жёлтый
COLOUR_RING_OUT = (255, 200, 0)  # кольцо опоры за краем листа, на полосе закраса — голубой
COLOUR_RING_DROPPED = (200, 0, 200)  # отсеяно из кольца: чужой цвет, тёмный закрас за краем — пурпурный
COLOUR_PAPER_AS_IS = (80, 200, 80)  # «бумага как есть» B — пиксели, по которым она строится — зелёный
# Трапеции страниц пунктиром: плоско лежащая — зелёный «принято», приподнятая — розовый (красный занят «срезано»),
# опущенная (внешний край свисает, защита контента как у плоской) — бирюзовый.
POSE_COLOUR = {PagePose.FLAT: (0, 150, 0), PagePose.RAISED: (180, 105, 255), PagePose.LOWERED: (170, 150, 0)}
POSE_NAME = {PagePose.FLAT: "плоско", PagePose.RAISED: "приподнята", PagePose.LOWERED: "опущена"}
# Пунктир: штрих и пробел, px холста.
DASH_PX, GAP_PX = 14, 9
# Трапеции страниц — точками: шаг и радиус точки, px холста (масштаб сегментации).
DOT_STEP_PX, DOT_RADIUS_PX = 10, 2
# Прозрачность пунктира сырых блоков surya (без припусков).
RAW_BLOCK_ALPHA = 0.5
# Подписи типов блоков surya: кегль, px итоговой картинки; прозрачность букв и белой обводки под ними.
LABEL_SIZE = 15
LABEL_ALPHA = 0.8
LABEL_HALO_ALPHA = 0.7
# Отступ подписи внутрь блока от его верхнего левого угла, px итоговой картинки.
LABEL_INSET_PX = 3
# Кегль подписей трапеций на итоговой картинке оверлея, px.
CAPTION_SIZE = 18
# Прозрачность областей.
AREA_ALPHA = 0.45
# Ширина итоговой картинки оверлея, px.
OVERLAY_WIDTH = 2600
# Ширина картинки шага сегментации (``--debug-steps``), px.
STEP_WIDTH = 1800
# Цвета экземпляров-страниц SAM 3 по номеру метки (по кругу).
INSTANCE_COLOURS = ((0, 150, 0), (220, 90, 20), (0, 165, 255), (180, 105, 255), (200, 60, 200), (140, 140, 0))
COLOUR_BG = (120, 120, 120)  # фон (коврик, заливка от рамки кадра) — серый
COLOUR_HULLS = (200, 140, 60)  # оболочки своей печати — вспомогательный (``stages.COLOUR_HINT``)
COLOUR_SPINE = (255, 0, 140)  # прямая-разделитель страниц — фиолетовый
COLOUR_BOUND = (0, 255, 255)  # ограничитель внешнего края страницы — жёлтый
COLOUR_BOOK_LINE = (255, 170, 90)  # прямая края книги и её точки — светло-синий (тон грубой маски книги)
# Подгоночные области страниц на шаге 07: прежний сплайн (``page_boundary``) — оранжевый, справочно; новая область
# (``page_frames``: сплайн с узлами на изгибах, сторона у корешка — по линии корешка) — тёмно-красная.
COLOUR_OLD_SPLINE = (0, 165, 255)
COLOUR_FRAME = (0, 0, 160)
# Полоса-вклейка: маска — полупрозрачная оранжевая область, вспомогательные линии — розовый пунктир, линия корешка —
# тёмно-фиолетовая сплошная (разделитель v2 рядом — фиолетовый).
COLOUR_STRIP = (0, 165, 255)
COLOUR_AUX = (180, 105, 255)
COLOUR_GUTTER_SPINE = (120, 0, 80)
STRIP_ALPHA = 0.35
KNOT_RADIUS = 7
COLOUR_DROPPED = (0, 0, 220)  # точки края книги, отброшенные (у рамки, рука, палец) — красный
# Толщина прямых разворота на холсте масштаба сегментации, px.
LINE_PX = 3


class OverlayTag(StrEnum):
    """Тип отладочной картинки v2: своя подпапка ``--debug`` и постфикс имени — ``<тег>/<подпапки>/<кадр>_<тег>.jpg``."""

    # Основной оверлей: «кадр с разметкой | итог».
    MAIN = "main"
    # Тень пальца: зона поиска и карта осветления (только при ``--finger-shadow``).
    SHADOW = "shadow"
    # Вход surya: выпрямленный кадр с рамками блоков.
    SURYA = "surya"
    # Шаги сегментации бумаги (только при ``--debug-steps``), номер — порядок шага.
    BOOK = "01_book"  # фон, оболочки печати, грубая маска книги, вето фона
    SAM3 = "02_sam3"  # экземпляры-страницы SAM 3 (и второй проход по кадру без крупной краски)
    SNAP = "03_snap"  # SAM 3 plain и доводка края до фона
    GLUE = "04_glue"  # склейка страниц, уточнение склейки, карманы
    SIDES = "05_sides"  # второй проход SAM 3, кромка листа, палитра, выравнивание сторон, защита контента
    PAGES = "06_pages"  # итоговая бумага и трапеции страниц
    LINES = "07_lines"  # прямые разворота: края книги, разделитель страниц, ограничители внешних краёв


def overlay_path(debug_dir: Path, rel: Path, tag: OverlayTag) -> Path:
    """Путь отладочной картинки кадра в папке ``--debug``.

    Args:
        debug_dir: Корень отладочных картинок прогона (``--debug``).
        rel: Путь кадра относительно входной папки (подпапки повторяются внутри папки типа).
        tag: Тип картинки — имя подпапки и постфикс имени файла.

    Returns:
        ``debug_dir/<тег>/<подпапки>/<кадр>_<тег>.jpg``.
    """
    return debug_dir / tag.value / rel.parent / f"{rel.stem}_{tag.value}.jpg"


def _resize(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Булева маска → размер ``shape`` ближайшим соседом.

    Args:
        mask: Булева маска.
        shape: ``(H, W)``.

    Returns:
        Булева маска.
    """
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def _contour(canvas: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], width: int) -> None:
    """Рисует внешние контуры маски на холсте.

    Args:
        canvas: Холст BGR (меняется на месте — это локальный холст оверлея).
        mask: Булева маска размера холста.
        colour: Цвет BGR.
        width: Толщина, px.
    """
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, cs, -1, colour, width)


@dataclass(frozen=True)
class OverlayLayers:
    """Слои разметки кадра для оверлея.

    Поля:
        masks: ``pipeline.PaperMasks`` — бумага по шагам (масштаб сегментации).
        protected: Добавлено защитой контента (масштаб сегментации).
        fingers: Маски закраса (``paint``, ``core``) размера кадра.
        returned: Возвращённое copy back (bool) размера кадра.
        polys: Все блоки surya, px кадра.
        pages: Страницы с трапециями (px сегментации) и классом положения.
        box: Осевая рамка бумаги в px кадра или None.
        pad: Припуск блоков surya (защита контента и copy back), px кадра (по X, по Y): контур зоны с припусками.
        labels: Типы блоков surya по порядку ``polys`` или None (кэш ``layout.json`` версии 3 — без подписей).
        shadow_area: Зона поиска тени пальца (bool) размера кадра или None (``--no-finger-shadow``).
        lines: Прямые разворота (масштаб сегментации) или None (не две страницы).
    """

    masks: object
    protected: np.ndarray
    fingers: object
    returned: np.ndarray
    polys: list[np.ndarray]
    pages: list[PageQuad]
    box: tuple[int, int, int, int] | None
    pad: tuple[int, int] = (0, 0)
    labels: list[str] | None = None
    shadow_area: np.ndarray | None = None
    lines: SpreadLines | None = None


def _draw_line(canvas: np.ndarray, line: Line, colour: tuple[int, int, int], width: int, dashed: bool) -> None:
    """Отрезок прямой разворота по её пролёту, сплошной или пунктиром.

    Args:
        canvas: Холст масштаба сегментации (меняется на месте — локальный холст оверлея).
        line: Прямая.
        colour: Цвет BGR.
        width: Толщина, px.
        dashed: Пунктир (виртуальный ограничитель).
    """
    a, b = line.ends()
    if dashed:
        # Открытый отрезок пунктиром: штрих и пробел вдоль него (замкнутый ``_dashed_poly`` прошёл бы его дважды).
        length = float(np.linalg.norm(b - a))
        for t0 in np.arange(0.0, length, DASH_PX + GAP_PX):
            p0, p1 = a + (b - a) * (t0 / length), a + (b - a) * (min(t0 + DASH_PX, length) / length)
            cv2.line(
                canvas, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), colour, width, cv2.LINE_AA
            )
    else:
        cv2.line(canvas, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, width, cv2.LINE_AA)


def _draw_spread_lines(canvas: np.ndarray, lines: SpreadLines, points: bool) -> None:
    """Прямые разворота: края книги (тонко), разделитель, ограничители (виртуальный — пунктиром).

    Args:
        canvas: Холст масштаба сегментации (меняется на месте).
        lines: Прямые разворота.
        points: Рисовать ли точки края книги — вошедшие в прямую и отброшенные (картинка шага ``07_lines``).
    """
    for edge in (lines.book_left, lines.book_right):
        if points:
            for pts, colour in ((edge.points, COLOUR_BOOK_LINE), (edge.dropped, COLOUR_DROPPED)):
                for x, y in np.round(pts).astype(int):
                    cv2.circle(canvas, (x, y), 1, colour, -1)
        if edge.line is not None:
            _draw_line(canvas, edge.line, COLOUR_BOOK_LINE, 1, dashed=not edge.visible)
    _draw_line(canvas, lines.spine, COLOUR_SPINE, LINE_PX, dashed=False)
    for bound, kind in ((lines.left, lines.left_kind), (lines.right, lines.right_kind)):
        if bound is not None:
            _draw_line(canvas, bound, COLOUR_BOUND, LINE_PX, dashed=kind is EdgeKind.VIRTUAL)


def _line_captions(picture: np.ndarray, lines: SpreadLines, scale: float) -> np.ndarray:
    """Подписи наклонов прямых разворота у их нижних концов (шрифт DejaVu, белая подложка).

    Args:
        picture: Картинка BGR.
        lines: Прямые разворота (масштаб сегментации).
        scale: Масштаб картинки относительно масштаба сегментации.

    Returns:
        Картинка BGR с подписями.
    """
    image = Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _font(CAPTION_SIZE)
    kind_name = {EdgeKind.BOOK: "по книге", EdgeKind.VIRTUAL: "виртуальный", EdgeKind.NONE: "нет"}
    items = [(lines.spine, f"разделитель ({lines.method.value}) {lines.spine.tilt_deg:+.2f}°", COLOUR_SPINE)]
    for bound, kind in ((lines.left, lines.left_kind), (lines.right, lines.right_kind)):
        if bound is not None:
            items.append((bound, f"{kind_name[kind]} {bound.tilt_deg:+.2f}°", COLOUR_BOUND))
    for line, text, colour in items:
        x, y = line.ends()[1] * scale
        box = draw.textbbox((0, 0), text, font=font)
        org = (int(min(max(x - (box[2] - box[0]) / 2, 0), picture.shape[1] - (box[2] - box[0]) - 1)), int(y) + 6)
        draw.rectangle(draw.textbbox(org, text, font=font), fill=(255, 255, 255))
        b, g, r = colour
        # Жёлтый на белом не читается — подпись ограничителя тёмным тоном.
        fill = (r, g, b) if colour != COLOUR_BOUND else (150, 120, 0)
        draw.text(org, text, font=font, fill=fill)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


LINES_LEGEND = [
    LegendEntry("прямая-разделитель страниц", COLOUR_SPINE),
    LegendEntry("ограничитель внешнего края страницы: сплошной — по краю книги, пунктир — виртуальный", COLOUR_BOUND),
    LegendEntry("прямая края книги: сплошная — край в кадре, пунктир — за кадром", COLOUR_BOOK_LINE),
]


def _dashed_poly(canvas: np.ndarray, pts: np.ndarray, colour: tuple[int, int, int], width: int) -> None:
    """Замкнутый пунктирный многоугольник на холсте.

    Args:
        canvas: Холст BGR (локальный холст оверлея, меняется на месте).
        pts: Вершины (N, 2) в px холста.
        colour: Цвет BGR.
        width: Толщина, px.
    """
    for i in range(len(pts)):
        a, b = pts[i].astype(float), pts[(i + 1) % len(pts)].astype(float)
        length = float(np.linalg.norm(b - a))
        step = DASH_PX + GAP_PX
        for t0 in np.arange(0.0, length, step):
            t1 = min(t0 + DASH_PX, length)
            p0 = a + (b - a) * (t0 / max(length, 1e-9))
            p1 = a + (b - a) * (t1 / max(length, 1e-9))
            q0, q1 = tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int))
            cv2.line(canvas, q0, q1, colour, width, cv2.LINE_AA)


def _dotted_poly(canvas: np.ndarray, pts: np.ndarray, colour: tuple[int, int, int], radius: int) -> None:
    """Замкнутый многоугольник точками (справочная линия: тоньше и тише пунктира).

    Args:
        canvas: Холст BGR (локальный холст оверлея, меняется на месте).
        pts: Вершины (N, 2) в px холста.
        colour: Цвет BGR.
        radius: Радиус точки, px.
    """
    for i in range(len(pts)):
        a, b = pts[i].astype(float), pts[(i + 1) % len(pts)].astype(float)
        length = float(np.linalg.norm(b - a))
        for t0 in np.arange(0.0, length, DOT_STEP_PX):
            p = a + (b - a) * (t0 / max(length, 1e-9))
            cv2.circle(canvas, tuple(np.round(p).astype(int)), radius, colour, -1, cv2.LINE_AA)


def _draw_blocks(canvas: np.ndarray, polys: list[np.ndarray], scale: float, pad: tuple[int, int]) -> None:
    """Блоки surya: сырые полигоны — полупрозрачным пунктиром, зона с припусками защиты — тонким сплошным контуром.

    Args:
        canvas: Холст масштаба сегментации (меняется на месте).
        polys: Полигоны блоков, px кадра.
        scale: Масштаб холста относительно кадра.
        pad: Припуск, px кадра (по X, по Y).
    """
    if not polys:
        return
    small = [np.asarray(p, np.float32) * scale for p in polys]
    padded = polygons_to_mask(canvas.shape[:2], small, (max(1, round(pad[0] * scale)), max(1, round(pad[1] * scale))))
    _contour(canvas, padded > 0, COLOUR_SURYA, 1)
    layer = canvas.copy()
    for poly in small:
        _dashed_poly(layer, poly, COLOUR_SURYA, 1)
    cv2.addWeighted(layer, RAW_BLOCK_ALPHA, canvas, 1 - RAW_BLOCK_ALPHA, 0, canvas)


def block_labels(
    picture: np.ndarray, polys: list[np.ndarray], labels: list[str] | None, size: int = LABEL_SIZE
) -> np.ndarray:
    """Типы блоков surya полупрозрачным шрифтом DejaVu у верхнего левого угла блока, внутри него.

    Угол — вершина многоугольника с наименьшим ``x + y``: у наклонного четырёхугольника после обратной гомографии это
    его верхний левый угол. Буквы — тёмным тоном цвета блоков с прозрачностью ``LABEL_ALPHA``, под ними полупрозрачная
    белая обводка: подпись читается и на тексте страницы, и на тёмном фоне, а текст под ней не закрыт.

    Args:
        picture: Картинка BGR uint8.
        polys: Многоугольники блоков (N, 2) в px ``picture``.
        labels: Типы блоков по порядку ``polys``; None или пусто — картинка возвращается как есть.
        size: Кегль, px.

    Returns:
        Новая картинка BGR uint8 с подписями.
    """
    if not labels:
        return picture
    image = Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)).convert("RGBA")
    # Подписи — в отдельном прозрачном слое, затем смешиваются с картинкой.
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(size)
    b, g, r = COLOUR_SURYA_LABEL
    for poly, label in zip(polys, labels):
        pts = np.asarray(poly, np.float32).reshape(-1, 2)
        x, y = pts[np.argmin(pts.sum(axis=1))]
        org = (int(max(x, 0)) + LABEL_INSET_PX, int(max(y, 0)) + LABEL_INSET_PX)
        # Сначала обводка (полупрозрачная белая), потом сами буквы поверх неё.
        draw.text(org, label, font=font, fill=(255, 255, 255, int(255 * LABEL_HALO_ALPHA)), stroke_width=2,
                  stroke_fill=(255, 255, 255, int(255 * LABEL_HALO_ALPHA)))  # fmt: skip
        draw.text(org, label, font=font, fill=(r, g, b, int(255 * LABEL_ALPHA)))
    out = Image.alpha_composite(image, layer).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def _page_caption(page: PageQuad) -> str:
    """Подпись трапеции: класс и признаки.

    Args:
        page: Страница.

    Returns:
        Строка.
    """
    f = page.features
    return f"{POSE_NAME[page.pose]} ш{f.width_share:.2f} св{f.light_share:.2f} з{f.fill:.2f} к{f.keystone:.0f}° в{f.outer_ratio:.2f}"


def _draw_pages(canvas: np.ndarray, pages: list[PageQuad]) -> None:
    """Трапеции страниц тонкими точками цвета класса (справочные: главные теперь — подгоночные рамки сплайнами).

    Args:
        canvas: Холст масштаба сегментации (меняется на месте).
        pages: Страницы.
    """
    for page in pages:
        if page.quad is not None:
            _dotted_poly(canvas, page.quad, POSE_COLOUR[page.pose], DOT_RADIUS_PX)


def _page_captions(picture: np.ndarray, pages: list[PageQuad], scale: float) -> np.ndarray:
    """Подписи трапеций (класс и признаки) шрифтом DejaVu на белой подложке у левого верхнего угла трапеции.

    Args:
        picture: Итоговая картинка пары BGR.
        pages: Страницы.
        scale: Масштаб картинки относительно масштаба сегментации.

    Returns:
        Картинка BGR с подписями.
    """
    image = Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _font(CAPTION_SIZE)
    for page in pages:
        if page.quad is None:
            continue
        x, y = page.quad.min(axis=0) * scale
        org = (int(max(x, 0)) + 4, int(max(y, 0)) + 4)
        text = _page_caption(page)
        draw.rectangle(draw.textbbox(org, text, font=font), fill=(255, 255, 255))
        b, g, r = POSE_COLOUR[page.pose]
        draw.text(org, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_overlay(frame: np.ndarray, book: dict, layers: OverlayLayers, result: np.ndarray, title: str) -> np.ndarray:
    """Оверлей кадра: исходный кадр с разметкой шагов | итоговый прямоугольник; шапка и легенда — в полях.

    Args:
        frame: Исходный кадр BGR (с пальцами).
        book: Маски шага «фон и книга» (``book`` — грубая маска книги, масштаб сегментации).
        layers: Слои разметки.
        result: Итоговый прямоугольник BGR.
        title: Шапка.

    Returns:
        Картинка BGR.
    """
    masks = layers.masks
    h, w = shape = masks.paper.shape
    canvas = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    # Области — полупрозрачно в отдельном слое.
    layer = canvas.copy()
    layer[_resize(layers.fingers["paint"], shape)] = COLOUR_PAINT
    layer[_resize(layers.fingers["core"], shape)] = COLOUR_CORE
    layer[_resize(layers.returned, shape)] = COLOUR_BACK
    layer[masks.paper & ~masks.plain] = COLOUR_ADDED
    layer[masks.cut] = COLOUR_CUT
    layer[masks.ink_added] = COLOUR_INK_ADDED
    layer[masks.side_added] = COLOUR_SIDE_ADDED
    layer[masks.side_cut] = COLOUR_SIDE_CUT
    if masks.pocket_added is not None:
        layer[masks.pocket_added] = COLOUR_POCKET
    if masks.palette_added is not None:
        layer[masks.palette_added] = COLOUR_PALETTE
    layer[layers.protected] = COLOUR_PROTECT
    cv2.addWeighted(layer, AREA_ALPHA, canvas, 1 - AREA_ALPHA, 0, canvas)
    # Границы — тонкими линиями поверх.
    _draw_blocks(canvas, layers.polys, h / frame.shape[0], layers.pad)
    _contour(canvas, book["book"], COLOUR_BOOK, 2)
    _contour(canvas, masks.plain, COLOUR_PLAIN, 1)
    _contour(canvas, masks.paper | layers.protected, COLOUR_PAPER, 2)
    if layers.shadow_area is not None:
        _contour(canvas, _resize(layers.shadow_area, shape), COLOUR_SHADOW, 3)
    if layers.box is not None:
        sy, sx = h / frame.shape[0], w / frame.shape[1]
        x0, y0, x1, y1 = layers.box
        cv2.rectangle(canvas, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)), COLOUR_BOX, 2)
    _draw_pages(canvas, layers.pages)
    if layers.lines is not None:
        _draw_spread_lines(canvas, layers.lines, points=False)
    # Итог справа той же высоты, белый зазор между.
    rs = h / result.shape[0]
    right = cv2.resize(result, (max(1, int(result.shape[1] * rs)), h), interpolation=cv2.INTER_AREA)
    pair = np.hstack([canvas, np.full((h, 20, 3), 255, np.uint8), right])
    s = OVERLAY_WIDTH / pair.shape[1]
    pair = cv2.resize(pair, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pair = _page_captions(pair, layers.pages, s)
    if layers.lines is not None:
        pair = _line_captions(pair, layers.lines, s)
    # Типы блоков surya — у рамок без припусков (после обратного переноса с выпрямленного кадра), на итоговом масштабе.
    to_pair = h / frame.shape[0] * s
    pair = block_labels(pair, [np.asarray(p, np.float32) * to_pair for p in layers.polys], layers.labels)
    legend = [
        LegendEntry(
            "зона закраса пальца (LaMa): ядро + асимметричная дилатация",
            COLOUR_PAINT,
            alpha=AREA_ALPHA,
            style=SampleStyle.BOX,
        ),
        LegendEntry("ядро пальца (зона пальца детектора)", COLOUR_CORE, alpha=AREA_ALPHA, style=SampleStyle.BOX),
        LegendEntry(f"зона поиска тени пальца (ядро + {SHADOW_REACH_MM:g} мм, в книге)", COLOUR_SHADOW),
        LegendEntry(
            "возвращено с оригинала: блоки surya внутри книги", COLOUR_BACK, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry(
            "добавлено к SAM 3: доводка края до фона и склейка", COLOUR_ADDED, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry("срезано уточнением склейки (не бумага)", COLOUR_CUT, alpha=AREA_ALPHA, style=SampleStyle.BOX),
        LegendEntry(
            "залито выравниванием боковой стороны (яма)", COLOUR_SIDE_ADDED, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry(
            "срезано выравниванием боковой стороны (выброс)", COLOUR_SIDE_CUT, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry(
            "добавлено как карман, листок или вклейка на странице",
            COLOUR_POCKET,
            alpha=AREA_ALPHA,
            style=SampleStyle.BOX,
        ),
        LegendEntry(
            "добавлено у края цветной страницы по её палитре", COLOUR_PALETTE, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry(
            "добавлено вторым проходом SAM 3 (кадр без крупной краски)",
            COLOUR_INK_ADDED,
            alpha=AREA_ALPHA,
            style=SampleStyle.BOX,
        ),
        LegendEntry(
            "добавлено защитой контента (surya плоских страниц в книге)",
            COLOUR_PROTECT,
            alpha=AREA_ALPHA,
            style=SampleStyle.BOX,
        ),
        LegendEntry("блок surya с припусками (защита контента и copy back)", COLOUR_SURYA),
        LegendEntry("блок surya без припусков (пунктир)", COLOUR_SURYA, alpha=RAW_BLOCK_ALPHA),
        LegendEntry(
            "тип блока surya у его угла (Text — текст, SectionHeader — заголовок, Picture — картинка…)",
            COLOUR_SURYA_LABEL,
            alpha=LABEL_ALPHA,
        ),
        LegendEntry("грубая маска книги (не фон)", COLOUR_BOOK),
        LegendEntry("бумага SAM 3 (plain)", COLOUR_PLAIN),
        LegendEntry("итоговая бумага (с защитой)", COLOUR_PAPER),
        LegendEntry("прямоугольник вырезки", COLOUR_BOX),
        LegendEntry("трапеция (точки): страница лежит плоско", POSE_COLOUR[PagePose.FLAT]),
        LegendEntry("трапеция (точки): страница приподнята", POSE_COLOUR[PagePose.RAISED]),
        LegendEntry("трапеция (точки): страница опущена (внешний край свисает)", POSE_COLOUR[PagePose.LOWERED]),
        *LINES_LEGEND,
    ]
    sub = (
        "слева — кадр с разметкой шагов (у трапеций: ш — ширина к соседней, св — свет, з — заполнение, к — сход "
        "сторон), справа — итог"
    )
    return framed(pair, [title, sub], legend)


@dataclass(frozen=True)
class StepInputs:
    """Всё, по чему рисуются картинки шагов сегментации кадра (масштаб сегментации).

    Поля:
        img: Кадр без пальцев (``seg.png``), BGR.
        book_raw: Маски шага «фон и книга» до вето фона (``book.npz``: ``content``, ``bg``, ``book``, ``hulls``).
        book: Маски, которыми считалась бумага (после вето фона или те же ``book_raw``).
        sam3: Выход SAM 3 (``raw``, ``inst``; при втором проходе — ``raw_noink``, ``inst_noink``).
        masks: ``pipeline.PaperMasks`` — бумага по шагам.
        protected: Добавлено защитой контента (bool).
        pages: Страницы с трапециями и классом положения.
        lines: Прямые разворота или None (не две страницы).
        fit: Подгонка разворота (``pipeline.SpreadFit``: полоса, линия корешка, прежние рамки и подгоночные области)
            или None.
    """

    img: np.ndarray
    book_raw: dict
    book: dict
    sam3: dict
    masks: object
    protected: np.ndarray
    pages: list[PageQuad]
    lines: SpreadLines | None = None
    fit: object | None = None


def _area(entry: str, colour: tuple[int, int, int]) -> LegendEntry:
    """Строка легенды для полупрозрачной области цвета ``colour`` (прозрачность ``AREA_ALPHA``)."""
    return LegendEntry(entry, colour, alpha=AREA_ALPHA, style=SampleStyle.BOX)


def _fill_areas(canvas: np.ndarray, areas: list[tuple[np.ndarray | None, tuple[int, int, int]]]) -> np.ndarray:
    """Полупрозрачные области по порядку (поздняя перекрывает раннюю) поверх копии холста.

    Args:
        canvas: Холст BGR.
        areas: Пары ``(маска bool или None, цвет)``; None пропускается.

    Returns:
        Новый холст BGR.
    """
    out = canvas.copy()
    layer = canvas.copy()
    for mask, colour in areas:
        if mask is not None:
            layer[mask] = colour
    cv2.addWeighted(layer, AREA_ALPHA, out, 1 - AREA_ALPHA, 0, out)
    return out


def _step_frame(canvas: np.ndarray, title: str, sub: str, legend: list[LegendEntry]) -> np.ndarray:
    """Холст шага → ширина ``STEP_WIDTH``, шапка и легенда в полях.

    Args:
        canvas: Холст масштаба сегментации с разметкой.
        title: Первая строка шапки (кадр, шаг).
        sub: Вторая строка шапки (что на картинке, числа шага).
        legend: Легенда.

    Returns:
        Картинка BGR.
    """
    s = STEP_WIDTH / canvas.shape[1]
    small = cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return framed(small, [title, sub], legend)


def _share(mask: np.ndarray, of: np.ndarray) -> str:
    """Площадь маски в процентах от площади ``of`` (строка для шапки)."""
    return f"{100 * mask.sum() / max(int(of.sum()), 1):.2f} %"


def draw_steps(inputs: StepInputs, title: str) -> dict[OverlayTag, np.ndarray]:
    """Картинки шагов сегментации бумаги кадра: по одной на шаг, каждая — в свою подпапку ``--debug``.

    Шаги и что на них: 01 — фон, оболочки печати, грубая маска книги и что поменяло вето фона; 02 — экземпляры-страницы
    SAM 3 (второй проход — контуром); 03 — SAM 3 ``plain`` и доводка края до фона; 04 — склейка страниц, срезанное
    уточнением, карманы; 05 — добавки второго прохода, выравнивание сторон, палитра, защита контента; 06 — итоговая
    бумага и трапеции страниц. Подложка — кадр без пальцев на масштабе сегментации.

    Args:
        inputs: Маски шагов кадра.
        title: Начало шапки (кадр).

    Returns:
        Словарь ``тег → картинка BGR``.
    """
    img, book, raw_book, sam3, masks = inputs.img, inputs.book, inputs.book_raw, inputs.sam3, inputs.masks
    out: dict[OverlayTag, np.ndarray] = {}

    # 01: фон, оболочки печати, книга; вето фона — что оно вернуло книге и что убрало.
    vetoed = book["book"] & ~raw_book["book"]
    unvetoed = raw_book["book"] & ~book["book"]
    canvas = _fill_areas(img, [(book["bg"], COLOUR_BG), (vetoed, COLOUR_ADDED), (unvetoed, COLOUR_CUT)])
    _contour(canvas, book["hulls"], COLOUR_HULLS, 1)
    _contour(canvas, book["book"], COLOUR_BOOK, 2)
    out[OverlayTag.BOOK] = _step_frame(
        canvas,
        f"{title} — шаг 01: фон и грубая маска книги",
        f"фон {_share(book['bg'], np.ones_like(book['bg']))} кадра; вето вернуло книге {_share(vetoed, book['book'])}"
        f", убрало {_share(unvetoed, book['book'])} книги",
        [
            _area("фон (заливка от рамки кадра, после вето)", COLOUR_BG),
            _area("вето фона: возвращено книге (SAM 3 видит страницу, SAM 2 — не фон)", COLOUR_ADDED),
            _area("вето фона: убрано из книги", COLOUR_CUT),
            LegendEntry("оболочки своей печати", COLOUR_HULLS),
            LegendEntry("грубая маска книги (не фон)", COLOUR_BOOK),
        ],
    )

    # 02: экземпляры-страницы SAM 3 цветом по номеру; второй проход — контуром.
    inst = sam3["inst"]
    labels = [int(v) for v in np.unique(inst) if v > 0]
    canvas = _fill_areas(img, [(inst == v, INSTANCE_COLOURS[(v - 1) % len(INSTANCE_COLOURS)]) for v in labels])
    _contour(canvas, sam3["raw"], COLOUR_PLAIN, 1)
    legend = [
        _area(f"экземпляр-страница SAM 3 № {v}", INSTANCE_COLOURS[(v - 1) % len(INSTANCE_COLOURS)]) for v in labels
    ]
    legend.append(LegendEntry("сырая маска SAM 3 (объединение экземпляров)", COLOUR_PLAIN))
    sub = f"экземпляров-страниц: {len(labels)}"
    if "inst_noink" in sam3:
        noink = [int(v) for v in np.unique(sam3["inst_noink"]) if v > 0]
        for v in noink:
            _contour(canvas, sam3["inst_noink"] == v, COLOUR_INK_ADDED, 2)
        legend.append(LegendEntry("экземпляр второго прохода SAM 3 (кадр без крупной краски)", COLOUR_INK_ADDED))
        sub += f"; во втором проходе: {len(noink)}"
    out[OverlayTag.SAM3] = _step_frame(canvas, f"{title} — шаг 02: SAM 3", sub, legend)

    # 03: plain (постобработка SAM 3 внутри книги) и доводка края до фона.
    snap_added = masks.snapped & ~masks.plain
    snap_cut = masks.plain & ~masks.snapped
    canvas = _fill_areas(img, [(masks.plain, COLOUR_PAPER), (snap_added, COLOUR_ADDED), (snap_cut, COLOUR_CUT)])
    _contour(canvas, book["book"], COLOUR_BOOK, 1)
    out[OverlayTag.SNAP] = _step_frame(
        canvas,
        f"{title} — шаг 03: SAM 3 plain и доводка края до фона (snap_bg)",
        f"доводка добавила {_share(snap_added, masks.plain)}, убрала {_share(snap_cut, masks.plain)} plain",
        [
            _area("бумага SAM 3 plain", COLOUR_PAPER),
            _area("добавлено доводкой края до фона", COLOUR_ADDED),
            _area("убрано доводкой края", COLOUR_CUT),
            LegendEntry("грубая маска книги", COLOUR_BOOK),
        ],
    )

    # 04: склейка страниц (мост между экземплярами), уточнение склейки по цвету, карманы.
    glue_added = masks.glued & ~masks.snapped
    pocket = masks.pocket_added
    canvas = _fill_areas(img, [(glue_added, COLOUR_ADDED), (masks.cut, COLOUR_CUT), (pocket, COLOUR_POCKET)])
    _contour(canvas, masks.snapped, COLOUR_PLAIN, 1)
    refined = (masks.glued & ~masks.cut) | (pocket if pocket is not None else False)
    _contour(canvas, refined, COLOUR_PAPER, 2)
    sub = f"склейка добавила {_share(glue_added, masks.snapped)}, уточнение срезало {_share(masks.cut, masks.glued)}"
    if pocket is not None:
        sub += f", карманы {_share(pocket, masks.snapped)}"
    out[OverlayTag.GLUE] = _step_frame(
        canvas,
        f"{title} — шаг 04: склейка страниц, уточнение склейки, карманы",
        sub,
        [
            _area("добавлено склейкой страниц (hull bridge)", COLOUR_ADDED),
            _area("срезано уточнением склейки по цвету (не бумага)", COLOUR_CUT),
            _area("добавлено как карман, листок или вклейка", COLOUR_POCKET),
            LegendEntry("бумага после доводки края (вход шага)", COLOUR_PLAIN),
            LegendEntry("бумага после уточнения и карманов (выход шага)", COLOUR_PAPER),
        ],
    )

    # 05: второй проход SAM 3, кромка листа и выравнивание сторон, палитра цветных страниц, защита контента.
    canvas = _fill_areas(
        img,
        [
            (masks.ink_added, COLOUR_INK_ADDED),
            (masks.side_added, COLOUR_SIDE_ADDED),
            (masks.side_cut, COLOUR_SIDE_CUT),
            (masks.palette_added, COLOUR_PALETTE),
            (inputs.protected, COLOUR_PROTECT),
        ],
    )
    _contour(canvas, refined, COLOUR_PLAIN, 1)
    _contour(canvas, masks.paper | inputs.protected, COLOUR_PAPER, 2)
    out[OverlayTag.SIDES] = _step_frame(
        canvas,
        f"{title} — шаг 05: второй проход SAM 3, стороны листа, защита контента",
        f"второй проход {_share(masks.ink_added, masks.paper)}, стороны +{_share(masks.side_added, masks.paper)} "
        f"−{_share(masks.side_cut, masks.paper)}, защита {_share(inputs.protected, masks.paper)} итоговой бумаги",
        [
            _area("добавлено вторым проходом SAM 3 (кадр без крупной краски)", COLOUR_INK_ADDED),
            _area("залито кромкой листа и выравниванием стороны (яма)", COLOUR_SIDE_ADDED),
            _area("срезано выравниванием стороны (выброс)", COLOUR_SIDE_CUT),
            _area("добавлено у края цветной страницы по палитре", COLOUR_PALETTE),
            _area("добавлено защитой контента (surya плоских страниц)", COLOUR_PROTECT),
            LegendEntry("бумага после шага 04 (вход шага)", COLOUR_PLAIN),
            LegendEntry("итоговая бумага (с защитой)", COLOUR_PAPER),
        ],
    )

    # 06: итоговая бумага и трапеции страниц с подписями признаков.
    canvas = _fill_areas(img, [(masks.paper | inputs.protected, COLOUR_PAPER)])
    _draw_pages(canvas, inputs.pages)
    # Подписи трапеций — на уменьшенном холсте, до приклейки шапки и легенды.
    s = STEP_WIDTH / canvas.shape[1]
    small = _page_captions(cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), inputs.pages, s)
    picture = framed(
        small,
        [
            f"{title} — шаг 06: итоговая бумага и трапеции страниц",
            "у трапеций: ш — ширина к соседней, св — свет, з — заполнение, к — сход сторон",
        ],
        [
            _area("итоговая бумага (с защитой)", COLOUR_PAPER),
            LegendEntry("трапеция (точки): страница лежит плоско", POSE_COLOUR[PagePose.FLAT]),
            LegendEntry("трапеция (точки): страница приподнята", POSE_COLOUR[PagePose.RAISED]),
            LegendEntry("трапеция (точки): страница опущена (внешний край свисает)", POSE_COLOUR[PagePose.LOWERED]),
        ],
    )
    out[OverlayTag.PAGES] = picture
    if inputs.lines is not None:
        out[OverlayTag.LINES] = draw_lines_step(img, inputs.pages, inputs.lines, book["book"], title, inputs.fit)
    return out


def _edge_place(edge) -> str:
    """Подпись видимости края книги: «в кадре» или «за кадром» с долей строк у рамки."""
    return "в кадре" if edge.visible else f"за кадром ({100 * edge.border_share:.0f} % строк у рамки)"


def draw_lines_step(
    img: np.ndarray, pages: list[PageQuad], lines: SpreadLines, book: np.ndarray, title: str, fit: object | None = None
) -> np.ndarray:
    """Картинка шага ``07_lines``: края книги с точками, разделитель страниц, ограничители, трапеции, полоса-вклейка с
    вспомогательными линиями, линия корешка и подгоночные области страниц — для сверки.

    Args:
        img: Кадр масштаба сегментации, BGR.
        pages: Страницы с трапециями.
        lines: Прямые разворота.
        book: Грубая маска книги (контур).
        title: Начало шапки (кадр).
        fit: Подгонка разворота (``pipeline.SpreadFit``) или None (не рисуется).

    Returns:
        Картинка BGR.
    """
    canvas = img.copy()
    strip = getattr(fit, "strip", None)
    if strip is not None:
        layer = canvas.copy()
        layer[strip.mask] = COLOUR_STRIP
        cv2.addWeighted(layer, STRIP_ALPHA, canvas, 1 - STRIP_ALPHA, 0, canvas)
    _contour(canvas, book, COLOUR_BOOK, 1)
    _draw_pages(canvas, pages)
    if fit is not None:
        for boundary in fit.old:
            cv2.polylines(
                canvas, [np.round(boundary.polygon()).astype(np.int32)], True, COLOUR_OLD_SPLINE, 2, cv2.LINE_AA
            )
        for frame in fit.frames:
            cv2.polylines(
                canvas, [np.round(frame.polygon()).astype(np.int32)], True, COLOUR_FRAME, LINE_PX, cv2.LINE_AA
            )
            for x, y in np.round(frame.knots).astype(int):
                cv2.circle(canvas, (int(x), int(y)), KNOT_RADIUS, COLOUR_FRAME, 2, cv2.LINE_AA)
    _draw_spread_lines(canvas, lines, points=True)
    if strip is not None:
        _draw_line(canvas, strip.left, COLOUR_AUX, LINE_PX, dashed=True)
        _draw_line(canvas, strip.right, COLOUR_AUX, LINE_PX, dashed=True)
        _draw_line(canvas, fit.spine, COLOUR_GUTTER_SPINE, LINE_PX, dashed=False)
    s = STEP_WIDTH / canvas.shape[1]
    small = _line_captions(cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), lines, s)

    sub = (f"край книги слева {_edge_place(lines.book_left)}, справа {_edge_place(lines.book_right)}; наклоны: книга "
           + " / ".join("—" if e.line is None else f"{e.line.tilt_deg:+.2f}°" for e in (lines.book_left, lines.book_right))
           + f", разделитель {lines.spine.tilt_deg:+.2f}°")  # fmt: skip
    head = [f"{title} — шаг 07: прямые разворота и подгоночные области страниц", sub]
    if fit is not None:
        gutter = (f"полоса-вклейка {strip.width_mm:.1f} мм, края {strip.left.tilt_deg:+.2f}° / {strip.right.tilt_deg:+.2f}°"
                  if strip is not None else "полосы-вклейки нет")  # fmt: skip
        head.append(f"щель SAM 3 у корешка {fit.gap_mm:.1f} мм — {gutter}; корешок {fit.spine.tilt_deg:+.2f}°: "
                    f"{fit.spine_source}")  # fmt: skip
    legend = [
        *LINES_LEGEND,
        LegendEntry("точки края книги, по которым строится прямая", COLOUR_BOOK_LINE),
        LegendEntry("точки края книги отброшены: у рамки кадра, рядом рука или палец", COLOUR_DROPPED),
        LegendEntry("грубая маска книги", COLOUR_BOOK),
        LegendEntry("трапеция (точки): страница лежит плоско", POSE_COLOUR[PagePose.FLAT]),
        LegendEntry("трапеция (точки): страница приподнята", POSE_COLOUR[PagePose.RAISED]),
        LegendEntry("трапеция (точки): страница опущена (внешний край свисает)", POSE_COLOUR[PagePose.LOWERED]),
    ]
    if fit is not None:
        legend += [
            LegendEntry(
                "подгоночная область: внешняя сторона, верх и низ — сплайн с узлами на изгибах, у корешка — по линии "
                "корешка; у полосы — по маске до зоны полосы, дальше касательной до линии корешка",
                COLOUR_FRAME,
            ),
            LegendEntry("кружок — добавленный узел сплайна на изгибе", COLOUR_FRAME),
            LegendEntry("прежний сплайн рамки страницы (для сравнения)", COLOUR_OLD_SPLINE),
            LegendEntry("полоса-вклейка (маска)", COLOUR_STRIP, alpha=STRIP_ALPHA, style=SampleStyle.BOX),
            LegendEntry("вспомогательные линии: края полосы", COLOUR_AUX, style=SampleStyle.DASHED),
            LegendEntry(
                "линия корешка у полосы: через её середину, наклон — по краям полосы или разделителю (см. шапку)",
                COLOUR_GUTTER_SPINE,
            ),
        ]
    return framed(small, head, legend)


# Карта осветления на картинке тени: шкала прибавки, уровни (насыщение цвета) и наименьшая видимая прибавка.
SHADOW_HEAT_MAX = 30.0
SHADOW_HEAT_MIN = 1.0
# Высота панели на картинке тени, px; поле вокруг области поиска, доля её рамки.
SHADOW_PANEL_H = 700
SHADOW_MARGIN = 0.15


def _heat_colour(level: float) -> tuple[int, int, int]:
    """Цвет BGR прибавки ``level`` уровней на шкале ``COLORMAP_TURBO`` от 0 до ``SHADOW_HEAT_MAX``."""
    v = np.uint8(np.clip(level / SHADOW_HEAT_MAX * 255, 0, 255))
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[v]], np.uint8), cv2.COLORMAP_TURBO)[0, 0])


def _shadow_panels(
    before: np.ndarray, shadow: FingerShadow, paint: np.ndarray, core: np.ndarray, rows: slice, cols: slice
) -> np.ndarray:
    """Панели одной рамки: итог до осветления с зонами | карта прибавки | опора (кольцо R, отсеянное, пиксели B).

    Args:
        before: Итог до осветления на масштабе карты (¼), BGR.
        shadow: Карта тени.
        paint: Зона закраса на масштабе карты (bool).
        core: Ядро пальца на масштабе карты (bool).
        rows: Строки рамки.
        cols: Столбцы рамки.

    Returns:
        Панели бок о бок, BGR, высота ``SHADOW_PANEL_H``.
    """
    crop = before[rows, cols]
    k = SHADOW_PANEL_H / crop.shape[0]
    size = (max(1, round(crop.shape[1] * k)), SHADOW_PANEL_H)
    big = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    shape = (size[1], size[0])
    zones = big.copy()
    layer = zones.copy()
    layer[_resize(paint[rows, cols], shape)] = COLOUR_PAINT
    layer[_resize(core[rows, cols], shape)] = COLOUR_CORE
    zones = cv2.addWeighted(layer, AREA_ALPHA, zones, 1 - AREA_ALPHA, 0)
    _contour(zones, _resize(shadow.skin[rows, cols], shape), COLOUR_SKIN, 2)
    _contour(zones, _resize(shadow.area[rows, cols], shape), COLOUR_SHADOW, 3)
    # Карта: средняя по каналам прибавка, цвет по шкале, прозрачность растёт с прибавкой (нули — серый итог).
    level = cv2.resize(shadow.delta[rows, cols].mean(axis=2), size, interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.clip(level / SHADOW_HEAT_MAX * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    grey = cv2.cvtColor(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    alpha = (np.clip(level / (4 * SHADOW_HEAT_MIN), 0, 1) * 0.75)[..., None]
    estimate = (grey * (1 - alpha) + heat * alpha).astype(np.uint8)
    _contour(estimate, _resize(shadow.area[rows, cols], shape), COLOUR_SHADOW, 3)
    gap = np.full((SHADOW_PANEL_H, 16, 3), 255, np.uint8)
    if shadow.ring is None:
        return np.hstack([zones, gap, estimate])
    # Опора: откуда берётся бумага для pull-push — кольцо R (на листе и за краем), отсеянное, пиксели B.
    support = grey.copy()
    layer = support.copy()
    if shadow.paper is not None:
        layer[_resize(shadow.paper[rows, cols], shape)] = COLOUR_PAPER_AS_IS
    if shadow.ring_dropped is not None:
        layer[_resize(shadow.ring_dropped[rows, cols], shape)] = COLOUR_RING_DROPPED
    book = shadow.book if shadow.book is not None else np.ones_like(shadow.ring)
    layer[_resize((shadow.ring & book)[rows, cols], shape)] = COLOUR_RING
    layer[_resize((shadow.ring & ~book)[rows, cols], shape)] = COLOUR_RING_OUT
    support = cv2.addWeighted(layer, 0.6, support, 0.4, 0)
    _contour(support, _resize(shadow.area[rows, cols], shape), COLOUR_SHADOW, 3)
    return np.hstack([zones, gap, estimate, gap, support])


def draw_finger_shadow(
    before: np.ndarray, shadow: FingerShadow, paint: np.ndarray, core: np.ndarray, px_per_mm: float, title: str
) -> np.ndarray | None:
    """Картинка тени пальца: по строке на компоненту зоны поиска — «область поиска с зонами | карта прибавки».

    Args:
        before: Итог до осветления BGR (полный масштаб).
        shadow: Карта тени (масштаб ``finger_shadow.SHADOW_SCALE``).
        paint: Зона закраса пальца в координатах итога (bool, полный масштаб).
        core: Ядро пальца в координатах итога (bool, полный масштаб).
        px_per_mm: Плотность итога, px/мм (площадь зоны в подписи).
        title: Шапка.

    Returns:
        Картинка BGR или None, если зоны поиска нет.
    """
    if not shadow.area.any():
        return None
    shape = shadow.area.shape
    small = cv2.resize(before, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    paint_s, core_s = _resize(paint, shape), _resize(core, shape)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(shadow.area.astype(np.uint8), connectivity=8)
    rows_out = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        # Поле — до внешнего края кольца опоры (кольцо в REF_RING_MM от ядра, зона — в SHADOW_REACH_MM).
        mm = px_per_mm * shape[0] / before.shape[0]
        ring_px = int((REF_RING_MM[1] - SHADOW_REACH_MM) * mm) + 4
        mx, my = max(int(w * SHADOW_MARGIN) + 4, ring_px), max(int(h * SHADOW_MARGIN) + 4, ring_px)
        rows = slice(max(0, y - my), min(shape[0], y + h + my))
        cols = slice(max(0, x - mx), min(shape[1], x + w + mx))
        level = shadow.delta[labels == i].mean(axis=1)
        caption = (
            f"зона поиска {i}: {stats[i, cv2.CC_STAT_AREA] / mm**2:.0f} мм²; прибавка в ней средняя "
            f"{level.mean():.1f}, 99 % — {np.percentile(level, 99):.1f} уровней"
        )
        panels = _shadow_panels(small, shadow, paint_s, core_s, rows, cols)
        rows_out.append(framed(panels, [caption], []))
    width = max(r.shape[1] for r in rows_out)
    canvas = np.vstack(
        [
            cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for r in rows_out
        ]
    )
    legend = [
        LegendEntry("ядро пальца", COLOUR_CORE, alpha=AREA_ALPHA, style=SampleStyle.BOX),
        LegendEntry(
            "зона закраса пальца: ядро + асимметричная дилатация", COLOUR_PAINT, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry(f"зона поиска тени (ядро + {SHADOW_REACH_MM:g} мм, в книге)", COLOUR_SHADOW),
        LegendEntry("кожа в итоге — не оценивается и не осветляется", COLOUR_SKIN),
        LegendEntry(
            f"опора R: кольцо {REF_RING_MM[0]:g}–{REF_RING_MM[1]:g} мм от ядра на листе",
            COLOUR_RING,
            alpha=0.6,
            style=SampleStyle.BOX,
        ),
        LegendEntry("опора R за краем листа (полоса закраса)", COLOUR_RING_OUT, alpha=0.6, style=SampleStyle.BOX),
        LegendEntry(
            "отсеяно из опоры: чужой цвет, тёмный закрас за краем",
            COLOUR_RING_DROPPED,
            alpha=0.6,
            style=SampleStyle.BOX,
        ),
        LegendEntry(
            "«бумага как есть» B: чистая бумага без краски, кожи, печати",
            COLOUR_PAPER_AS_IS,
            alpha=0.6,
            style=SampleStyle.BOX,
        ),
    ] + [
        LegendEntry(f"прибавка {v:g} уровней", _heat_colour(v), alpha=0.75, style=SampleStyle.BOX)
        for v in (2, 8, 15, 30)
    ]
    sub = (
        "слева — итог до осветления с зонами, в середине — карта прибавки, справа — откуда берётся бумага для pull-push"
    )
    return framed(canvas, [title, sub], legend)
