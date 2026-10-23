"""Второе преобразование по тексту: выборки «точка → цель» по осям строк (в горизонталь) и выровненным сторонам текстовых блоков (в вертикаль) из ``page_layout.text_blocks``, гладкое поле смещений тонкой пластиной на сетке и составная карта с поверхностью Кунса."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.hints import LayoutHints
from ocr_utils.page_layout.text_blocks.leaders import inside_spans
from ocr_utils.page_layout.text_blocks.lines import clean_run
from ocr_utils.page_layout.text_blocks.page import PageAnalysis, analyse_gray
from ocr_utils.units import mm_to_px, mm_to_px_float, px_to_mm
from research.page_rectify.coons import Layout, Patch, PageRect, rect_fractions

# Короче этого ось строки или выровненная сторона в выборки не идёт, мм: на коротких наклон и изгиб — шум формы букв.
MIN_RUN_MM = 25.0
# Сторона блока короче этого в выборки не идёт, мм; стороны одной отбивки колонки (медианы x ближе COLUMN_TOL_MM)
# выпрямляются на общую вертикаль. Допуск меньше обычного отступа цитаты или вставки (5+ мм), но больше остаточного
# увода отбивки после Кунса (до 2.6 мм — ХиП 1991/07 IMG_0090).
MIN_SIDE_MM = 8.0
COLUMN_TOL_MM = 3.5
# Шаг выборок вдоль оси или стороны, мм; на концах — гуще: самые сильные изгибы строк — на концах, у корешка лист
# уходит в сгиб на последних 3–5 мм (ХиП 1991/01 IMG_1057), и выборки через 3 мм от начала оси до конца не дотягивались.
SAMPLE_STEP_MM = 3.0
END_STEP_MM = 1.0
END_ZONE_MM = 8.0
# Радиус, в котором выборки считаются дублями (одна строка из двух пересекающихся блоков или страниц), мм.
DUPLICATE_RADIUS_MM = 2.0
# Шаг узлов сетки поля, мм.
FIELD_STEP_MM = 4.0
# Длина, на которой поле ещё может заметно гнуться, мм: жёсткость тонкой пластины ``λ = (L / шаг)^4 · κ``. Синтетика
# (волна строк ±0.5 мм, строки через 3.4 мм): при 10 мм поле снимает волну периода 50 мм на 96 %, 100 мм — на 100 %,
# 25 мм — на 60 %; при 20 мм — 60 % и 95 %, 17 % (тест ``test_text_field``).
FIELD_SMOOTH_MM = 10.0
# У корешка поле гибче: лист уходит в сгиб, и строки загибаются на последних 5–8 мм (ХиП 1991/01 IMG_1057, правая
# страница). При 10 мм остаток выборок в 5 мм от корешка — медиана 0.55 мм против 0.05 мм в теле страницы, при 4 мм —
# 0.12 мм. Длина изгиба 4 мм — в полосе ``GUTTER_ZONE_MM`` от линии корешка, дальше на ``GUTTER_RAMP_MM`` она линейно
# растёт до ``FIELD_SMOOTH_MM``: в теле страницы поле не повторяет шум формы букв.
GUTTER_SMOOTH_MM = 4.0
# Поперёк строк (по y) поле жёсткое: длина изгиба — несколько шагов строк. Иначе гибкое у корешка поле менялось на
# масштабе шага строк (~4 мм), соседние строки тянулись к своим целям в разные стороны, и начала строк у корешка
# мялись по вертикали («Они сводятся», «грузчику» — ХиП 1991/01 IMG_1057). Систематический загиб концов многих строк
# подряд поле так повторяет, а одну выбившуюся строку — нет. 8 мм, а не 15: при 15 сноски внизу страницы наследовали
# изгиб текста над ними (01/IMG_1040: размах строки сноски 0.43 мм → 0.23); на контрольных 95 % размаха строк
# 0.50 → 0.41 (1057), 0.44 → 0.38 (0127), 0.58 → 0.52 (0412), 0.42 → 0.34 (0424), масштаб 99 % |s−1| +0.005.
ACROSS_SMOOTH_MM = 8.0
GUTTER_ZONE_MM = 15.0
GUTTER_RAMP_MM = 15.0
STIFFNESS_SCALE = 0.05
# Малое притяжение поля к нулю (вес на узел): где данных нет, поле гаснет, а не уходит линейно в бесконечность.
ZERO_PULL = 1e-4
# Масштаб по вертикали: строки выпрямляются сдвигом, а не сжатием межстрочья (``fit_component``). Цель строки —
# свободная поправка к медиане с малым притяжением к нулю; вес штрафа за ∂dy/∂y — на узел.
SCALE_WEIGHT = 2.0
TARGET_PULL = 1e-3
# Робастность: порог Хьюбера и отбрасывание выборок с большим остатком, мм; число проходов.
HUBER_MM = 0.5
OUTLIER_MM = 3.0
IRLS_PASSES = 3
# Потолок модуля смещения, мм: сверх него поле обрезается, кадр помечается.
MAX_SHIFT_MM = 8.0
# Меньше стольких осей строк на странице — полного второго преобразования для неё нет.
MIN_LINES = 8
# Разреженный режим (:class:`FieldMode`): у страницы с малым числом строк (титул, фото с подписью — ХиП 1991/04
# IMG_0154: 2 строки, 142 мм осей, опоры на 88 мм по высоте, подпись после Кунса наклонена на 2°) поле строится, если
# строк не меньше SPARSE_MIN_LINES, их суммарная длина не меньше SPARSE_MIN_LENGTH_MM и опоры занимают по высоте не
# меньше SPARSE_MIN_SPAN_MM (две строки рядом наклон страницы не задают); нули ставятся только в узлах дальше
# SPARSE_ZERO_MM от любой опоры — в фото и на пустых полях поле гаснет к Кунсу, а не выдумывается.
SPARSE_MIN_LINES = 2
SPARSE_MIN_LENGTH_MM = 100.0
SPARSE_MIN_SPAN_MM = 40.0
SPARSE_ZERO_MM = 15.0
# Шаг «нулевых» выборок на странице без поля (цветная, мало строк), мм, и их вес.
ZERO_STEP_MM = 8.0
ZERO_WEIGHT = 4.0
# Поле страницы при анализе: прямоугольник первого преобразования плюс столько, мм.
CROP_MARGIN_MM = 5.0
# Утолщение штрихов по вертикали в копии для разбора, мм (0.25 мм = 3 px при 300 dpi, ядро симметрично — ось строки,
# центр масс краски, не сдвигается). Фильтр «тонкая черта» разбора (``segment._thin_rule``) выбрасывает строку, если
# медианная толщина краски в столбце ≤ 0.6 мм = 7 px; у основного текста ПХ 1975 на фотоскане она ровно 7 px (у ХиП
# 1991 — 10): на ph_gutter_strip neighbor/03/IMG_0022 разбор находил 7 и 1 строку из ~50 на страницу, с утолщением —
# 46 и 57; ПХ 1975 04/IMG_0020 — 0/1 → 60/58; ХиП 1991 09/IMG_0424 — 46/47 → 45/45 (разбор форка, 2026-10-22).
THICKEN_MM = 0.25
# Линейки (``find_rules``): рабочая копия, dpi; отрезок открытия, мм; наименьшая длина, мм; наибольшая средняя толщина,
# мм; окно сглаживания осевой, мм. Замер по 83 кадрам итогов v2 (ХиП 1991, ПХ 1975, 2026-10-22): кандидаты 14–29 мм —
# почти все черты над сносками (~20 мм, 0.2–0.3 мм толщиной; ХиП 1991/03 IMG_0107 — без опоры черта гнулась вслед за
# строкой сноски), поэтому длина от 15 мм; толщина линеек таблиц и одиноких черт — 0.1–0.5 мм, двойная линейка
# колонтитула (толстая + тонкая, сливаются расширением поперёк) — 1.15–1.54 мм (01/IMG_1046 отбрасывалась при 0.8),
# дальше по толщине — только край листа и пустые страницы (от 5.9 мм); строка букв в открытии не выживает.
RULE_WORK_DPI = 150
RULE_KERNEL_MM = 8.0
RULE_MIN_MM = 15.0
RULE_MAX_THICK_MM = 1.8
RULE_SMOOTH_MM = 2.0
# Расширение краски поперёк линейки перед открытием, мм (≈2 px при 150 dpi): терпит наклон черты до slack / отрезок.
RULE_SLACK_MM = 0.35
# Линейка ближе этого к стороне прямоугольника страницы — край тени или бумаги, а не черта, мм.
RULE_EDGE_MM = 4.0
# Линейка-сирота: краска её компонент не выходит поперёк дальше этого от линейки, мм (засечки оси графика, стрелки,
# прилипшие буквы — дальше); поле вокруг рисунков и фото, внутри которого опор нет, мм.
RULE_ORPHAN_BAND_MM = 0.8
# Замкнутая рамка (:func:`_is_frame`): ширина полосы вдоль стороны, мм; доля краски компоненты внутри полос не больше;
# доля длины каждой стороны, занятая краской, не меньше; бумага снаружи стороны у края страницы ищется до стольких мм.
FRAME_BAND_MM = 1.8
FRAME_INNER_SHARE = 0.02
FRAME_SIDE_COVER = 0.8
FRAME_PAPER_MM = 2.0
RULE_ZONE_MARGIN_MM = 1.0
# Смешивание полей страниц в пустой полосе у корешка: отступ от края текста и наибольшая полуширина полосы, мм.
BLEND_TEXT_MARGIN_MM = 1.0
BLEND_MAX_MM = 8.0
# Уже этого полоса смешивания не строится (резкий разрыв на линии разреза), мм.
BLEND_MIN_MM = 2.0
# Край текста у корешка — этот перцентиль концов строк и сторон (а не крайний конец), %.
BLEND_OUTLIER_PCT = 5.0
# Составная карта: шаг сетки, на которой считается Кунс, px, и полоса вокруг шва страниц, считаемая точно, px.
COMPOSE_STEP_PX = 4
COMPOSE_SEAM_PX = 12


class SampleKind(StrEnum):
    """Откуда выборка."""

    LINE = "line"  # ось строки → горизонталь (составляющая dy)
    SIDE = "side"  # выровненная сторона блока → вертикаль (составляющая dx)
    ZERO = "zero"  # страница без поля: смещение 0 (обе составляющие)


@dataclass(frozen=True)
class Samples:
    """Выборки одной составляющей поля.

    Поля:
        at: Где на холсте итога задано смещение (N, 2) — положение точки после выпрямления.
        value: Смещение, px холста: где брать пиксель выпрямленного кадра относительно ``at`` (N,).
        weight: Вес (N,) после деления на плотность дублей.
        source: Точка на выпрямленном кадре (N, 2) — для оверлея «точка → цель».
        kind: Вид выборки по каждой точке (N,) — значения :class:`SampleKind`.
        group: Номер строки или стороны, из которой выборка (N,): дублями считаются только выборки разных групп.
    """

    at: np.ndarray
    value: np.ndarray
    weight: np.ndarray
    source: np.ndarray
    kind: np.ndarray
    group: np.ndarray

    @classmethod
    def empty(cls) -> Samples:
        """Пустой набор."""
        z = np.zeros((0, 2))
        return cls(z, np.zeros(0), np.zeros(0), z, np.zeros(0, dtype=object), np.zeros(0, dtype=np.int64))

    def __add__(self, other: Samples) -> Samples:
        """Склейка двух наборов; номера групп второго сдвигаются за номера первого (нулевые, ``-1``, — как есть)."""
        shift = int(self.group.max()) + 1 if len(self) else 0
        group = np.where(other.group >= 0, other.group + shift, other.group)
        joined = [np.concatenate([a, b]) for a, b in zip(self.astuple()[:-1], other.astuple()[:-1])]
        return Samples(*joined, np.concatenate([self.group, group]))

    def astuple(self) -> tuple[np.ndarray, ...]:
        """Поля по порядку."""
        return self.at, self.value, self.weight, self.source, self.kind, self.group

    def __len__(self) -> int:
        return len(self.value)

    def take(self, sel: np.ndarray) -> Samples:
        """Часть выборок по булевой маске или индексам."""
        return Samples(*(a[sel] for a in self.astuple()))


class RuleAxis(StrEnum):
    """Направление линейки."""

    HORIZONTAL = "horizontal"  # выпрямляется в горизонталь (составляющая dy)
    VERTICAL = "vertical"  # выпрямляется в вертикаль (составляющая dx)


@dataclass(frozen=True)
class Rule:
    """Линейка страницы: одинокая черта, линейка таблицы, подчёркивание, отбивка.

    Поля:
        axis: Направление.
        points: Осевая линия (N, 2) px холста, по возрастанию вдоль линейки.
        frame: Сторона замкнутой рамки вокруг полосы или колонки.
    """

    axis: RuleAxis
    points: np.ndarray
    frame: bool = False  # сторона замкнутой рамки (:func:`orphan_rules`)


class RegionKindR(StrEnum):
    """Вид области страницы из детекторов ``page_layout`` (по пикселям, без surya)."""

    RASTER = "raster"  # фото, растр
    TABLE = "table"  # таблица
    LINE_ART = "line_art"  # схема, рисунок, график
    EQUATION = "equation"  # формула (surya ``Equation`` из кэша v2)


class FieldMode(StrEnum):
    """Как страница участвует во втором преобразовании."""

    FULL = "full"  # строк достаточно: поле по всем опорам
    SPARSE = "sparse"  # строк мало, но они длинные и разнесены по высоте: поле у опор, вдали — нули
    NONE = "none"  # цветная или опор нет: только нули


@dataclass(frozen=True)
class PageRegion:
    """Область страницы на холсте ``(x0, y0, x1, y1)`` px."""

    kind: RegionKindR
    box: tuple[float, float, float, float]

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Какие точки (N, 2) внутри рамки с полем ``margin`` px."""
        x0, y0, x1, y1 = self.box
        return ((points[:, 0] >= x0 - margin) & (points[:, 0] <= x1 + margin)
                & (points[:, 1] >= y0 - margin) & (points[:, 1] <= y1 + margin))  # fmt: skip


def page_regions(canvas: np.ndarray, box: tuple[int, int, int, int], dpi: float) -> tuple[PageRegion, ...]:
    """Растр, таблицы и line art страницы детекторами ``page_layout`` по пикселям (без surya; ~0.5–1.5 с).

    Args:
        canvas: Выпрямленный кадр BGR.
        box: Вырезка страницы ``(x0, y0, x1, y1)``.
        dpi: Разрешение холста.

    Returns:
        Области на холсте.
    """
    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, Variant

    x0, y0, x1, y1 = box
    image = PageImage.from_array(canvas[y0:y1, x0:x1], dpi=dpi, variant=Variant.SCAN, cache_name="page_rectify")
    layout = PageLayout(image, {Find.RASTER, Find.TABLES, Find.LINE_ART}, LayoutOptions(use_surya=False)).process(None)
    out = []
    for kind, found in ((RegionKindR.RASTER, layout.raster_pics), (RegionKindR.TABLE, layout.tables),
                        (RegionKindR.LINE_ART, layout.line_arts)):  # fmt: skip
        for region in found:
            b = region.box
            out.append(PageRegion(kind, (b.x0 + x0, b.y0 + y0, b.x1 + x0, b.y1 + y0)))
    return tuple(out)


def output_to_canvas(points: np.ndarray, maps: tuple[np.ndarray, np.ndarray], step: int = 8) -> np.ndarray:
    """Точки итога v2 (N, 2) → холст выпрямленного кадра: ближайший узел прореженной карты Кунса (обратная карта).

    Args:
        points: Точки px итога v2.
        maps: ``(map_x, map_y)`` карты холст → итог v2 (``rectify_maps``).
        step: Шаг прореживания карты, px (точность — около шага).

    Returns:
        Точки px холста.
    """
    sub_x, sub_y = maps[0][::step, ::step], maps[1][::step, ::step]
    tree = cKDTree(np.stack([sub_x.ravel(), sub_y.ravel()], 1))
    _, idx = tree.query(np.asarray(points, np.float64))
    rows, cols = np.unravel_index(idx, sub_x.shape)
    return np.stack([cols * step, rows * step], 1).astype(np.float64)


def equation_regions(
    equations: tuple[np.ndarray, ...], to_output, maps: tuple[np.ndarray, np.ndarray], rect: PageRect
) -> tuple[PageRegion, ...]:
    """Формулы страницы на холсте: многоугольники surya (px кадра) → итог v2 → холст; рамка по вершинам.

    Args:
        equations: Многоугольники ``Equation`` в px кадра.
        to_output: Перевод px кадра → px итога v2 (сдвиг на рамку вырезки).
        maps: Карты Кунса холст → итог v2.
        rect: Прямоугольник страницы (берутся формулы с центром в нём).

    Returns:
        Области формул.
    """
    out = []
    for poly in equations:
        pts = output_to_canvas(to_output(poly), maps)
        cx, cy = pts.mean(axis=0)
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            out.append(
                PageRegion(RegionKindR.EQUATION, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
            )
    return tuple(out)


def no_text_zones(regions: tuple[PageRegion, ...]) -> tuple[PageRegion, ...]:
    """Области, где опор поля нет: строки и линейки внутри рисунка или фото — часть изображения, а не текст."""
    return tuple(r for r in regions if r.kind in (RegionKindR.RASTER, RegionKindR.LINE_ART, RegionKindR.EQUATION))


def outside(points: np.ndarray, zones: tuple[PageRegion, ...], margin: float) -> np.ndarray:
    """Какие точки (N, 2) вне всех зон (с полем ``margin``)."""
    keep = np.ones(len(points), bool)
    for zone in zones:
        keep &= ~zone.contains(points, margin)
    return keep


@dataclass(frozen=True)
class PageText:
    """Разбор текста одной страницы выпрямленного кадра.

    Поля:
        analysis: Разбор ``analyse_gray`` (px рабочей копии 150 dpi).
        offset: Левый верхний угол вырезки на холсте, px.
        scale: px холста на px рабочей копии.
        rules: Линейки страницы (горизонтальные и вертикальные) на холсте, годные в опоры поля.
        regions: Растр, таблицы и line art страницы на холсте (``page_regions``).
    """

    analysis: PageAnalysis
    offset: np.ndarray
    scale: float
    rules: tuple[Rule, ...] = ()
    regions: tuple[PageRegion, ...] = ()

    def to_canvas(self, points: np.ndarray) -> np.ndarray:
        """Точки рабочей копии (N, 2) → px холста."""
        return np.asarray(points, np.float64) * self.scale + self.offset


def crop_box(
    rect: PageRect, shape: tuple[int, int], margin_px: float, spine_x: float, left_page: bool
) -> tuple[int, int, int, int]:
    """Вырезка страницы на холсте: прямоугольник плюс поле, но не дальше линии корешка ``(x0, y0, x1, y1)``.

    Со стороны корешка поле не берётся: иначе в вырезку попадали концы строк соседней страницы, и разбор сращивал их
    со своими строками (перенос «го-» левой страницы с началом строки правой — ХиП 1991/01 IMG_1057).

    Args:
        rect: Прямоугольник страницы.
        shape: ``(H, W)`` холста.
        margin_px: Поле вокруг прямоугольника, px.
        spine_x: X линии корешка на холсте.
        left_page: Левая страница (корешок справа).

    Returns:
        Вырезка в пределах холста.
    """
    h, w = shape
    x0, y0 = max(int(rect.x0 - margin_px), 0), max(int(rect.y0 - margin_px), 0)
    x1, y1 = min(int(np.ceil(rect.x1 + margin_px)), w), min(int(np.ceil(rect.y1 + margin_px)), h)
    if left_page:
        x1 = min(x1, int(np.floor(spine_x)))
    else:
        x0 = max(x0, int(np.ceil(spine_x)))
    return x0, y0, x1, y1


def analyze_page(
    canvas: np.ndarray, box: tuple[int, int, int, int], dpi: float, rect: PageRect, regions: tuple[PageRegion, ...] = ()
) -> PageText:
    """Разбор текста страницы: вырезка → серый → 300 dpi → ``analyse_gray`` с движком по краске (CPU).

    Args:
        canvas: Выпрямленный кадр BGR.
        box: Вырезка ``(x0, y0, x1, y1)``.
        dpi: Разрешение холста.
        rect: Прямоугольник страницы на холсте (линейки у его сторон отбрасываются — :func:`inner_rules`).
        regions: Растр, таблицы и line art страницы (линейки в рисунках и фото отбрасываются, вне таблиц — только
            сироты: :func:`orphan_rules`).

    Returns:
        Разбор и привязка к холсту.
    """
    x0, y0, x1, y1 = box
    part = canvas[y0:y1, x0:x1]
    # Без забеливания цвета: в ХиП и ПХ «цветные элементы» — это в основном цветная печать, а фото чёрно-белые, так что
    # забеливание по цвету стирало текст (01/IMG_1088) и от ложных строк на фото не защищало (решение 2026-10-22).
    gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
    f = RENDER_DPI / dpi
    gray300 = thicken_strokes(cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    offset = np.array([x0, y0], np.float64)
    # Рисунки и фото — преграды разбора: строка через рамку не сцепляется, блок по её горизонтальным рёбрам делится.
    # Без этого узкая колонка у рисунка и полная ширина под ним собирались в один блок, и правая сторона выходила
    # рваной (ХиП 1991/01 IMG_0986: 20 из 34 строк «с отступом»).
    barriers = region_barriers(regions, offset, dpi, (x1 - x0, y1 - y0))
    analysis = analyse_gray(gray300, InkEngine(), hints=LayoutHints(barriers=barriers))
    if barriers:
        # Второй проход: линии раздела над и под рисунком — между последним рядом обтекания и первым рядом на всю
        # ширину (по рядам первого прохода; рамка рисунка с текстом не совпадает до строки).
        cuts = barrier_cuts(barriers, analysis.axes, (x1 - x0) * WORK_DPI / dpi)
        analysis = analyse_gray(gray300, InkEngine(), hints=LayoutHints(barriers=barriers, rules=cuts))
    # Рабочая копия — ``analysis.dpi`` (150): px холста на px копии = dpi холста / dpi копии.
    rules = orphan_rules(find_rules(gray, dpi, offset), gray, offset, dpi, regions)
    rules = inner_rules(rules, rect, dpi, gray, offset)
    scale = dpi / analysis.dpi
    return PageText(analysis, offset, scale, rules, regions)


def region_barriers(
    regions: tuple[PageRegion, ...], offset: np.ndarray, dpi: float, size: tuple[int, int]
) -> tuple[tuple[int, int, int, int], ...]:
    """Рамки line art, растра и таблиц страницы как преграды разбора (``LayoutHints.barriers``), px рабочей копии.

    Текст, упёршийся боком в рисунок или таблицу, выровнен по ней иначе, чем текст во всю ширину, — это разные блоки;
    межколонник двух колонок над таблицей не продлевается в неё (ПХ 1975/01 IMG_0083).

    Args:
        regions: Области страницы на холсте.
        offset: Левый верхний угол вырезки на холсте.
        dpi: Разрешение холста.
        size: Ширина и высота вырезки, px холста (рамка обрезается по ней).

    Returns:
        Рамки ``(x0, y0, x1, y1)`` в px рабочей копии :data:`WORK_DPI` вырезки; рамки вне вырезки пропускаются.
    """
    f = WORK_DPI / dpi
    w, h = int(size[0] * f), int(size[1] * f)
    out = []
    for region in regions:
        if region.kind not in (RegionKindR.LINE_ART, RegionKindR.RASTER, RegionKindR.TABLE):
            continue
        x0, y0, x1, y1 = region.box
        bx0 = max(int(np.floor((x0 - offset[0]) * f)), 0)
        by0 = max(int(np.floor((y0 - offset[1]) * f)), 0)
        bx1 = min(int(np.ceil((x1 - offset[0]) * f)), w)
        by1 = min(int(np.ceil((y1 - offset[1]) * f)), h)
        if bx1 - bx0 >= 2 and by1 - by0 >= 2:
            out.append((bx0, by0, bx1, by1))
    return tuple(out)


def barrier_cuts(
    barriers: tuple[tuple[int, int, int, int], ...], axes: tuple, width: float
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Линии раздела блоков над и под рисунком на всю ширину вырезки (``LayoutHints.rules``), px рабочей копии.

    Рёбра рамки делят блок, только если лежат под самими рядами (``regroup._divided``), а узкая колонка у рисунка
    стоит сбоку от рамки: рёбра её не накрывают, и обтекание срасталось с полной шириной под рисунком в один блок с
    рваной стороной (ХиП 1991/01 IMG_0986: 20 из 34 строк «с отступом»). Линия раздела ставится посередине между
    соседними рядами, где ряд «сбоку от рамки» сменяется рядом «через рамку» (поиск в шаге строк от ребра); не нашлось
    смены — по самому ребру. Блок соседней колонки на той же высоте тоже делится, но обе части остаются со сторонами.

    Args:
        barriers: Рамки рисунков ``(x0, y0, x1, y1)``, px рабочей копии.
        axes: Оси строк первого прохода (``LineAxis``, px рабочей копии).
        width: Ширина вырезки, px рабочей копии.

    Returns:
        Горизонтальные ломаные из двух точек.
    """
    rows = sorted(((float(np.median(a.points[:, 1])), a.x0, a.x1) for a in axes), key=lambda r: r[0])
    ys = np.array([r[0] for r in rows])
    pitch = float(np.median(np.diff(ys))) if len(ys) > 2 else 0.0
    out = []
    for bx0, by0, bx1, by1 in barriers:
        tol = 0.25 * (bx1 - bx0)
        # Ряд «через рамку» заходит по x внутрь рамки дальше допуска; остальные ряды — сбоку от неё или вне колонки.
        across = [x0 < bx1 - tol and x1 > bx0 + tol for _, x0, x1 in rows]
        for edge, below in ((by0, False), (by1, True)):
            cut = float(edge)
            for k in range(len(rows) - 1):
                upper, lower = rows[k][0], rows[k + 1][0]
                if not (edge - 1.5 * pitch <= (upper + lower) / 2 <= edge + 1.5 * pitch):
                    continue
                if across[k] != across[k + 1] and across[k + 1] == below:
                    cut = (upper + lower) / 2
                    break
            out.append(((0.0, cut), (float(width), cut)))
    return tuple(out)


def orphan_rules(
    rules: tuple[Rule, ...], gray: np.ndarray, offset: np.ndarray, dpi: float, regions: tuple[PageRegion, ...]
) -> tuple[Rule, ...]:
    """Линейки, по которым можно выпрямлять: не в рисунке и не в фото, а вне таблиц — только сироты.

    Линейка внутри line art или растра (ось графика, кривая, штриховка) — часть рисунка: её форма не обязана быть
    прямой. Вне таблиц линейка годится, только если она сирота: краска её связной компоненты почти вся на самой
    линейке (к оси графика примыкают засечки и подписи, к рамке схемы — стрелки и блоки). Линейки таблиц (горизонтальные
    и вертикальные) примыкают друг к другу законно и берутся без проверки.

    Args:
        rules: Линейки страницы (px холста).
        gray: Серая вырезка страницы (разрешение холста).
        offset: Левый верхний угол вырезки на холсте.
        dpi: Разрешение холста.
        regions: Области страницы.

    Returns:
        Годные линейки.
    """
    zones, tables = no_text_zones(regions), [r for r in regions if r.kind is RegionKindR.TABLE]
    f = RULE_WORK_DPI / dpi
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    margin = mm_to_px_float(RULE_ZONE_MARGIN_MM, dpi)
    band = mm_to_px_float(RULE_ORPHAN_BAND_MM, RULE_WORK_DPI)
    kept = []
    for rule in rules:
        if (~outside(rule.points, zones, margin)).any():
            continue
        if any(t.contains(rule.points, margin).mean() > 0.5 for t in tables):
            kept.append(rule)
            continue
        # Сирота: компоненты краски под линейкой не выходят поперёк за полосу ``band`` вокруг неё.
        pts = (rule.points - offset) * f
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, ink.shape[1] - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, ink.shape[0] - 1)
        ids = np.unique(labels[yi, xi])
        ids = ids[ids > 0]
        if not len(ids):
            continue
        horizontal = rule.axis is RuleAxis.HORIZONTAL
        # Сторона замкнутой рамки (вокруг полосы, колонки, врезки): её компонента — вся рамка, и поперёк краска уходит
        # на всю высоту рамки; сиротой она не была бы никогда (1972/09 IMG_0105_1L).
        frame = next((i for i in ids if _is_frame(labels == i, stats[i], rule, pts, horizontal)), None)
        if frame is not None:
            kept.append(replace(rule, frame=True))
            continue
        lo, hi = (pts[:, 1].min(), pts[:, 1].max()) if horizontal else (pts[:, 0].min(), pts[:, 0].max())
        start = stats[ids, cv2.CC_STAT_TOP if horizontal else cv2.CC_STAT_LEFT]
        size = stats[ids, cv2.CC_STAT_HEIGHT if horizontal else cv2.CC_STAT_WIDTH]
        if (start >= lo - band).all() and (start + size <= hi + band).all():
            kept.append(rule)
    return tuple(kept)


def _is_frame(component: np.ndarray, stat: np.ndarray, rule: Rule, pts: np.ndarray, horizontal: bool) -> bool:
    """Замкнутая ли рамка компонента краски под линейкой, и лежит ли линейка на её стороне (копия ``RULE_WORK_DPI``).

    Рамка: обе стороны прямоугольника компоненты не короче :data:`RULE_MIN_MM`; краска лежит в полосах шириной
    :data:`FRAME_BAND_MM` вдоль четырёх сторон (внутри — пусто: текст рамки — свои компоненты), и все четыре полосы
    заняты краской на большей части длины; линейка — вдоль одной из сторон.

    Args:
        component: Маска компоненты (размер копии).
        stat: Строка ``connectedComponentsWithStats`` компоненты.
        rule: Линейка.
        pts: Осевая линия линейки на копии.
        horizontal: Линейка горизонтальная.

    Returns:
        ``True`` — линейка — сторона рамки.
    """
    x, y, w, h = (int(v) for v in stat[:4])
    if min(w, h) < mm_to_px_float(RULE_MIN_MM, RULE_WORK_DPI):
        return False
    band = max(mm_to_px(FRAME_BAND_MM, RULE_WORK_DPI), 2)
    box = component[y : y + h, x : x + w]
    inner = box[band:-band, band:-band]
    if inner.size == 0 or inner.sum() > FRAME_INNER_SHARE * box.sum():
        return False
    sides = (box[:band].any(axis=0), box[-band:].any(axis=0), box[:, :band].any(axis=1), box[:, -band:].any(axis=1))
    if min(float(side.mean()) for side in sides) < FRAME_SIDE_COVER:
        return False
    across = float(np.median(pts[:, 1] if horizontal else pts[:, 0]))
    edges = (y, y + h) if horizontal else (x, x + w)
    return min(abs(across - e) for e in edges) <= band


def inner_rules(
    rules: tuple[Rule, ...],
    rect: PageRect,
    dpi: float,
    gray: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> tuple[Rule, ...]:
    """Линейки не у края страницы: медиана поперечной координаты дальше :data:`RULE_EDGE_MM` от сторон прямоугольника.

    У края листа детектор принимает за линейку край тени или кромку бумаги (ХиП 1991/01 IMG_0927, низ пустой левой
    страницы); настоящие черты — под колонтитулом, над сносками — лежат от края на 10 мм и дальше.

    Args:
        rules: Линейки страницы.
        rect: Прямоугольник страницы на холсте.
        dpi: Разрешение холста.

    Returns:
        Линейки без лежащих у края.
    """
    edge = mm_to_px_float(RULE_EDGE_MM, dpi)
    kept = []
    for rule in rules:
        across = float(np.median(rule.points[:, 1 if rule.axis is RuleAxis.HORIZONTAL else 0]))
        lo, hi = (rect.y0, rect.y1) if rule.axis is RuleAxis.HORIZONTAL else (rect.x0, rect.x1)
        if lo + edge < across < hi - edge or (rule.frame and _paper_outside(rule, rect, dpi, gray, offset)):
            kept.append(rule)
    return tuple(kept)


def _paper_outside(rule: Rule, rect: PageRect, dpi: float, gray: np.ndarray | None, offset: np.ndarray | None) -> bool:
    """Есть ли бумага снаружи стороны рамки у края страницы: полоса от 0.5 до :data:`FRAME_PAPER_MM` мм за линейкой (в
    сторону ближнего края) светлее порога Оцу вырезки — значит, это рамка, а не тень кромки листа.

    Args:
        rule: Сторона рамки (px холста).
        rect: Прямоугольник страницы.
        dpi: Разрешение холста.
        gray: Серая вырезка страницы или ``None`` (проверки нет — нет и бумаги).
        offset: Левый верхний угол вырезки на холсте.

    Returns:
        ``True`` — снаружи бумага.
    """
    if gray is None or offset is None:
        return False
    horizontal = rule.axis is RuleAxis.HORIZONTAL
    across = float(np.median(rule.points[:, 1 if horizontal else 0]))
    lo, hi = (rect.y0, rect.y1) if horizontal else (rect.x0, rect.x1)
    sign = -1.0 if across - lo < hi - across else 1.0
    a, b = across + sign * mm_to_px_float(0.5, dpi), across + sign * mm_to_px_float(FRAME_PAPER_MM, dpi)
    a, b = sorted((a, b))
    along = rule.points[:, 0 if horizontal else 1]
    t0, t1 = float(along.min()), float(along.max())
    if horizontal:
        strip = gray[
            max(int(a - offset[1]), 0) : max(int(b - offset[1]), 0),
            max(int(t0 - offset[0]), 0) : max(int(t1 - offset[0]), 0),
        ]
    else:
        strip = gray[
            max(int(t0 - offset[1]), 0) : max(int(t1 - offset[1]), 0),
            max(int(a - offset[0]), 0) : max(int(b - offset[0]), 0),
        ]
    if strip.size == 0:
        return False
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(np.median(strip)) > threshold


def find_rules(gray: np.ndarray, dpi: float, offset: np.ndarray) -> tuple[Rule, ...]:
    """Линейки страницы по краске: длинные тонкие прямые по горизонтали и по вертикали.

    Копия :data:`RULE_WORK_DPI`, краска — порог Оцу. Морфологическое открытие отрезком :data:`RULE_KERNEL_MM` вдоль
    направления оставляет только длинные пробеги краски (буквы — короче); связная компонента — линейка, если она длиннее
    :data:`RULE_MIN_MM` и тонкая: средняя толщина (площадь / длина) не больше :data:`RULE_MAX_THICK_MM`. Осевая линия —
    средняя координата краски поперёк по каждому столбцу (строке) компоненты, сглаженная.

    Args:
        gray: Серая вырезка страницы (разрешение холста).
        dpi: Разрешение холста.
        offset: Левый верхний угол вырезки на холсте.

    Returns:
        Линейки в px холста.
    """
    f = RULE_WORK_DPI / dpi
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel_len = max(mm_to_px(RULE_KERNEL_MM, RULE_WORK_DPI), 3)
    min_len = mm_to_px_float(RULE_MIN_MM, RULE_WORK_DPI)
    max_thick = mm_to_px_float(RULE_MAX_THICK_MM, RULE_WORK_DPI)
    smooth = max(int(mm_to_px(RULE_SMOOTH_MM, RULE_WORK_DPI)) | 1, 3)
    slack = max(mm_to_px(RULE_SLACK_MM, RULE_WORK_DPI), 1)
    rules = []
    for axis, kernel in ((RuleAxis.HORIZONTAL, (kernel_len, 1)), (RuleAxis.VERTICAL, (1, kernel_len))):
        # Поперёк линейки краска сначала расширяется на ``slack``: тонкая черта, загнутая к концу (у корешка — ХиП
        # 1991/01 IMG_1088, выходные данные), переходит со строки на строку пикселей и не давала прямого пробега длиной
        # в отрезок открытия — линейка обрывалась там, где начинался изгиб.
        across = (1, 2 * slack + 1) if axis is RuleAxis.HORIZONTAL else (2 * slack + 1, 1)
        widened = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, across))
        opened = cv2.morphologyEx(widened, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        # Компоненты — по открытому, а сама линейка (толщина, осевая) — по исходной краске внутри компоненты.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        labels = np.where(ink > 0, labels, 0)
        along = 0 if axis is RuleAxis.HORIZONTAL else 1
        areas = np.bincount(labels.ravel(), minlength=count)
        for k in range(1, count):
            length = stats[k, cv2.CC_STAT_WIDTH if along == 0 else cv2.CC_STAT_HEIGHT]
            if length < min_len or areas[k] == 0 or areas[k] / length > max_thick:
                continue
            ys, xs = np.nonzero(labels == k)
            t, c = (xs, ys) if along == 0 else (ys, xs)
            # Осевая линия: средняя поперечная координата по каждому положению вдоль линейки.
            ts = np.unique(t)
            cs = np.bincount(t - ts[0], weights=c)[ts - ts[0]] / np.bincount(t - ts[0])[ts - ts[0]]
            if len(cs) >= smooth:
                cs = np.convolve(np.pad(cs, smooth // 2, mode="edge"), np.ones(smooth) / smooth, mode="valid")
            pts = np.stack([ts, cs], 1) if along == 0 else np.stack([cs, ts], 1)
            rules.append(Rule(axis, pts / f + offset))
    return tuple(rules)


def rule_samples(page: PageText, dpi: float) -> tuple[Samples, Samples, int]:
    """Выборки по линейкам: горизонтальная — в горизонталь (как ось строки), вертикальная — в вертикаль (как сторона).

    Цель — медиана поперечной координаты линейки; горизонтальные идут свободными целями, как строки.

    Args:
        page: Разбор страницы с линейками.
        dpi: Разрешение холста.

    Returns:
        Выборки ``dx`` (вертикальные), выборки ``dy`` (горизонтальные), число горизонтальных линеек.
    """
    step = mm_to_px_float(SAMPLE_STEP_MM, dpi)
    end_step, end_zone = mm_to_px_float(END_STEP_MM, dpi), mm_to_px_float(END_ZONE_MM, dpi)
    h_parts, v_parts = [], []
    for rule in page.rules:
        along = 0 if rule.axis is RuleAxis.HORIZONTAL else 1
        pts = resample(rule.points, along, step, end_step, end_zone)
        target = float(np.median(pts[:, 1 - along]))
        at = pts.copy()
        at[:, 1 - along] = target
        n = len(pts)
        if along == 0:
            v_parts.append(Samples(at, pts[:, 1] - target, np.ones(n), pts, np.full(n, SampleKind.LINE),
                                   np.full(n, len(v_parts))))  # fmt: skip
        else:
            h_parts.append(Samples(at, pts[:, 0] - target, np.ones(n), pts, np.full(n, SampleKind.SIDE),
                                   np.full(n, len(h_parts))))  # fmt: skip
    return sum(h_parts, Samples.empty()), sum(v_parts, Samples.empty()), len(v_parts)


def thicken_strokes(gray300: np.ndarray, height_mm: float = THICKEN_MM) -> np.ndarray:
    """Серый с утолщёнными по вертикали тёмными штрихами — только для разбора текста, не для итога.

    Вертикальный min-фильтр (тёмное расползается вверх и вниз на ``(k-1)/2`` px): горизонтальные штрихи букв толще,
    столбцы строки набирают краску выше порога «черты», а межстрочный просвет (у ПХ 1975 ~1 мм = 12 px) не смыкается.
    Горизонталь не трогается — края блоков и межколонники остаются на месте.

    Args:
        gray300: Серая вырезка страницы, uint8, 300 dpi.
        height_mm: Высота фильтра, мм бумаги; округляется до нечётного числа пикселей (симметричное ядро).

    Returns:
        Серый того же размера.
    """
    k = max(mm_to_px(height_mm, RENDER_DPI), 1)
    k += 1 - k % 2  # нечётное ядро — без сдвига оси
    if k <= 1:
        return gray300
    return cv2.erode(gray300, np.ones((k, 1), np.uint8))


def resample(points: np.ndarray, along: int, step: float, end_step: float, end_zone: float) -> np.ndarray:
    """Точки ломаной с шагом ``step`` по координате ``along`` (0 — по x, 1 — по y), на концах — с шагом ``end_step``.

    Args:
        points: Ломаная (N, 2).
        along: 0 — шаг по x (ось строки), 1 — по y (сторона блока).
        step: Шаг в теле, px.
        end_step: Шаг в ``end_zone`` от каждого конца, px; сами концы входят всегда.
        end_zone: Длина концевого участка, px.

    Returns:
        Точки (M, 2), упорядоченные по ``along``.
    """
    pts = points[np.argsort(points[:, along])]
    t = pts[:, along]
    t0, t1 = float(t[0]), float(t[-1])
    zone = min(end_zone, (t1 - t0) / 2)
    grid = np.unique(np.concatenate([np.arange(t0, t1 + 1e-9, step), np.arange(t0, t0 + zone, end_step),
                                     np.arange(t1, t1 - zone, -end_step), [t0, t1]]))  # fmt: skip
    other = np.interp(grid, t, pts[:, 1 - along])
    return np.stack([grid, other], axis=1) if along == 0 else np.stack([other, grid], axis=1)


def line_curve(axis) -> np.ndarray:
    """Опорная кривая строки для поля и мер: базовая линия (``LineAxis.base_points``), а без неё — ось центра масс без
    участков над точками и запятыми; участки перескока на соседнюю строку вырезаны (``clean_run``).

    База стоит на низу букв без выносных, и форма отдельных букв её не уводит: ось центра масс задиралась на прописных
    «П», «Р» заголовков на 1.3 мм (ХиП 1991/01 IMG_0993), уходила вниз на курсивных «Н», «П» (IMG_0998) и на
    надстрочной цифре сноски (03/IMG_0107).

    Args:
        axis: Ось строки (``LineAxis``, px рабочей копии).

    Returns:
        Точки ``(N, 2)`` px рабочей копии.
    """
    base = axis.base_points
    if base is not None:
        return clean_run(np.asarray(base, np.float64), axis.jump_spans)
    pts = clean_run(np.asarray(axis.points, np.float64), axis.jump_spans)
    if axis.mark_spans:
        pts = pts[~inside_spans(pts[:, 0], list(axis.mark_spans))]
    return pts


def line_samples(page: PageText, dpi: float) -> tuple[Samples, int]:
    """Выборки по строкам: каждая точка базовой линии строки (:func:`line_curve`) должна встать на медианную высоту
    своей строки.

    Args:
        page: Разбор страницы.
        dpi: Разрешение холста.

    Returns:
        Выборки составляющей ``dy`` и число годных строк.
    """
    min_len, step = mm_to_px_float(MIN_RUN_MM, dpi), mm_to_px_float(SAMPLE_STEP_MM, dpi)
    end_step, end_zone = mm_to_px_float(END_STEP_MM, dpi), mm_to_px_float(END_ZONE_MM, dpi)
    zones, margin = no_text_zones(page.regions), mm_to_px_float(RULE_ZONE_MARGIN_MM, dpi)
    parts, count = [], 0
    for axis in page.analysis.axes:
        pts = line_curve(axis)
        if len(pts) < 2:
            continue
        pts = page.to_canvas(pts)
        # Строка внутри рисунка или фото (подпись на схеме, «строки» штриховки) — не опора поля.
        pts = pts[outside(pts, zones, margin)]
        if len(pts) < 2 or np.ptp(pts[:, 0]) < min_len:
            continue
        pts = resample(pts, 0, step, end_step, end_zone)
        target = float(np.median(pts[:, 1]))
        at = np.stack([pts[:, 0], np.full(len(pts), target)], axis=1)
        parts.append(Samples(at, pts[:, 1] - target, np.ones(len(pts)), pts, np.full(len(pts), SampleKind.LINE),
                             np.full(len(pts), len(parts))))  # fmt: skip
        count += 1
    return sum(parts, Samples.empty()), count


def aligned_sides(page: PageText, dpi: float) -> list[tuple[int, np.ndarray]]:
    """Выровненные стороны блоков страницы на холсте: ``(0 — левая, 1 — правая, точки (N, 2) сверху вниз)``.

    Сторона — гладкий тренд по телу блока (``core_left``/``core_right``), без недостоверных участков (``unreliable_*``
    — ступеньки по выносам), не короче :data:`MIN_SIDE_MM`.

    Args:
        page: Разбор страницы.
        dpi: Разрешение холста.

    Returns:
        Стороны.
    """
    min_len = mm_to_px_float(MIN_SIDE_MM, dpi)
    zones, margin = no_text_zones(page.regions), mm_to_px_float(RULE_ZONE_MARGIN_MM, dpi)
    sides = []
    for block, alignment in zip(page.analysis.blocks, page.analysis.alignments):
        envelope = block.envelope
        for which, stats, core, outer, unreliable in (
            (0, alignment.left, envelope.core_left, envelope.left, getattr(envelope, "unreliable_left", ())),
            (1, alignment.right, envelope.core_right, envelope.right, getattr(envelope, "unreliable_right", ())),
        ):
            curve = core if core is not None else outer
            if not stats.aligned or curve is None or len(curve) < 2:
                continue
            pts = np.asarray(curve, np.float64)
            if unreliable:
                pts = pts[~inside_spans(pts[:, 1], list(unreliable))]
            if len(pts) < 2:
                continue
            pts = page.to_canvas(pts)
            pts = pts[outside(pts, zones, margin)]
            if len(pts) >= 2 and np.ptp(pts[:, 1]) >= min_len:
                sides.append((which, pts[np.argsort(pts[:, 1])]))
    return sides


def column_groups(medians: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Группы сторон одной отбивки колонки: медианы x по порядку, соседние ближе ``tolerance`` — в одной группе.

    Args:
        medians: Медианы x сторон (одного вида — левые или правые).
        tolerance: Предельный зазор между соседними медианами, px.

    Returns:
        Индексы сторон по группам.
    """
    order = np.argsort(medians)
    groups, current = [], [order[0]] if len(order) else []
    for prev, idx in zip(order, order[1:]):
        if medians[idx] - medians[prev] <= tolerance:
            current.append(idx)
        else:
            groups.append(np.array(current))
            current = [idx]
    if current:
        groups.append(np.array(current))
    return groups


def side_samples(page: PageText, dpi: float) -> Samples:
    """Выборки по выровненным сторонам блоков: точка стороны должна встать на общую вертикаль отбивки колонки.

    Стороны одного вида (левые или правые) с медианами x ближе :data:`COLUMN_TOL_MM` — одна отбивка колонки, разбитая
    заголовками на блоки: цель у них общая — медиана x всех их точек. С отдельной вертикалью у каждого блока куски
    выпрямлялись, а отбивка колонки оставалась ступенчатой или наклонной (ХиП 1991/07 IMG_0090: стороны правой
    колонки на 226.9, 226.3, 224.3, 224.4 мм), а под коротким блоком внизу горизонтальную составляющую ничто не держало
    (06/IMG_0412). Короткие стороны (от :data:`MIN_SIDE_MM`) идут в дело, если группа в сумме не короче
    :data:`MIN_RUN_MM`.

    Args:
        page: Разбор страницы.
        dpi: Разрешение холста.

    Returns:
        Выборки составляющей ``dx``.
    """
    step = mm_to_px_float(SAMPLE_STEP_MM, dpi)
    end_step, end_zone = mm_to_px_float(END_STEP_MM, dpi), mm_to_px_float(END_ZONE_MM, dpi)
    min_len, tolerance = mm_to_px_float(MIN_RUN_MM, dpi), mm_to_px_float(COLUMN_TOL_MM, dpi)
    sides = aligned_sides(page, dpi)
    parts = []
    for which in (0, 1):
        own = [resample(pts, 1, step, end_step, end_zone) for kind, pts in sides if kind == which]
        if not own:
            continue
        medians = np.array([np.median(pts[:, 0]) for pts in own])
        for group in column_groups(medians, tolerance):
            members = [own[i] for i in group]
            if sum(np.ptp(pts[:, 1]) for pts in members) < min_len:
                continue
            # Выборки идут равномерно по высоте, поэтому медиана по всем точкам взвешена длиной сторон.
            target = float(np.median(np.concatenate([pts[:, 0] for pts in members])))
            for pts in members:
                at = np.stack([np.full(len(pts), target), pts[:, 1]], axis=1)
                parts.append(Samples(at, pts[:, 0] - target, np.ones(len(pts)), pts,
                                     np.full(len(pts), SampleKind.SIDE), np.full(len(pts), len(parts))))  # fmt: skip
    return sum(parts, Samples.empty())


def field_mode(count: int, supports: Samples, dpi: float) -> FieldMode:
    """Режим страницы по её опорам.

    Args:
        count: Строк и горизонтальных линеек в выборках.
        supports: Выборки строк и линеек ``dy`` страницы (по группам — длина опор, по ``at`` — размах по высоте).
        dpi: Разрешение холста.

    Returns:
        :class:`FieldMode`.
    """
    if count >= MIN_LINES:
        return FieldMode.FULL
    if count < SPARSE_MIN_LINES or not len(supports.at):
        return FieldMode.NONE
    length = sum(float(np.ptp(supports.at[supports.group == g, 0])) for g in np.unique(supports.group))
    span = float(np.ptp(supports.at[:, 1]))
    if length >= mm_to_px_float(SPARSE_MIN_LENGTH_MM, dpi) and span >= mm_to_px_float(SPARSE_MIN_SPAN_MM, dpi):
        return FieldMode.SPARSE
    return FieldMode.NONE


def far_zeros(zeros: Samples, supports: Samples, dpi: float) -> Samples:
    """Нулевые выборки дальше :data:`SPARSE_ZERO_MM` от всех опор (у опор поле решают они).

    Args:
        zeros: Нулевые выборки страницы (:func:`zero_samples`).
        supports: Опоры поля этой составляющей.
        dpi: Разрешение холста.

    Returns:
        Подмножество ``zeros``.
    """
    if not len(supports.at) or not len(zeros.at):
        return zeros
    distance, _ = cKDTree(supports.at).query(zeros.at)
    return zeros.take(distance > mm_to_px_float(SPARSE_ZERO_MM, dpi))


def zero_samples(rect: PageRect, dpi: float) -> Samples:
    """Выборки «смещение 0» по сетке на прямоугольнике страницы (страница без второго преобразования)."""
    step = mm_to_px_float(ZERO_STEP_MM, dpi)
    xs = np.arange(rect.x0, rect.x1 + 1e-9, step)
    ys = np.arange(rect.y0, rect.y1 + 1e-9, step)
    at = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    n = len(at)
    return Samples(at, np.zeros(n), np.full(n, ZERO_WEIGHT), at, np.full(n, SampleKind.ZERO), np.full(n, -1))


def dedupe_weights(samples: Samples, dpi: float) -> Samples:
    """Вес выборки делится на число разных групп (строк, сторон) среди выборок в радиусе :data:`DUPLICATE_RADIUS_MM`.

    Одна строка, пришедшая из двух пересекающихся блоков или из вырезок двух страниц у корешка, весит как одна, а
    разные цели этих дублей МНК усредняет. Соседние выборки своей же строки (густые на концах) дублями не считаются.

    Args:
        samples: Выборки одной составляющей.
        dpi: Разрешение холста.

    Returns:
        Выборки с поправленными весами.
    """
    if not len(samples):
        return samples
    near = cKDTree(samples.at).query_ball_point(samples.at, mm_to_px_float(DUPLICATE_RADIUS_MM, dpi))
    counts = np.array([len(np.unique(samples.group[n])) for n in near])
    return Samples(samples.at, samples.value, samples.weight / counts, samples.source, samples.kind, samples.group)


@dataclass(frozen=True)
class Grid:
    """Сетка узлов поля на холсте: узел ``(i, j)`` — в точке ``(j·step, i·step)``."""

    step: float
    nx: int
    ny: int

    @classmethod
    def over(cls, width: int, height: int, step: float) -> Grid:
        """Сетка, накрывающая холст ``width × height``."""
        return cls(step, int(np.ceil((width - 1) / step)) + 2, int(np.ceil((height - 1) / step)) + 2)

    @property
    def size(self) -> int:
        """Число узлов."""
        return self.nx * self.ny

    def interpolation(self, at: np.ndarray) -> sparse.csr_matrix:
        """Разреженная матрица билинейной интерполяции значений узлов в точки ``at`` (N, 2)."""
        gx = np.clip(at[:, 0] / self.step, 0, self.nx - 1 - 1e-9)
        gy = np.clip(at[:, 1] / self.step, 0, self.ny - 1 - 1e-9)
        j, i = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - j, gy - i
        rows = np.repeat(np.arange(len(at)), 4)
        cols = np.stack([i * self.nx + j, i * self.nx + j + 1, (i + 1) * self.nx + j, (i + 1) * self.nx + j + 1], 1)
        vals = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
        return sparse.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(len(at), self.size))

    def bending(self, along: np.ndarray, across: np.ndarray) -> sparse.csr_matrix:
        """Анизотропная энергия изгиба на узлах: ``Dxxᵀ·diag(λx)·Dxx + 2·Dxyᵀ·diag(λy)·Dxy + Dyyᵀ·diag(λy)·Dyy``.

        Изгиб вдоль строк (``Dxx``) — с жёсткостью ``along``, поперёк строк и кручение (``Dyy``, ``Dxy``) — с ``across``:
        поле может круто загибаться к концу строки, если так загибаются все соседние строки, но не от строки к строке.

        Args:
            along: Жёсткость ``λx`` в каждом узле (ny, nx); разность берёт жёсткость своего центра.
            across: Жёсткость ``λy`` в каждом узле (ny, nx).

        Returns:
            Разреженная матрица (узлы × узлы).
        """
        d2x, d2y = second_diff(self.nx), second_diff(self.ny)
        d1x, d1y = first_diff(self.nx), first_diff(self.ny)
        dxx = sparse.kron(sparse.identity(self.ny), d2x)
        dyy = sparse.kron(d2y, sparse.identity(self.nx))
        dxy = sparse.kron(d1y, d1x)
        # Центры разностей: ``Dxx`` — внутренние узлы по x, ``Dyy`` — по y, ``Dxy`` — середины ячеек.
        w_xx = sparse.diags(along[:, 1:-1].ravel())
        w_yy = sparse.diags(across[1:-1, :].ravel())
        cells = (across[:-1, :-1] + across[1:, :-1] + across[:-1, 1:] + across[1:, 1:]) / 4
        w_xy = sparse.diags(cells.ravel())
        return (dxx.T @ w_xx @ dxx + 2 * dxy.T @ w_xy @ dxy + dyy.T @ w_yy @ dyy).tocsr()


def first_diff(n: int) -> sparse.csr_matrix:
    """Первые разности ``(n-1) × n``."""
    return sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)).tocsr()


def second_diff(n: int) -> sparse.csr_matrix:
    """Вторые разности ``(n-2) × n``."""
    return sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)).tocsr()


@dataclass(frozen=True)
class ComponentFit:
    """Поле одной составляющей на узлах и итоговые веса выборок.

    Поля:
        nodes: Значения на узлах (ny, nx), px холста.
        weight: Веса выборок после робастных проходов (0 — выброшена).
        residual: Остаток выборок после подгонки, px.
    """

    nodes: np.ndarray
    weight: np.ndarray
    residual: np.ndarray


def fit_component(
    samples: Samples,
    grid: Grid,
    bending: sparse.csr_matrix,
    dpi: float,
    free_targets: bool = False,
    scale_weight: float = 0.0,
) -> ComponentFit:
    """Гладкое поле одной составляющей: МНК выборок плюс энергия изгиба, робастно (Хьюбер, отсев выбросов).

    С ``free_targets`` высота цели каждой строки — не жёсткая медиана, а свободная неизвестная (поправка ``c`` к
    медиане, своя у строки): решатель сам ставит строки так, чтобы поле было гладким. Вместе со штрафом
    ``scale_weight · (∂d/∂y)²`` это значит: строки выпрямляются сдвигом, а не сжатием межстрочья. С жёсткими
    медианами начала нижних строк у корешка, разошедшиеся веером, сжимались по высоте на 10–15 % вместе с буквами
    (ХиП 1991/03 IMG_0127, «…Комиссию о их заключении»).

    Args:
        samples: Выборки (веса уже поделены на плотность дублей).
        grid: Сетка узлов.
        bending: Энергия изгиба с жёсткостью (``Grid.bending``).
        dpi: Разрешение холста (пороги — в мм).
        free_targets: Цели строк (``SampleKind.LINE``, по ``group``) — свободные неизвестные.
        scale_weight: Вес штрафа за изменение масштаба по вертикали (разности узлов поля по y).

    Returns:
        Поле на узлах и веса выборок.
    """
    if not len(samples):
        return ComponentFit(np.zeros((grid.ny, grid.nx)), np.zeros(0), np.zeros(0))
    a = grid.interpolation(samples.at)
    q = bending + ZERO_PULL * sparse.identity(grid.size)
    if scale_weight > 0:
        d1y = sparse.kron(first_diff(grid.ny), sparse.identity(grid.nx))
        q = q + scale_weight * (d1y.T @ d1y)
    lines = (samples.kind == SampleKind.LINE) if free_targets else np.zeros(len(samples), bool)
    groups, index = np.unique(samples.group[lines], return_inverse=True)
    if len(groups):
        # Столбец на строку: выборка строки g получает в уравнение поправку её цели c_g.
        b = sparse.csr_matrix((np.ones(int(lines.sum())), (np.flatnonzero(lines), index)),
                              shape=(len(samples), len(groups)))  # fmt: skip
        a = sparse.hstack([a, b]).tocsr()
        q = sparse.block_diag([q, TARGET_PULL * sparse.identity(len(groups))]).tocsr()
    huber, outlier = mm_to_px_float(HUBER_MM, dpi), mm_to_px_float(OUTLIER_MM, dpi)
    robust = np.ones(len(samples))
    z = np.zeros(a.shape[1])
    for _ in range(IRLS_PASSES):
        w = samples.weight * robust
        aw = a.T.multiply(w)  # Aᵀ·W без плотной диагонали
        z = spsolve((aw @ a + q).tocsc(), aw @ samples.value)
        r = np.abs(a @ z - samples.value)
        # Хьюбер: квадрат до порога, линейно дальше; выбросы (перескок оси на соседнюю строку) — вес 0. Нулевые
        # выборки (цветная страница) не трогаются.
        robust = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-9))
        robust[(r > outlier) & (samples.kind != SampleKind.ZERO)] = 0.0
    nodes = z[: grid.size].reshape(grid.ny, grid.nx)
    return ComponentFit(nodes, samples.weight * robust, a @ z - samples.value)


def stiffness_of(step_px: float, dpi: float, smooth_mm: float) -> float:
    """Жёсткость ``λ`` из длины изгиба в мм: ``(L / шаг)^4 · κ``."""
    return (mm_to_px_float(smooth_mm, dpi) / step_px) ** 4 * STIFFNESS_SCALE


@dataclass(frozen=True)
class TextField:
    """Поле второго преобразования на холсте выпрямленного кадра — своё у каждой страницы, с разрывом по корешку.

    Поля:
        grid: Сетка узлов (общая разметка; у каждой страницы свои значения).
        spine_x: X линии разреза на холсте: левее — поле левой страницы, правее — правой.
        dx: Горизонтальная составляющая: узлы ``(2, ny, nx)`` по страницам, веса и остатки — по ``h_samples``.
        dy: Вертикальная составляющая, то же по ``v_samples``.
        h_samples: Выборки ``dx`` (стороны блоков и нули), сперва левой страницы, затем правой.
        v_samples: Выборки ``dy`` (оси строк и нули), в том же порядке.
        clipped: Модуль поля где-то превысил потолок и обрезан.
        blend: Полоса холста ``(x0, x1)`` у корешка без текста, где поля двух страниц плавно смешиваются; при
            ``x0 == x1`` — резкий разрыв на линии разреза.
    """

    grid: Grid
    spine_x: float
    dx: ComponentFit
    dy: ComponentFit
    h_samples: Samples
    v_samples: Samples
    clipped: bool
    blend: tuple[float, float]

    def dense(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Поле в каждом пикселе холста (бикубически по узлам своей страницы), float32 ``(dx, dy)``.

        Левее полосы смешивания — поле левой страницы, правее — правой, в полосе — плавный переход (вес — гладкая
        ступенька по x).
        """
        cols = (np.arange(width, dtype=np.float32) / self.grid.step)[None, :].repeat(height, 0)
        rows = (np.arange(height, dtype=np.float32) / self.grid.step)[:, None].repeat(width, 1)
        x0, x1 = self.blend
        if x1 > x0:
            t = np.clip((np.arange(width, dtype=np.float32) - x0) / (x1 - x0), 0.0, 1.0)
            weight = (t * t * (3 - 2 * t))[None, :]  # доля правой страницы
        else:
            weight = (np.arange(width) >= int(np.ceil(self.spine_x))).astype(np.float32)[None, :]
        # Столбцы, где нужна каждая страница (с весом > 0).
        need = [np.flatnonzero(weight[0] < 1), np.flatnonzero(weight[0] > 0)]
        out = []
        for comp in (self.dx, self.dy):
            plane = np.zeros((height, width), np.float32)
            for page, (nodes, idx) in enumerate(zip(comp.nodes, need)):
                if not len(idx):
                    continue
                part = slice(int(idx[0]), int(idx[-1]) + 1)
                values = cv2.remap(nodes.astype(np.float32), cols[:, part], rows[:, part], cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
                w = weight[:, part] if page == 1 else 1 - weight[:, part]
                plane[:, part] += values * w
            out.append(plane)
        return out[0], out[1]


def stiffness_map(grid: Grid, spine_x: float, dpi: float, smooth_mm: float, gutter_mm: float) -> np.ndarray:
    """Жёсткость в узлах: длина изгиба ``gutter_mm`` у линии корешка, линейно растущая до ``smooth_mm`` в теле страницы.

    Args:
        grid: Сетка.
        spine_x: X линии корешка на холсте.
        dpi: Разрешение холста.
        smooth_mm: Длина изгиба в теле страницы, мм.
        gutter_mm: Длина изгиба у корешка, мм.

    Returns:
        ``λ`` по узлам (ny, nx).
    """
    dist_mm = np.abs(np.arange(grid.nx) * grid.step - spine_x) / mm_to_px_float(1.0, dpi)
    share = np.clip((dist_mm - GUTTER_ZONE_MM) / GUTTER_RAMP_MM, 0.0, 1.0)
    length = gutter_mm + (smooth_mm - gutter_mm) * share
    return np.repeat(stiffness_of(grid.step, dpi, 1.0) * length[None, :] ** 4, grid.ny, axis=0)


def fit_field(
    h_samples: Samples,
    v_samples: Samples,
    layout: Layout,
    dpi: float,
    step_mm: float,
    smooth_mm: float,
    gutter_mm: float | None = None,
    across_mm: float = ACROSS_SMOOTH_MM,
    scale_weight: float = SCALE_WEIGHT,
) -> TextField:
    """Поле по страницам: у каждой своё (по выборкам, лежащим по её сторону линии разреза), на корешке — разрыв.

    Общее поле на разворот тянуло изгиб строк одной страницы у корешка в концы строк другой (ХиП 1991/01 IMG_1057:
    правый конец левой страницы у корешка получил +0.46 мм от загиба правой).

    Args:
        h_samples: Выборки ``dx``.
        v_samples: Выборки ``dy``.
        layout: Расклад холста.
        dpi: Разрешение холста.
        step_mm: Шаг узлов, мм.
        smooth_mm: Длина изгиба в теле страницы, мм.
        gutter_mm: Длина изгиба у корешка, мм; None — та же, что в теле (поле однородное).
        across_mm: Длина изгиба поперёк строк, мм.
        scale_weight: Вес штрафа за изменение масштаба по вертикали (0 — цели строк жёсткие медианы, без штрафа).

    Returns:
        Поле.
    """
    step = mm_to_px_float(step_mm, dpi)
    grid = Grid.over(layout.width, layout.height, step)
    stiffness = stiffness_map(grid, layout.spine_x, dpi, smooth_mm, smooth_mm if gutter_mm is None else gutter_mm)
    across = np.full_like(stiffness, stiffness_of(grid.step, dpi, across_mm))
    bending = grid.bending(stiffness, across)
    comps = []
    ordered = []
    for samples in (h_samples, v_samples):
        left = samples.at[:, 0] < layout.spine_x
        parts = [dedupe_weights(samples.take(sel), dpi) for sel in (left, ~left)]
        vertical = samples is v_samples
        fits = [fit_component(part, grid, bending, dpi, free_targets=vertical and scale_weight > 0,
                              scale_weight=scale_weight if vertical else 0.0) for part in parts]  # fmt: skip
        comps.append(ComponentFit(np.stack([f.nodes for f in fits]), np.concatenate([f.weight for f in fits]),
                                  np.concatenate([f.residual for f in fits])))  # fmt: skip
        ordered.append(parts[0] + parts[1])
    dx, dy = comps
    cap = mm_to_px_float(MAX_SHIFT_MM, dpi)
    norm = np.hypot(dx.nodes, dy.nodes)
    clipped = bool((norm > cap).any())
    if clipped:
        k = np.minimum(1.0, cap / np.maximum(norm, 1e-9))
        dx, dy = (ComponentFit(c.nodes * k, c.weight, c.residual) for c in (dx, dy))
    blend = blend_band(ordered[0] + ordered[1], layout.spine_x, dpi)
    return TextField(grid, layout.spine_x, dx, dy, ordered[0], ordered[1], clipped, blend)


def blend_band(samples: Samples, spine_x: float, dpi: float) -> tuple[float, float]:
    """Полоса смешивания полей страниц у корешка: между краями текста двух страниц, не шире ``±BLEND_MAX_MM``.

    Поля страниц у корешка расходятся (ХиП 1991/08 IMG_0329: +0.11 и −0.65 мм), и при резком разрыве бумага у
    корешка двоилась или выпадала, а линия разреза на итоге ломалась. В пустой полосе между текстами поля можно
    смешать: текст каждой страницы остаётся под своим полем.

    Args:
        samples: Выборки строк и сторон обеих страниц (точки ``source`` — на выпрямленном кадре).
        spine_x: X линии корешка.
        dpi: Разрешение холста.

    Returns:
        ``(x0, x1)``; ``x0 == x1 == spine_x`` — смешивать негде (текст у самого корешка).
    """
    margin, reach = mm_to_px_float(BLEND_TEXT_MARGIN_MM, dpi), mm_to_px_float(BLEND_MAX_MM, dpi)
    real = samples.kind != SampleKind.ZERO
    xs, groups = samples.source[real, 0], samples.group[real]
    # Край текста страницы у корешка — по концам строк и сторон (у каждой группы — ближняя к корешку точка), с
    # перцентилем по группам: одна ложная ось, ушедшая в тень сгиба до самого корешка (08/IMG_0329), его не двигает.
    left_ends = [xs[groups == g].max() for g in np.unique(groups[xs < spine_x])]
    right_ends = [xs[groups == g].min() for g in np.unique(groups[xs >= spine_x])]
    x0 = spine_x - reach
    if left_ends:
        x0 = max(float(np.percentile(left_ends, 100 - BLEND_OUTLIER_PCT)) + margin, x0)
    x1 = spine_x + reach
    if right_ends:
        x1 = min(float(np.percentile(right_ends, BLEND_OUTLIER_PCT)) - margin, x1)
    if x1 - x0 < mm_to_px_float(BLEND_MIN_MM, dpi):
        # Узкая полоса — переход круче резкого разрыва.
        return spine_x, spine_x
    return x0, x1


def coons_at(patches: tuple[Patch, Patch], layout: Layout, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Точки итога v2 по Кунсу для точек выпрямленного кадра: левее линии корешка — левая страница, правее — правая.

    Args:
        patches: Области левой и правой страницы.
        layout: Расклад холста.
        xs: X точек выпрямленного кадра (N,).
        ys: Y точек (N,).

    Returns:
        Точки (N, 2) px итога v2.
    """
    out = np.empty((len(xs), 2))
    left = xs < layout.spine_x
    for patch, rect, sel in zip(patches, layout.rects, (left, ~left)):
        if sel.any():
            u, v = rect_fractions(rect, xs[sel], ys[sel])
            out[sel] = patch.points(u, v)
    return out


def compose_maps(
    patches: tuple[Patch, Patch], layout: Layout, dx: np.ndarray, dy: np.ndarray, coarse: int = COMPOSE_STEP_PX
) -> tuple[np.ndarray, np.ndarray]:
    """Составная карта: пиксель холста → точка выпрямленного кадра ``(X + dx, Y + dy)`` → точка итога v2 по Кунсу.

    Кунс считается заново в смещённых точках (а не выборкой из карты первого преобразования): у краёв холста
    смещённая точка может выйти за холст, а поверхность продолжается за область и там. Считается он на сетке через
    ``coarse`` px и доводится до полного размера бикубически (карта гладкая; в каждой точке — 2.7 с на кадр, на сетке
    4 px — в 16 раз меньше точек). Где карта рвётся (шов страниц: Кунс левой и правой, поля двух страниц) или ломается
    (стороны областей: за ними Кунс продолжается касательной), доводка смазала бы её, поэтому полосы
    :data:`COMPOSE_SEAM_PX` вокруг шва и сторон считаются точно в каждом пикселе.

    Args:
        patches: Области левой и правой страницы.
        layout: Расклад холста.
        dx: Горизонтальная составляющая поля (H, W).
        dy: Вертикальная составляющая (H, W).
        coarse: Шаг сетки, px; 1 — в каждом пикселе.

    Returns:
        ``(map_x, map_y)`` float32 размера холста — в координаты итога v2.
    """
    h, w = layout.height, layout.width
    # Узлы сетки (с последним столбцом и строкой холста, чтобы доводка не выходила за край).
    gx = np.unique(np.r_[np.arange(0, w, coarse), w - 1])
    gy = np.unique(np.r_[np.arange(0, h, coarse), h - 1])
    xs = (gx[None, :] + dx[np.ix_(gy, gx)]).ravel()
    ys = (gy[:, None] + dy[np.ix_(gy, gx)]).ravel()
    nodes = coons_at(patches, layout, xs, ys).reshape(len(gy), len(gx), 2).astype(np.float32)
    # Доводка: координата пикселя в долях узлов (узлы неравномерны только у последнего столбца и строки).
    fx = np.interp(np.arange(w), gx, np.arange(len(gx))).astype(np.float32)
    fy = np.interp(np.arange(h), gy, np.arange(len(gy))).astype(np.float32)
    cols, rows = np.meshgrid(fx, fy)
    maps = [cv2.remap(np.ascontiguousarray(nodes[..., c]), cols, rows, cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_REPLICATE) for c in (0, 1)]  # fmt: skip
    if coarse > 1:
        # Точно — полосы, где карта рвётся или ломается: у столбца разреза, у кривой X + dx = линия корешка (разрыв
        # страниц) и у всех сторон прямоугольников страниц в смещённых координатах (за стороной Кунс переходит с
        # кривой на касательную — излом; без этого на поле левой страницы ошибка доводки доходила до 1.9 px).
        split = int(np.ceil(layout.spine_x))
        col_idx = np.arange(w, dtype=np.float32)[None, :]
        row_idx = np.arange(h, dtype=np.float32)[:, None]
        shifted_x, shifted_y = col_idx + dx, row_idx + dy
        band = (np.abs(col_idx - split) <= COMPOSE_SEAM_PX) | (np.abs(shifted_x - layout.spine_x) <= COMPOSE_SEAM_PX)
        for rect in layout.rects:
            for edge in (rect.x0, rect.x1):
                band |= np.abs(shifted_x - edge) <= COMPOSE_SEAM_PX
            for edge in (rect.y0, rect.y1):
                band |= np.abs(shifted_y - edge) <= COMPOSE_SEAM_PX
        r, c = np.nonzero(band)
        exact = coons_at(patches, layout, c + dx[r, c].astype(np.float64), r + dy[r, c].astype(np.float64))
        maps[0][r, c], maps[1][r, c] = exact[:, 0], exact[:, 1]
    return maps[0], maps[1]
