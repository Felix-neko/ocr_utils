"""Таблица по линейкам: морфология вместо сети.

ЗАЧЕМ ИМЕННО ЭТО ПЕРВЫМ. Таблицы в этих журналах ЛИНОВАНЫ: колонки разделены вертикальными
линейками, шапка отбита двойной горизонтальной, а вот внешней рамки слева и справа часто
нет вовсе. Значит, самый прямой признак таблицы — не «сеть узнала таблицу», а собственно
линейки, и они же сразу дают сетку ячеек, за которой иначе пришлось бы идти ко второй
модели. Это тот же приём, на котором стоит img2table, и он не требует ни GPU, ни весов.

КАК. Бинаризация Otsu, затем открытие длинным горизонтальным и длинным вертикальным ядром —
остаётся только то, что тянется на много миллиметров подряд, то есть линейки. Дальше
компоненты связности дают отрезки, а объединение отрезков, стоящих рядом, — таблицу.

МАСШТАБ. Всё считается на копии 150 dpi. Замер на трёх известных полосах: разжатие через
``Image.draft`` занимает 0.1 с, линейки при ядре 40-60 px (7-10 мм) находятся все, число
компонент устойчиво (страница 28: 9 горизонтальных и 9 вертикальных при ядрах 40 и 60).
На 600 dpi та же морфология стоила бы шестнадцатикратно дороже и не дала бы ничего нового:
рамку таблицы не нужно знать точнее полумиллиметра, её всё равно берут с полями.

РАЗМЕРЫ — В МИЛЛИМЕТРАХ, а не в долях кадра. Полосы пака различаются по размеру (после
распрямления от 3589x6445 до 3830x5892 px), и доля кадра означала бы разный физический
порог на разных полосах. Пересчёт в пиксели — единственное место, где живёт dpi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box, TableBox, intersection, union

logger = logging.getLogger(__name__)

# Рабочее разрешение поиска линеек.
WORK_DPI = 150

# Самая короткая линейка, которую считаем линейкой: 8 мм. Короче бывают только подчёркивания
# в тексте и обрывки штриховки. При 150 dpi это 47 px — середина проверенного диапазона 40-60.
MIN_RULE_MM = 8.0

# Самая толстая: 1.0 мм. Толщина меряется как ПЛОЩАДЬ, ДЕЛЁННАЯ НА ДЛИНУ, а не по
# габаритному прямоугольнику. Замер на 10 полосах пака (132 линейки): средняя толщина
# держится в 2.0-3.0 px при 150 dpi (0.34-0.51 мм) и по габариту — до 12 px, потому что
# у чуть наклонной линейки прямоугольник шире самого штриха. По габариту порог 1.2 мм
# выбрасывал настоящие линейки таблицы (страница 28, «Таблица 2»: три вертикали шириной
# 9-11 px), по средней толщине они проходят с запасом.
MAX_RULE_THICKNESS_MM = 1.0

# Разрыв, через который две линейки считаются одной таблицей: 6 мм. Больше — и в таблицу
# затягивает соседний текстовый блок с подчёркиванием, меньше — и таблица без внешней рамки
# распадается на отдельные графы.
CLUSTER_GAP_MM = 6.0

# Таблица не меньше этого по обеим сторонам: 25 мм. Отсекает рамочки вокруг формул и
# одиночные подчёркнутые заголовки.
MIN_TABLE_SIDE_MM = 25.0

# Сколько линеек каждого направления обязано быть в таблице. Две горизонтальных и одна
# вертикальная — это минимальная таблица из двух граф; всё, что меньше, — не таблица.
MIN_HORIZONTAL = 2
MIN_VERTICAL = 1

# Сквозная линейка — та, что перекрывает не меньше этой доли ширины таблицы; таких обязано
# найтись хотя бы две. Это главный отсев ложных находок, и он замерен на 15 скоплениях
# линеек: у всех 13 настоящих таблиц сквозных линеек 2-4 (у таблицы всегда есть верхняя и
# нижняя границы либо отбивка шапки), а у штрихового рисунка картотечного шкафа, который
# морфология тоже считает скоплением линеек, — 1 и 0.
#
# Считать по МЕДИАННОЙ длине линейки нельзя, хотя соблазнительно: у таблицы с многоуровневой
# шапкой (1972/06, «Форма лимитной карточки») девять линеек, из них сквозных две, а семь
# коротких — разделители подзаголовков, и медиана даёт 0.15, как у рисунка.
LONG_RULE_SPAN = 0.8
MIN_LONG_RULES = 2

# Доля площади под краской, выше которой находка считается не таблицей, а полутоновой
# печатью. Замер там же: таблицы 0.08-0.17, штриховой рисунок 0.31-0.38.
MAX_INK_SHARE = 0.25

# Разброс углов линеек, выше которого таблица помечается как «кривая»: полградуса. Замер
# порога — на паке; флаг ни на что не влияет, кроме отчёта, и нужен, чтобы отделить
# «алгоритм не нашёл» от «страница поехала».
CURVED_SPREAD_DEG = 0.5


def mm_to_px(millimeters: float, dpi: int) -> int:
    return max(1, round(millimeters * dpi / 25.4))


@dataclass(frozen=True)
class Segment:
    """Отрезок линейки в координатах рабочей копии."""

    box: Box
    horizontal: bool
    angle_deg: float  # положительный — правый конец ниже левого (поворот по часовой)

    @property
    def length(self) -> int:
        return self.box.width if self.horizontal else self.box.height


@dataclass(frozen=True)
class Lines:
    horizontal: list[Segment]
    vertical: list[Segment]
    horizontal_mask: np.ndarray
    vertical_mask: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        return cv2.bitwise_or(self.horizontal_mask, self.vertical_mask)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Краска белым на чёрном. Otsu: фон полос уже выровнен предыдущими шагами конвейера."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def _axis_mask(binary: np.ndarray, length_px: int, horizontal: bool) -> np.ndarray:
    """Что тянется вдоль оси не меньше ``length_px`` подряд."""
    shape = (1, length_px) if horizontal else (length_px, 1)
    kernel = np.ones(shape, np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    # Закрытие тем же ядром сшивает линейку, разорванную буквой или дырой в бумаге.
    #
    # ``borderValue=0`` обязателен. По умолчанию OpenCV считает, что за краем кадра всё
    # белое, и тогда сжатие после расширения у самого края не отрабатывает: линейка,
    # начинающаяся в 30 px от края, расползается ДО края и тянет за собой рамку таблицы.
    # На синтетическом тесте это давало рамку во весь кадр вместо рамки по линейкам.
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _segments(mask: np.ndarray, horizontal: bool, max_thickness_px: int) -> list[Segment]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    found: list[Segment] = []
    for index in range(1, count):
        left, top, width, height, area = stats[index]
        length = width if horizontal else height
        if area / max(length, 1) > max_thickness_px:
            continue
        box = Box(int(left), int(top), int(left + width), int(top + height))
        found.append(Segment(box, horizontal, _angle_of(labels, index, horizontal, box)))
    return found


def _angle_of(labels: np.ndarray, index: int, horizontal: bool, box: "Box | None" = None) -> float:
    """Наклон отрезка: линейная регрессия по пикселям компоненты.

    Через регрессию, а не через ``minAreaRect``: у почти горизонтального отрезка толщиной
    в два пикселя прямоугольник минимальной площади скачет между 0 и 90 градусами, а
    регрессия по всем пикселям устойчива и заодно бесплатно даёт наклон.

    ``box`` — габарит компоненты: пиксели ищутся только в нём. Без него сравнение идёт по
    всей полосе на каждую компоненту, и на шестистах фрагментах четвёртой версии это
    полторы секунды из двух с половиной на полосу.
    """
    if box is not None:
        window = labels[box.slice]
        rows, columns = np.nonzero(window == index)
        rows = rows + box.y0
        columns = columns + box.x0
    else:
        rows, columns = np.nonzero(labels == index)
    if rows.size < 8:
        return 0.0
    if horizontal:
        along, across = columns.astype(np.float64), rows.astype(np.float64)
    else:
        along, across = rows.astype(np.float64), columns.astype(np.float64)
    spread = along.max() - along.min()
    if spread < 4:
        return 0.0
    slope = np.polyfit(along, across, 1)[0]
    angle = float(np.degrees(np.arctan(slope)))
    # У вертикали положительный наклон «вправо вниз» означает поворот ПРОТИВ часовой,
    # поэтому знак приводится к общей валюте — «на сколько повернуть по часовой».
    return angle if horizontal else -angle


def find_lines(gray: np.ndarray, dpi: int = WORK_DPI, min_length_px: "int | None" = None) -> Lines:
    """Линейки страницы или вырезанной таблицы.

    ``min_length_px`` задают, когда кадр меньше самой линейки: внутри ячейки шириной 8 мм
    обрывок разделителя короче общего порога, но убрать его надо (см. ``strip_rules``).
    """
    binary = binarize(gray)
    length_px = min_length_px or mm_to_px(MIN_RULE_MM, dpi)
    thickness_px = mm_to_px(MAX_RULE_THICKNESS_MM, dpi)
    horizontal_mask = _axis_mask(binary, length_px, horizontal=True)
    vertical_mask = _axis_mask(binary, length_px, horizontal=False)
    return Lines(
        horizontal=_segments(horizontal_mask, True, thickness_px),
        vertical=_segments(vertical_mask, False, thickness_px),
        horizontal_mask=horizontal_mask,
        vertical_mask=vertical_mask,
    )


def skew_of(lines: Lines) -> tuple[float, float]:
    """Медианный угол линеек и разброс углов, в градусах по часовой.

    Медиана, а не среднее: одна линейка, слипшаяся с подчёркиванием соседнего абзаца, даёт
    выброс в несколько градусов, и среднее увело бы за собой всю таблицу.
    """
    weights = [(segment.angle_deg, segment.length) for segment in lines.horizontal if segment.length > 0]
    if not weights:
        return 0.0, 0.0
    angles = np.array([angle for angle, _ in weights])
    longest = sorted(weights, key=lambda item: -item[1])[: max(3, len(weights) // 2)]
    principal = np.array([angle for angle, _ in longest])
    return float(np.median(principal)), float(np.percentile(angles, 90) - np.percentile(angles, 10))


@dataclass(frozen=True)
class ClusterPolicy:
    """Пороги кластеризации одним объектом, чтобы третья версия детектора отличалась от второй
    набором чисел, а не копией кода.

    ``DEFAULT_POLICY`` повторяет константы модуля бит в бит: пока никто не подаёт другую
    политику, поведение остаётся прежним, и это проверяется побайтовым сравнением ``находки.csv``.
    """

    gap_mm: float = CLUSTER_GAP_MM
    # Допустимые размеры — СПИСОК вариантов «длинная сторона, короткая сторона», и скопление
    # проходит, если подходит хотя бы под один. Одной парой чисел тут не обойтись: правило
    # «обе стороны не меньше 25 мм» не видит широкую короткую шапку бланка (130x16 мм), а
    # правило «длинная не меньше 40, короткая не меньше 10» теряет небольшую квадратную
    # таблицу 39x29 мм, которую старое правило находило. Нужны оба, и это выяснилось на
    # синтетическом тесте, а не на паке.
    min_sides_mm: tuple[tuple[float, float], ...] = ((MIN_TABLE_SIDE_MM, MIN_TABLE_SIDE_MM),)
    min_horizontal: int = MIN_HORIZONTAL
    min_vertical: int = MIN_VERTICAL
    long_rule_span: float = LONG_RULE_SPAN
    min_long_rules: int = MIN_LONG_RULES
    long_rules_any_axis: bool = False
    max_ink_share: float = MAX_INK_SHARE
    merge_share: float = 0.3
    merge_max_growth: float = 1.6


DEFAULT_POLICY = ClusterPolicy()


@dataclass(frozen=True)
class Cluster:
    """Скопление линеек: рамка и ТЕ САМЫЕ отрезки, из которых она получилась.

    Отрезки носятся вместе с рамкой не для красоты. Раньше метрики считались один раз, при
    создании находки, и слияние двух находок их не пересчитывало — после слияния ``long_rules``,
    ``h_span``, ``ink`` и ``skew_deg`` оставались от первой рамки и относились уже к другой
    ширине. Если кластер помнит свои отрезки, слить два кластера означает сложить два списка,
    а метрики посчитать заново — соврать негде.
    """

    box: Box
    horizontal: list[Segment]
    vertical: list[Segment]

    @property
    def segments(self) -> list[Segment]:
        return self.horizontal + self.vertical


def _cluster_box(segments: list[Segment], width: int, height: int) -> Box:
    """Рамка по САМИМ линейкам, а не по раздутой маске: иначе таблица вырастет на зазор склейки."""
    return Box(
        min(s.box.x0 for s in segments),
        min(s.box.y0 for s in segments),
        max(s.box.x1 for s in segments),
        max(s.box.y1 for s in segments),
    ).clipped(width, height)


def long_rules_of(cluster: Cluster, policy: ClusterPolicy = DEFAULT_POLICY) -> int:
    """Сколько линеек перекрывают свою сторону рамки хотя бы на ``long_rule_span``.

    ``long_rules_any_axis`` включает в счёт и вертикали. Замер на 18 полосах, где таблица есть,
    а детектор её не нашёл: у шести из них сквозных горизонталей одна (внешней рамки сверху и
    снизу просто нет), зато сквозных вертикалей от двух до десяти. Считать только горизонтали —
    значит по построению не видеть бланки, размеченные вертикалями граф.
    """
    count = int(
        (
            np.array([s.length for s in cluster.horizontal], dtype=float) / max(1, cluster.box.width)
            >= policy.long_rule_span
        ).sum()
    )
    if policy.long_rules_any_axis and cluster.vertical:
        count += int(
            (
                np.array([s.length for s in cluster.vertical], dtype=float) / max(1, cluster.box.height)
                >= policy.long_rule_span
            ).sum()
        )
    return count


def metrics_of(
    cluster: Cluster,
    policy: ClusterPolicy = DEFAULT_POLICY,
    binary_ink: "np.ndarray | None" = None,
    horizontal_mask: "np.ndarray | None" = None,
    vertical_mask: "np.ndarray | None" = None,
) -> tuple[float, float, dict[str, float]]:
    """Счёт, наклон и метрики кластера — чистая функция от его отрезков и рамки."""
    box = cluster.box
    spans = np.array([s.length for s in cluster.horizontal], dtype=float) / max(1, box.width)
    ink = 0.0
    if binary_ink is not None:
        ink = float(np.count_nonzero(binary_ink[box.slice])) / max(1, box.area)
    empty = np.zeros((1, 1), np.uint8)
    local = Lines(
        cluster.horizontal,
        cluster.vertical,
        horizontal_mask if horizontal_mask is not None else empty,
        vertical_mask if vertical_mask is not None else empty,
    )
    angle, spread = skew_of(local)
    metrics = {
        "h_lines": float(len(cluster.horizontal)),
        "v_lines": float(len(cluster.vertical)),
        "long_rules": float(long_rules_of(cluster, policy)),
        "h_span": float(spans.max()) if spans.size else 0.0,
        "ink": ink,
        "angle_spread": spread,
        "curved": float(spread > CURVED_SPREAD_DEG),
    }
    return min(1.0, len(cluster.horizontal) / 4.0), angle, metrics


def cluster_tables(
    lines: Lines,
    dpi: int = WORK_DPI,
    binary_ink: "np.ndarray | None" = None,
    policy: ClusterPolicy = DEFAULT_POLICY,
    rejected: "list[tuple[Box, str]] | None" = None,
    grouping: str = "dilate",
) -> list[TableBox]:
    """Отрезки → таблицы: соседние линейки склеиваются, лишние скопления отбрасываются.

    В ``rejected``, если он передан, складываются выброшенные скопления с причиной словами.
    Без него отсеянная кластеризацией находка исчезает бесследно, и понять, почему таблица на
    полосе не нашлась, можно только повторив весь отбор руками.

    ``grouping`` — как линейки собираются в скопления: ``"dilate"`` — склейка дилатацией на
    ``policy.gap_mm`` (вторая и третья версии), ``"cores"`` — связные ядра по пересечениям и
    продолжениям (четвёртая, см. ``rules.cores``). Отбор скоплений после этого общий.
    """
    height, width = lines.horizontal_mask.shape[:2]
    allowed = [(mm_to_px(long_mm, dpi), mm_to_px(short_mm, dpi)) for long_mm, short_mm in policy.min_sides_mm]

    if grouping == "cores":
        from ocr_utils.page_layout.tables.rules import cores

        regions: list[tuple[Box, list[Segment], list[Segment]]] = []
        for group in cores(lines, dpi):
            box = _cluster_box(group, width, height)
            regions.append((box, [s for s in group if s.horizontal], [s for s in group if not s.horizontal]))
    else:
        gap_px = mm_to_px(policy.gap_mm, dpi)
        glued = cv2.dilate(lines.mask, np.ones((gap_px, gap_px), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(glued, 8)
        regions = []
        for index in range(1, count):
            left, top, box_width, box_height, _ = stats[index]
            region = Box(int(left), int(top), int(left + box_width), int(top + box_height))
            inside_h = [s for s in lines.horizontal if _center_inside(s, region)]
            inside_v = [s for s in lines.vertical if _center_inside(s, region)]
            regions.append((region, inside_h, inside_v))

    def too_small(box: Box) -> bool:
        long_side, short_side = max(box.width, box.height), min(box.width, box.height)
        return not any(long_side >= long_px and short_side >= short_px for long_px, short_px in allowed)

    def note(box: Box, reason: str) -> None:
        if rejected is not None:
            rejected.append((box, reason))

    clusters: list[Cluster] = []
    for region, inside_h, inside_v in regions:
        if too_small(region):
            note(region, f"скопление {region.width}x{region.height} px мельче порога")
            continue
        if len(inside_h) < policy.min_horizontal or len(inside_v) < policy.min_vertical:
            note(region, f"линеек мало: {len(inside_h)} горизонтальных, {len(inside_v)} вертикальных")
            continue
        cluster = Cluster(_cluster_box(inside_h + inside_v, width, height), inside_h, inside_v)
        if too_small(cluster.box):
            note(cluster.box, f"рамка {cluster.box.width}x{cluster.box.height} px мельче порога")
            continue
        if long_rules_of(cluster, policy) < policy.min_long_rules:
            note(cluster.box, f"сквозных линеек {long_rules_of(cluster, policy)} < {policy.min_long_rules}")
            continue
        if binary_ink is not None:
            share = float(np.count_nonzero(binary_ink[cluster.box.slice])) / max(1, cluster.box.area)
            if share > policy.max_ink_share:
                note(cluster.box, f"краски {share:.2f} > {policy.max_ink_share}")
                continue
        clusters.append(cluster)

    merged = merge_clusters(sorted(clusters, key=lambda item: item.box.y0), width, height, policy)
    found: list[TableBox] = []
    for cluster in merged:
        score, angle, metrics = metrics_of(cluster, policy, binary_ink, lines.horizontal_mask, lines.vertical_mask)
        found.append(TableBox(box=cluster.box, score=score, source="ruling", skew_deg=angle, metrics=metrics))
    return found


def merge_clusters(
    clusters: list[Cluster], width: int, height: int, policy: ClusterPolicy = DEFAULT_POLICY
) -> list[Cluster]:
    """Слить скопления, которые говорят об одном и том же.

    Одна таблица иногда распадается на два кластера — например, если шапка отбита от тела
    широким просветом. Разбирать это в отчёте как две находки нечестно: рамки перекрываются,
    и человек видит одну таблицу.

    Доля перекрытия считается от МЕНЬШЕЙ площади, и это намеренно: вложенная рамка — почти
    всегда та же таблица, увиденная дважды. Но у такого правила есть обратная сторона: узкая
    боковая таблица, чей габарит целиком лежит внутри габарита широкой, сливалась с ней в один
    огромный бокс. Поэтому к доле добавлен потолок роста: объединение не должно быть больше
    ``merge_max_growth`` площадей большей из двух рамок.
    """
    merged: list[Cluster] = []
    for cluster in clusters:
        for index, existing in enumerate(merged):
            overlap = intersection(existing.box, cluster.box)
            if overlap is None:
                continue
            share = overlap.area / min(existing.box.area, cluster.box.area)
            if share < policy.merge_share:
                continue
            combined = union([existing.box, cluster.box])
            assert combined is not None
            if combined.area > policy.merge_max_growth * max(existing.box.area, cluster.box.area):
                continue
            segments = existing.segments + cluster.segments
            merged[index] = Cluster(
                _cluster_box(segments, width, height),
                existing.horizontal + cluster.horizontal,
                existing.vertical + cluster.vertical,
            )
            break
        else:
            merged.append(cluster)
    return merged


def _center_inside(segment: Segment, region: Box) -> bool:
    center_x = (segment.box.x0 + segment.box.x1) // 2
    center_y = (segment.box.y0 + segment.box.y1) // 2
    return region.x0 <= center_x <= region.x1 and region.y0 <= center_y <= region.y1
