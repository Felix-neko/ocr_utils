"""Вид объекта по линейкам: таблица, схема или рисунок.

ЗАЧЕМ. Блок-схему и таблицу лечат по-разному: у таблицы графу можно расширить под
горизонтальный текст, у схемы блок обязан остаться на месте, а повёрнутый текст вписывают в
него. Поэтому детектор четвёртой версии не отвергает схемы, а называет их: находка несёт
``kind``, и потребители фильтруют.

ПРИЗНАКИ — ПО ЛИНЕЙКАМ ЯДРА. У таблицы линейки образуют решётку: каждая горизонталь
пересекает почти все вертикали, вертикали стоят на немногих общих x (графы), есть сквозные
горизонтали во всю ширину. У блок-схемы из k прямоугольников каждая горизонталь пересекает
только две вертикали своего прямоугольника (плотность пересечений около 1/k), вертикали
стоят на 2k разных x, сквозных горизонталей нет. График и чертёж отделяет пол третьей версии:
мало букв и пустые ячейки.

Пороги стоят в пустых промежутках распределений по размеченным полосам (см. README пакета,
раздел про четвёртую версию); замер печатает команда ``compare-detector``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ocr_utils.scan_markup.table_detection.refine import crosses
from ocr_utils.scan_markup.table_detection.ruling import Segment, mm_to_px
from ocr_utils.scan_markup.table_detection.verify import FLOOR_FILLED_CELLS, FLOOR_GLYPH_OF_INK, Features
from ocr_utils.scan_markup.table_detection.geometry import KIND_DIAGRAM, KIND_DRAWING, KIND_TABLE, Box

# Допуск, с которым две вертикали считаются стоящими на одном x (одна графа): 1 мм.
ALIGN_TOL_MM = 1.0

# Допуск на пересечение — тот же, что в ``refine``.
CROSS_TOL_MM = 1.5

# Сквозная горизонталь — не короче этой доли ширины рамки.
LONG_SPAN = 0.8

# --- Пороги (калибруются по разметке, см. докстринг модуля) ---
# Схема — это МНОГО линеек (коробок), редкие пересечения и почти нет сквозных горизонталей.
# Замер по затравкам 190 размеченных полос: у схем медиана 12 горизонталей и 18 вертикалей,
# плотность пересечений 0.20 (три четверти ниже 0.71), сквозных горизонталей 0.27; у таблиц
# плотность 0.79 (три четверти выше 0.67), сквозных 0.75 (три четверти выше 0.60). Рамка
# страницы с колонками (4–6 горизонталей, 3–11 вертикалей, сквозных 0.5–1.0) и баннер рубрики
# (9 горизонталей, 4–7 вертикалей, 23 мм высотой) под схему не подходят по числу линеек,
# сквозным и размеру.
DIAGRAM_MIN_RULES = 12
DIAGRAM_MIN_VERTICALS = 6
DIAGRAM_MAX_CROSS_DENSITY = 0.45
DIAGRAM_MAX_LONG_SHARE = 0.3
DIAGRAM_MIN_SIDE_MM = 30.0

# Доля вертикалей, идущих хотя бы на треть высоты рамки. У таблицы вертикали граф идут через
# всё тело — доля от 0.44 (1971/06 с.87) до 1.0, у трёх таблиц с многоярусной шапкой, которые
# по плотности пересечений выглядели схемой (1973/08 с.18 и с.23, 1973/10 с.50), — 0.76, 1.0 и
# 0.84. У блок-схемы вертикали — стенки коробок, короткие: у 15 схем из 19 доля не выше 0.37.
# Порог в промежутке.
TALL_SPAN = 0.35
DIAGRAM_MAX_TALL_SHARE = 0.45

# Сквозных линеек (любой оси, от ``ruling.long_rules_of``) у схемы мало: коробка редко тянется
# на 0.8 ширины всей схемы. У бланка «Оперативная справка» (1974/04 с.75), который по плотности
# пересечений и коротким вертикалям выглядел схемой, сквозных девять.
DIAGRAM_MAX_LONG_RULES = 3


@dataclass(frozen=True)
class KindFeatures:
    n_horizontal: int
    n_vertical: int
    cross_density: float  # пересечения / (горизонтали × вертикали)
    col_align: float  # 1 − различных x вертикалей / вертикалей
    row_align: float  # то же для горизонталей
    long_share: float  # доля горизонталей во всю ширину
    tall_share: float  # доля вертикалей хотя бы на треть высоты рамки

    def as_row(self) -> dict[str, float]:
        return {
            "kind_h": float(self.n_horizontal),
            "kind_v": float(self.n_vertical),
            "cross_density": round(self.cross_density, 3),
            "col_align": round(self.col_align, 3),
            "row_align": round(self.row_align, 3),
            "long_share": round(self.long_share, 3),
            "tall_share": round(self.tall_share, 3),
        }


KIND_HEADER = ("kind_h", "kind_v", "cross_density", "col_align", "row_align", "long_share", "tall_share")


def _distinct(values: list[float], tolerance: float) -> int:
    count = 0
    last: float | None = None
    for value in sorted(values):
        if last is None or value - last > tolerance:
            count += 1
            last = value
    return count


def features_of(segments: list[Segment], box: Box, dpi: int) -> KindFeatures:
    horizontal = [s for s in segments if s.horizontal]
    vertical = [s for s in segments if not s.horizontal]
    tolerance = mm_to_px(CROSS_TOL_MM, dpi)
    crossings = sum(1 for h in horizontal for v in vertical if crosses(v, h, tolerance))
    pairs = len(horizontal) * len(vertical)
    align_tol = mm_to_px(ALIGN_TOL_MM, dpi)
    xs = [(s.box.x0 + s.box.x1) / 2 for s in vertical]
    ys = [(s.box.y0 + s.box.y1) / 2 for s in horizontal]
    col_align = 1.0 - _distinct(xs, align_tol) / len(xs) if xs else 0.0
    row_align = 1.0 - _distinct(ys, align_tol) / len(ys) if ys else 0.0
    long_share = sum(1 for s in horizontal if s.length >= LONG_SPAN * max(1, box.width)) / max(1, len(horizontal))
    tall_share = sum(1 for s in vertical if s.length >= TALL_SPAN * max(1, box.height)) / max(1, len(vertical))
    return KindFeatures(
        n_horizontal=len(horizontal),
        n_vertical=len(vertical),
        cross_density=crossings / pairs if pairs else 0.0,
        col_align=col_align,
        row_align=row_align,
        long_share=long_share,
        tall_share=tall_share,
    )


def is_diagram(kind: KindFeatures, box: Box, dpi: int, long_rules: float = 0.0, strict: bool = True) -> bool:
    """Похожи ли линейки на коробки блок-схемы, а не на решётку таблицы.

    ``strict`` добавляет признаки, которые отделяют схему от таблицы, ПРОХОДЯЩЕЙ проверку:
    короткие вертикали и мало сквозных линеек. Для скопления, которое проверку не прошло
    (мало сквозных, рёбра не на месте), они не нужны: там выбор не «таблица или схема», а
    «схема или мусор», и высокие боковые коробки схемы (1974/12 с.42) отбрасывать не за что.
    """
    if min(box.width, box.height) < mm_to_px(DIAGRAM_MIN_SIDE_MM, dpi):
        return False
    loose = (
        kind.n_horizontal + kind.n_vertical >= DIAGRAM_MIN_RULES
        and kind.n_vertical >= DIAGRAM_MIN_VERTICALS
        and kind.cross_density <= DIAGRAM_MAX_CROSS_DENSITY
        and kind.long_share <= DIAGRAM_MAX_LONG_SHARE
    )
    if not strict:
        return loose
    return loose and kind.tall_share <= DIAGRAM_MAX_TALL_SHARE and long_rules <= DIAGRAM_MAX_LONG_RULES


def classify(
    kind: KindFeatures, signs: "Features | None", box: Box, dpi: int, long_rules: float = 0.0
) -> tuple[str, str]:
    """Вид объекта и причина словами."""
    if signs is not None and signs.glyph_of_ink < FLOOR_GLYPH_OF_INK and signs.filled_cells < FLOOR_FILLED_CELLS:
        return KIND_DRAWING, (
            f"букв в краске {signs.glyph_of_ink:.0%}, заполнено ячеек {signs.filled_cells:.0%} — сетка графика или чертёж"
        )
    if is_diagram(kind, box, dpi, long_rules):
        return KIND_DIAGRAM, (
            f"линеек {kind.n_horizontal + kind.n_vertical}, пересечений {kind.cross_density:.2f} на пару, "
            f"сквозных {kind.long_share:.2f} — коробки, а не решётка"
        )
    return KIND_TABLE, ""
