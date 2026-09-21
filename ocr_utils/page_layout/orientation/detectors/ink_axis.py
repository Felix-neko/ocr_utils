"""Свой детектор: ось текста по сгущению глифов, знак — по выносным элементам.

ОСЬ. Компоненты связности размера глифа смыкаются по горизонтали и по вертикали (RLSA),
и сравнивается, вдоль какой оси получилось больше вытянутых «строк». Ключ здесь — отбор
компонент ПО РАЗМЕРУ до смыкания: полосы с боковой иллюстрацией сплошь состоят из длинных
штрихов чертежа, и без отбора смыкание цеплялось бы за них, а не за подпись. На разведке
по четырём известным боковым и четырём обычным полосам мера дала 1.000 в нужную сторону
на всех восьми.

ЗНАК. Ось не отличает 0 от 180 и 90 от 270 — для этого берётся модель выносных элементов
(Leptonica ``pixOrientDetect``, DFKI Joost et al.): у строки текста краска распределена
поперёк неравномерно, потому что верхние и нижние выносные встречаются с разной частотой.
Сама Leptonica сюда не годится (нужен 1 bpp, обучена на латинице, питоновских биндингов
нет), поэтому признак считается свой и калибруется по кириллице — см. UPRIGHT_ASYMMETRY.

Знак — слабое место: он меряет тонкую асимметрию там, где ось меряет грубую геометрию.
Поэтому при слабой асимметрии детектор честно поднимает ``axis_only`` и отдаёт решение
о стороне другим, а не выдаёт монетку за ответ.
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.orientation.detectors.base import Detector, Frame, Verdict, unknown

# Все размеры ниже — в пикселях копии 150 dpi (``Frame.gray150``), и рядом дан физический
# размер: детектор должен одинаково работать и на 600 dpi скане журнала, и на 300 dpi скане
# чего-то ещё, а привязка к долям кадра сломалась бы на первой же полосе другого формата.

# Размер глифа. 5 px при 150 dpi это 0.85 мм — примерно точка над «й» и запятая; 42 px это
# 7 мм, заметно больше кегля заголовка, но меньше высоты рамки таблицы.
GLYPH_MIN_PX = 5
GLYPH_MAX_PX = 42

# Компонента шире трёх максимальных высот глифа — это уже штрих чертежа или линейка, а не
# буква и не слипшееся слово.
GLYPH_MAX_WIDTH_PX = GLYPH_MAX_PX * 3

# Заполнение габаритного прямоугольника краской. Буква заполняет его хотя бы на 12%,
# длинная тонкая диагональ чертежа — заметно меньше.
GLYPH_MIN_FILL = 0.12

# Зазор смыкания RLSA: 8 px при 150 dpi это 1.35 мм — межсловный пробел кегля 10 пунктов.
# Больше — и смыкаются соседние колонки, меньше — и слова не собираются в строку.
RLSA_GAP_PX = 8

# Что считается «строкой» после смыкания: вытянута хотя бы вшестеро, толщиной не больше
# полутора кеглей заголовка и длиной хотя бы 60 px (1 см) — короче не бывает даже подпись.
LINE_ASPECT = 6.0
LINE_MAX_THICKNESS_PX = 45
LINE_MIN_LENGTH_PX = 60

# Меньше этой доли площади полосы, занятой «строками», — мерить нечего (фотография во всю
# полосу, пустая страница, чертёж вовсе без подписи). Замер: у обычной текстовой полосы
# пака-1 доля 0.17-0.19, у полосы с одной подписью вдоль края — сотые доли, поэтому порог
# стоит низко: он отсекает пустоту, а не малый текст.
MIN_LINE_INK_FRAC = 0.002

# Ниже этого перевеса одной оси над другой полоса считается неразличимой.
AXIS_MARGIN_THR = 0.25

# --- Знак поворота ---------------------------------------------------------
# Асимметрия краски поперёк строки: центр масс минус геометрический центр, в долях высоты
# строки, ось y вниз. Знак ЗАМЕРЕН на 400 случайных полосах пака-1 (395 из них с текстом,
# медиана 137 строк на полосу): у прямого текста медиана −0.0451, отрицательна на 99.7%
# полос; те же полосы, перевёрнутые на 180, дают +0.0447 и положительны на 99.7%.
#
# Знак именно такой, потому что в кириллице почти нет верхних выносных (одна «б»), зато
# нижних много — р, у, ф, д, ц, щ. Габарит строки тянется вниз, а масса краски остаётся
# в строчной высоте, то есть ВЫШЕ геометрического центра. Для латиницы, на которой обучена
# Leptonica, соотношение обратное — поэтому константа своя, а не взятая оттуда.
UPRIGHT_ASYMMETRY = -1.0

# Асимметрия слабее этой — считаем, что знак не определён. При 0.010 в замере выше решали
# 96.7% полос, и ни одна из них не решала неверно; при 0.030 решали бы 76.2% — запас, за
# который нечем платить.
ASYMMETRY_THR = 0.010

# Меньше этого числа строк — асимметрия считается по шуму, знак не определён.
MIN_LINES_FOR_SIGN = 4

# Число строк, начиная с которого асимметрии верим в полную силу. Обычная текстовая полоса
# пака-1 даёт 85-167 строк и асимметрию 0.036-0.075; полоса с боковой иллюстрацией — 10-30
# строк и 0.004-0.033, причём знак у неё врал на одной известной полосе из четырёх. То есть
# величина асимметрии сама по себе не говорит, можно ли ей верить, а число строк говорит,
# поэтому уверенность гасится именно по нему.
SIGN_FULL_LINES = 60


def glyph_mask(gray: np.ndarray) -> np.ndarray:
    """Маска компонент размера глифа: краска, отсеянная от штрихов чертежа и рамок."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if count <= 1:
        return np.zeros_like(ink)
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (
        (height >= GLYPH_MIN_PX)
        & (height <= GLYPH_MAX_PX)
        & (width >= 2)
        & (width <= GLYPH_MAX_WIDTH_PX)
        & (area >= GLYPH_MIN_FILL * width * height)
    )
    keep[0] = False  # нулевая метка — фон
    return np.where(keep[labels], np.uint8(255), np.uint8(0))


def _smear(mask: np.ndarray, horizontal: bool) -> np.ndarray:
    """Смыкание краски вдоль оси — слова собираются в строки."""
    kernel = np.ones((1, RLSA_GAP_PX), np.uint8) if horizontal else np.ones((RLSA_GAP_PX, 1), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def line_boxes(mask: np.ndarray, horizontal: bool) -> tuple[np.ndarray, float]:
    """Габариты «строк» вдоль оси и суммарная площадь краски в них."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(_smear(mask, horizontal), 8)
    if count <= 1:
        return np.empty((0, 4), int), 0.0
    stats = stats[1:]
    width = stats[:, cv2.CC_STAT_WIDTH].astype(float)
    height = stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    long_side, short_side = (width, height) if horizontal else (height, width)
    lineish = (
        (long_side >= LINE_ASPECT * np.maximum(short_side, 1.0))
        & (short_side <= LINE_MAX_THICKNESS_PX)
        & (long_side >= LINE_MIN_LENGTH_PX)
    )
    boxes = stats[lineish][:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
    return boxes, float(stats[lineish][:, cv2.CC_STAT_AREA].sum())


def ink_asymmetry(mask: np.ndarray) -> tuple[float, int]:
    """Средняя асимметрия краски поперёк ГОРИЗОНТАЛЬНЫХ строк и число учтённых строк.

    Положительное значение — центр масс ниже середины строки. Считается только по строкам:
    на пустом поле и на чертеже «поперёк строки» ничего не значит.
    """
    boxes, _ = line_boxes(mask, horizontal=True)
    values: list[float] = []
    for left, top, width, height in boxes:
        if height < GLYPH_MIN_PX * 2:
            continue  # строка ниже двух кеглей — это обрывок, выносные в ней не разглядеть
        profile = mask[top : top + height, left : left + width].sum(axis=1).astype(float)
        weight = profile.sum()
        if weight <= 0:
            continue
        centre = float((np.arange(height) * profile).sum() / weight)
        values.append((centre - (height - 1) / 2.0) / height)
    if len(values) < MIN_LINES_FOR_SIGN:
        return 0.0, len(values)
    return float(np.median(values)), len(values)


def axis_confidence_of(axis_score: float) -> float:
    return min(1.0, abs(axis_score))


def detect(frame: Frame) -> Verdict:
    mask = glyph_mask(frame.gray150)
    _, horizontal_ink = line_boxes(mask, horizontal=True)
    _, vertical_ink = line_boxes(mask, horizontal=False)

    total = horizontal_ink + vertical_ink
    if total < MIN_LINE_INK_FRAC * mask.size:
        return unknown("мало текста")

    axis_score = (horizontal_ink - vertical_ink) / total
    metrics = {"axis_score": axis_score, "line_ink": total / mask.size}
    if abs(axis_score) < AXIS_MARGIN_THR:
        return Verdict(0, 0.0, metrics=metrics, note="ось не различается")

    # Ось известна, осталась сторона. Маска поворачивается к первому кандидату, и асимметрия
    # всегда меряется у горизонтальных строк — так калибровочная константа нужна ровно одна.
    # Сторону выбираем только из допустимых: если 180 не разрешён, «книжная» ось означает
    # «поворот не нужен», и гадать по асимметрии не о чем.
    candidates = tuple(r for r in ((0, 180) if axis_score > 0 else (90, 270)) if r in frame.allowed)
    if not candidates:
        return Verdict(0, 0.0, metrics=metrics, note="ось вне набора допустимых углов")
    if len(candidates) == 1:
        return Verdict(candidates[0], axis_confidence_of(axis_score), metrics=metrics)
    turned = mask if candidates[0] == 0 else np.ascontiguousarray(np.rot90(mask, k=-(candidates[0] // 90)))
    asymmetry, lines = ink_asymmetry(turned)
    metrics["asymmetry"] = asymmetry
    metrics["lines"] = float(lines)

    axis_confidence = axis_confidence_of(axis_score)
    if abs(asymmetry) < ASYMMETRY_THR:
        # Ось видна, сторона — нет. Для книжной оси это значит «поворот не нужен» (0 и 180
        # одинаково правдоподобны, но 180 на полосе журнала — редкость), для боковой —
        # «повёрнута, сторону спросите у других».
        rotation = 0 if candidates[0] == 0 else 90
        return Verdict(rotation, axis_confidence, axis_only=True, metrics=metrics, note="сторона не различается")

    matches = (asymmetry > 0) == (UPRIGHT_ASYMMETRY > 0)
    rotation = candidates[0] if matches else candidates[1]
    sign_confidence = min(1.0, abs(asymmetry) / (ASYMMETRY_THR * 4)) * min(1.0, lines / SIGN_FULL_LINES)
    return Verdict(rotation, min(axis_confidence, sign_confidence), metrics=metrics)


ALGORITHM = Detector(
    name="ink_axis",
    summary="своя мера: ось по смыканию глифов (RLSA), сторона по выносным элементам",
    stage="cpu",
    run=detect,
)
