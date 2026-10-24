"""Границы растрового фото (перенос из стенда research/photo_bounds): маска «тон или текстура растра» в окне вокруг затравки, отсев текста, осевая рамка по маске, точки края поперёк каждой стороны, робастная подгонка сторон кривыми и мера чёткости края."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.units import mm_to_px_float

# --- Маска фото --------------------------------------------------------------------------------------------------------
# Размытие, гасящее растровую сетку: период сетки пака-1 ~0.4–0.5 мм (на 300 dpi — 5–6 px), σ 0.35 мм (4 px) сводит
# растр к непрерывному тону и почти не сдвигает край фото (край — ступенька, размытие симметрично).
TONE_SIGMA_MM = 0.35
# Окно локального разброса яркости (текстура растра в светлых местах снимка, где тон почти бумажный), мм.
TEXTURE_WIN_MM = 0.8
# Ядро снимка — строгая маска: тон ниже бумаги на ``CORE_TONE_DELTA`` или разброс выше ``CORE_TEXTURE_THR``. Так строки
# текста, подпись и просвет оборота почти не склеиваются со снимком (прогон v2 со слабыми порогами склеивал строки
# над снимком и подпись под ним — 1966/02 IMG_0095_2R, лишнее до 12 мм у 25 % сторон).
CORE_TONE_DELTA = 30.0
CORE_TEXTURE_THR = 12.0
# Светлый растр — слабые пороги, только для роста ядра: бумага пака-1 после ScanTailor почти белая и ровная (тон 251–253,
# разброс в окне 0.8 мм — 0 у медианы); светлые места снимков — тон 210–235 при разбросе 3–10 (замер 1974/11
# IMG_0053_1L, 1968/01 IMG_0015_2R, 1975/01 IMG_0046_2R).
LIGHT_TONE_DELTA = 12.0
LIGHT_TEXTURE_THR = 3.0
# Рост ядра: сторона сдвигается наружу шагом ``GROW_STEP_MM``, пока полоска шага вдоль всей стороны покрыта светлым
# растром не меньше чем на ``GROW_COVER``. Строка текста покрывает полоску на 20–45 % (между буквами бумага), светлый
# снимок — на 85–100 %.
GROW_STEP_MM = 0.25
# Местный порог светлого растра — разброс бумаги вниз (медиана − p10 светлой половины кольца) × этот множитель + 4.
PAPER_SPREAD_FACTOR = 2.0
# Светлая однородная плашка (небо, светлая стена у края снимка): тон ниже бумаги хотя бы на PLATE_TONE_DELTA (местный
# порог — разброс бумаги × 1.5 + 2, не меньше) при малом разбросе (≤ PLATE_MAX_TEXTURE — у текста он 10+). Рост рамки
# идёт и по ней, если полоска шага покрыта плашкой или светлым растром на ≥ PLATE_COVER. Замер МТС 1990/09 IMG_0310
# (холст выпрямления, бумага 254): небо 233–247, край плашки — ступенька 7–20 уровней по всей стороне.
PLATE_TONE_DELTA = 5.0
PLATE_MAX_TEXTURE = 6.0
PLATE_COVER = 0.85
# Плашка светлее этого (тон выше бумаги минус столько) — темнее уже снимок или краска.
PLATE_MAX_DELTA = 40.0

# --- Объект за рамкой (вырезки без фона, растушёвка, тонкие стрелы) ----------------------------------------------------
# Краска и тон объекта — всё темнее местной бумаги на плашечный порог (тот же, что у плашки): светлая растушёвка тоже
# объект (МТС 1990/10 IMG_0558 — облако растушёвки над рисованным краном). Сцепка — замыкание OBJECT_CLOSE_MM; к объекту
# относятся компоненты, задевающие ядро снимка, сжатое на OBJECT_CORE_INSET_MM (край ядра касается подписи и соседей).
OBJECT_CLOSE_MM = 0.4
# Объект — темнее бумаги хотя бы на столько уровней: просвет оборота и пятна бумаги пака-1 — 10–25 уровней (1969/10
# IMG_0020_2R: серое пятно у левого края снимка), растушёвка рисованного крана МТС 1990/10 IMG_0558 — 20–120.
OBJECT_TONE_DELTA = 25.0
# Слабое размытие для маски объекта, мм: гасит шум пикселя светлой растушёвки, но не сцепляет снимок с подписью.
OBJECT_SIGMA_MM = 0.12
OBJECT_CORE_INSET_MM = 1.0
# Объекта за рамкой больше этого, мм², — рамка расширяется до охвата объекта, а четырёхугольника нет (это вырезка, а не
# снимок с краем). Одиночные пятна и задетые буквы — до 1–2 мм².
OBJECT_OUT_MM2 = 3.0
# Текст рядом со снимком в объект не идёт: компонента краски вне рамки снимка размером с букву (высота TEXT_MIN_MM–
# TEXT_MAX_MM, ширина до TEXT_MAX_W_MM), у которой в ряду (центры по высоте ближе 0.5 высоты, по горизонтали ближе
# TEXT_ROW_GAP_MM) не меньше двух соседей того же кегля, — буква строки; её вырезают до сцепки объекта. Одиночный обломок
# стрелы или люлька размером с букву соседей в ряду не имеет и остаётся. Тень у края листа (ближе OBJECT_PAGE_EDGE_MM к
# краю полосы) — тоже не объект: по ней объект сцеплялся с подписью (1975/10 IMG_0014_1L — полоса тени в 4–7 мм от края
# кадра; рамка снимка от ядра этим не режется — режется только рост по объекту).
TEXT_MIN_MM = 0.8
TEXT_MAX_MM = 6.0
TEXT_MAX_W_MM = 10.0
TEXT_ROW_GAP_MM = 4.0
OBJECT_PAGE_EDGE_MM = 8.0
# Буквы для запрета расширения рамки — по сильной краске (темнее бумаги на TEXT_INK_DELTA): сор и грязь вокруг них
# букв не образуют. Расширение стороны отменяется, если в его полосе букв не меньше TEXT_GUARD_LETTERS (строка подписи —
# 10–40 букв).
TEXT_INK_DELTA = 90.0
# Ореол букв, вычитаемый из объекта вместе с буквами, мм (межбуквенные и межстрочные просветы подписи, залитые сором).
TEXT_HALO_MM = 1.5
TEXT_GUARD_LETTERS = 6
# Плашка с текстом у края компоненты снимка (заголовок на тонированной плашке, 1975/03 IMG_0103_1L) — не снимок:
# полоса строк у края компоненты, где доля растровой текстуры меньше PLATE_TEXTURE_SHARE, до первой полосы снимка
# (EDGE_BODY_MM подряд с долей не меньше PLATE_TEXTURE_SHARE), отрезается, если она тонирована (средний тон темнее бумаги
# на плашечный порог: белая бумага с подписью внутри коллажа — не плашка) и в ней не меньше TEXT_GUARD_LETTERS букв.
# Замер 1975/03 IMG_0103_1L: строки плашки 0.21–0.58 (с буквами до 0.87 в отдельных строках), снимок 0.89–0.97.
PLATE_TEXTURE_SHARE = 0.8
EDGE_BODY_MM = 8.0
PLATE_PHOTO_SHARE = 0.2
# Слабая краска для букв, останавливающих рост рамки (просвет оборота), уровней ниже бумаги.
GROW_LETTER_DELTA = 20.0
# Полоска роста с таким числом букв (строка текста пересекает десятки) — стоп; одно-два пятна в небе снимка не мешают.
GROW_STOP_LETTERS = 3
# Частичный рост стороны (:func:`grow_partial`): растр за стороной участками от PART_RUN_MM и всего от PART_MIN_MM
# вдоль неё, сцепленный с растром предыдущей полоски (сдвиг до PART_LINK_MM — край кепки идёт наискось).
PART_RUN_MM = 1.0
PART_MIN_MM = 3.0
PART_LINK_MM = 0.5
# Растровые точки бледного растра (:func:`halftone_map`): темнее светлого фона в окне DOT_WIN_MM (период сетки пака-1
# 0.4–0.5 мм) на DOT_CONTRAST уровней; растр — доля точек от DOT_SHARE в окне DOT_DENSITY_MM. Замер полос роста
# стенда v14: снимок 5–6 % точек на полосу (внутри части снимка больше), тень и бумага 0–1 %.
DOT_WIN_MM = 0.6
DOT_CONTRAST = 12.0
DOT_DENSITY_MM = 1.0
DOT_SHARE = 0.1
# Добивка стороны по точкам (:func:`dot_tip`): доля слабых точек (TIP_DOT_CONTRAST) в столбцах растущей части полоски
# шага (1975/11 IMG_0054_1L).
TIP_DOT_SHARE = 0.01
# Контраст точек добивки: у макушки кепки при 8 — 4.7 % точек в полоске у края, в небе рядом 0–0.7 %; при 12 — 0.6 %.
TIP_DOT_CONTRAST = 8.0
# Добивка — не дальше этого от места остановки, мм: у макушки кепки рост по плотности встаёт на 1.7 мм ниже края точек.
TIP_REACH_MM = 2.0
# Округа бледной буквы, где ищется бумага, мм.
LETTER_PAPER_MM = 1.0
# Объект — краска или растр: гладкий светлый тон (тень у края листа, сор бумаги, плашка) объектом не считается, если он
# светлее бумаги минус OBJECT_SMOOTH_DELTA и без растровой текстуры. Без этого тень у края листа шла в объект, и рамка
# у левых полос уходила на 2–5 мм влево за снимок до границы запретной полосы (1972/07 IMG_0020_1L, IMG_0011_1L,
# 1974/03 IMG_0140_1L, 1975/02 IMG_0060_1L, 1971/01 IMG_0023_1L — рамка обрывалась ровно на 8 мм от края кадра).
OBJECT_SMOOTH_DELTA = 60.0
# Проходов расширения рамки до охвата объекта.
OBJECT_PASSES = 4
# Часть объекта за рамкой меньше этого, мм², — крошка сора или просвета, а не объект (стрела крана 0.6 × 20 мм — 12 мм²).
OBJECT_PART_MM2 = 4.0
OBJECT_PART_INK = 0.25
# Части за рамкой, отходящие от неё не дальше этого, мм, — кайма вдоль стороны, а не объект.
OBJECT_FRINGE_MM = 1.5
# Поле за рамкой, в котором объект ещё не считается «за рамкой», мм (кромка растра у самого края снимка).
OBJECT_MARGIN_MM = 0.5
# Тёмная краска объекта, сплошь выходящая за сторону рамки (:func:`ink_overhang`): компонента краски задевает полосу
# INK_CROSS_MM внутри рамки, а за рамкой с полем у неё не меньше INK_OVERHANG_MM2 — рамка накрывает и её.
INK_CROSS_MM = 1.0
INK_OVERHANG_MM2 = 1.0

# --- Правило четырёх сторон ---------------------------------------------------------------------------------------------
# Четырёхугольник (и стороны в опорах выпрямления) — только если все четыре стороны чёткие и сходятся в углах:
# точки края каждой стороны доходят до угла ближе CORNER_GAP_MM сверх отступа CORNER_SKIP, угол в углах — 90 ± CORNER_
# ANGLE_DEG, многоугольник выпуклый, его охват — в QUAD_BOX_MM от рамки по маске. Замер по 332 фото пака-1 со всеми
# чёткими сторонами (прогон стенда v9): зазор сверх отступа — p95 1.6 мм, дальше хвост до 48 мм (34 фото больше 5 мм:
# угол достраивается кривой через светлый угол, правая сторона 1974/05 IMG_0054_1L уезжает внутрь на 6 мм); отклонение
# угла — p99 2.6°, максимум 5.3°.
CORNER_GAP_MM = 5.0
# ...или не дальше этой доли длины стороны (светлый угол снимка: у ХиП 1991/04 IMG_0154 правая сторона 55 мм не
# доходит до верхнего угла на 7.7 мм — белые колонны у края; сторона при этом ровная и чёткая на 86 % длины).
CORNER_GAP_SHARE = 0.15
CORNER_ANGLE_DEG = 4.0
QUAD_BOX_MM = 5.0
GROW_COVER = 0.7
# Замыкание маски — склеивает растровые точки светлых участков в сплошную область, мм.
CLOSE_MM = 0.5
# Размыкание маски: квадрат больше высоты строчных букв набора (кегль 8–10 пт, строчные 1.5–2 мм), чтобы строки
# текста, склеенные замыканием, выпали, а снимок (от 10 мм) остался.
OPEN_MM = 2.5
# Компонента маски — текст, а не фото, если в ней доля «бумажных» пикселей тона (не темнее бумаги на
# ``PAPER_TOL``) больше этой: между строками и буквами текста бумага, в снимке её почти нет.
TEXT_PAPER_SHARE = 0.35
PAPER_TOL = 15.0
# Нечёткая сторона глубже этого внутри затравки — назад на сторону затравки, мм. Затравка детектора шире снимка в
# медиане на 1.2 мм (поле), поэтому порог выше: иначе уточнение по маске отменялось бы у всех нечётких сторон.
SOFT_INSET_MM = 2.5
# Охват компоненты: строка (столбец) входит, если занята компонентой хотя бы на эту долю медианной заполненности.
# Тонкий отросток (тень у края листа шириной 1–3 мм против снимка 50–150 мм) даёт строки с долей 0.02–0.06.
ROBUST_EXTENT_SHARE = 0.5
# Разрыв полосы строк (столбцов) снимка, который ещё не делит её, мм: светлое место снимка с малой заполненностью.
EXTENT_GAP_MM = 1.5
# Компонента — набор, если доля её рядов со средним тоном не темнее бумаги на ``EMPTY_ROW_TOL`` больше этой (межстрочные
# просветы плотной подписи; ХиП 1991/06 IMG_0377 — подпись из 4 строк, детектор растра принял её за фото).
TEXT_EMPTY_ROWS = 0.12
EMPTY_ROW_TOL = 25.0
# Межстрочный просвет — не длиннее этого, мм (интерлиньяж набора 1–2 мм); длиннее — светлый участок снимка.
INTERLINE_MAX_MM = 2.5
# Компоненты одной группы (одного фото): рамки ближе этого, мм. Между двумя снимками на полосе — от 2–3 мм.
GROUP_GAP_MM = 1.5
# Самая маленькая сторона фото, мм.
MIN_PHOTO_SIDE_MM = 8.0
# Поле окна поиска вокруг затравки, мм. Затравка детектора срезает край снимка до 1 см (``raster.regions``), а у фото
# со светлым небом — половину снимка (МТС 1990/09 IMG_0310: затравка — тёмный низ, небо с люльками подъёмников на 55 мм
# выше), и у вырезок без фона объект торчит за затравку (стрела крана, МТС 1991/01 IMG_0615) — окно с запасом.
SEARCH_MM = 60.0

# --- Точки края --------------------------------------------------------------------------------------------------------
# Шаг точек вдоль стороны, мм, и отступ от углов (доля длины стороны): у угла профиль задевает соседнюю сторону.
EDGE_STEP_MM = 1.0
CORNER_SKIP = 0.04
# Профиль поперёк стороны: внутрь и наружу от стороны рамки по маске, мм. Внутрь больше — рамка по маске при
# перекосе снимка на 1° отходит от края на 2–3 мм у дальнего конца стороны.
PROFILE_IN_MM = 5.0
PROFILE_OUT_MM = 3.0
# Профиль усредняется вдоль стороны по полосе этой ширины, мм (гасит растр без лишнего размытия поперёк).
PROFILE_BAND_MM = 1.0
# Окна тона по обе стороны края, мм: снаружи ``[0.7, 1.5]``, внутри ``[-1.5, -0.7]``. Ближе 0.7 мм — хвост ступеньки,
# размытой σ ``TONE_SIGMA_MM`` (половина перехода ~2σ): с окном от 0.3 мм проверка «снаружи бумага» шла на грани
# порога, и у резких краёв (1970/01 IMG_0022_1L, низ) находилась треть точек.
SIDE_NEAR_MM = 0.7
SIDE_FAR_MM = 1.5
# Ступенька на краю — разность тона снаружи и внутри, уровней серого; меньше — точки края нет. Светлый край снимка
# на белой бумаге — 15–40 уровней.
EDGE_MIN_CONTRAST = 12.0
# Снаружи края — бумага: тон не ниже бумаги на столько уровней.
OUTSIDE_PAPER_TOL = 12.0
# Затенённая бумага снаружи края: гладкий тон (разброс не выше местного порога светлого растра) не ниже бумаги окна на
# SHADE_PAPER_TOL, при ступеньке от SHADE_MIN_CONTRAST. У 1972/02 IMG_0054_1L слева внизу бумага у края листа 239–241
# и серый клин тени 216 при бумаге окна 255: у настоящей кромки снимка (ступенька 140) «снаружи бумага» не выполнялось,
# и точка вставала на внешний край клина — 43 % точек левой стороны в выбросах, сторона нечёткая.
SHADE_PAPER_TOL = 45.0
SHADE_MIN_CONTRAST = 40.0
# Внутри края — растр снимка: в окне ``[-SIDE_FAR_MM, -SIDE_NEAR_MM]`` разброс всюду выше местного порога светлого
# растра или ступенька не меньше OBJECT_SMOOTH_DELTA (заметная краска: сплошь тёмный снимок без разброса). Гладкий
# серый клин тени (разброс 1–2, на 23 уровня темнее бумаги снаружи; у клина шириной 1.5 мм окно «внутри» задевает
# снимок, и ступенька по средним доходит до 40) — не снимок, и его внешний край не край фото.


# --- Подгонка сторон ---------------------------------------------------------------------------------------------------
# Старт подгонки — многочлен 2-й степени (дуга прогиба листа), робастно; по его весам — кубический сплайн с узлом на
# каждые SIDE_KNOT_MM длины стороны (не больше SIDE_MAX_KNOTS внутренних узлов; короче 2 × SIDE_KNOT_MM — многочлен).
# Замер по 1328 чётким сторонам пака-1 (точки края прогона стенда v9, проверка «на чётных — по нечётным»): остаток p90
# на длинных (от 100 мм) сторонах при многочлене — медиана 0.086 мм (p90 0.170), узел на 50 мм — 0.070, 35 — 0.065,
# 25 — 0.061 (p90 0.105), 18 — 0.057; расхождение подгонок по чётным и нечётным точкам (мера «ловит шум») — 0.013,
# 0.016, 0.019, 0.023 (p90 0.057), 0.027 мм. Узел на 25 мм — колено: дальше остаток почти не падает, а расхождение
# растёт. Фото-развороты (МТС 1990/03 IMG_1101: размах края от прямой после выпрямления 0.45–0.54 мм) гнутся сложнее дуги.
SIDE_DEGREE = 2
SIDE_KNOT_MM = 25.0
SIDE_MAX_KNOTS = 8
# Хьюбер и отсев выбросов подгонки, мм.
FIT_HUBER_MM = 0.15
FIT_OUTLIER_MM = 0.6
FIT_PASSES = 4

# --- Чёткость (пороги — по распределению на наборе, см. README) --------------------------------------------------------
# Сторона чёткая: точки края на доле длины не меньше этой, остаток p90 не больше ``SHARP_MAX_RESID_MM``, ширина
# перехода (медиана, ступенька / наибольшая производная) не больше ``SHARP_MAX_WIDTH_MM``. Ступенька, размытая
# σ ``TONE_SIGMA_MM``, имеет ширину σ·√(2π) ≈ 0.88 мм; виньетка — несколько мм. Контраст сам по себе не мера
# чёткости: светлый снимок с резким краем даёт ступеньку 15–30 уровней.
# Пороги — по разметке глазами (``sets/photo_classes.csv``, прогон v7): покрытие 0.7 и остаток 0.25 мм давали чёткими
# 282 из 385 фото с резким краем (73 %), 0.6 и 0.3 мм — 322 (84 %); среди нечётких глазами чёткими по-прежнему 2 из 6.
# При 0.7 низ снимка ХиП 1991/04 IMG_0154 (покрытие 0.69–0.75) становился чётким или нет от шума JPEG.
SHARP_MIN_COVER = 0.6
SHARP_MAX_RESID_MM = 0.3
SHARP_MAX_WIDTH_MM = 1.4


class Side(StrEnum):
    """Сторона фото."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"

    @property
    def horizontal(self) -> bool:
        """Сторона идёт вдоль x (верх, низ)."""
        return self in (Side.TOP, Side.BOTTOM)

    @property
    def outward(self) -> int:
        """Знак наружу по поперечной координате: верх и левая — к меньшим координатам."""
        return -1 if self in (Side.TOP, Side.LEFT) else 1


class EdgeClass(StrEnum):
    """Чёткость края фото по числу чётких сторон."""

    SHARP = "sharp"  # все четыре стороны чёткие: кривой четырёхугольник
    PARTIAL = "partial"  # 1–3 чёткие стороны
    SOFT = "soft"  # чётких сторон нет: виньетка, растворение в фон, фото без рамки — только рамка


@dataclass(frozen=True)
class Maps:
    """Карты окна поиска.

    Поля:
        origin: Левый верхний угол окна на полосе ``(x, y)``, px.
        tone: Тон (размытый серый), float32.
        texture: Локальный разброс яркости, float32.
        paper: Уровень бумаги окна.
        light_delta: На сколько ниже бумаги тон светлого растра (местный порог: не меньше :data:`LIGHT_TONE_DELTA`).
        light_texture: Разброс, с которого пиксель — светлый растр (местный: не меньше :data:`LIGHT_TEXTURE_THR`).
        plate_delta: На сколько ниже бумаги тон светлой плашки и объекта (местный: не меньше :data:`PLATE_TONE_DELTA`).
        fine: Тон со слабым размытием σ :data:`OBJECT_SIGMA_MM` — для маски объекта: сильное размытие ``tone`` сцепляет
            снимок с подписью в 1.5 мм.
        page_shape: ``(H, W)`` полосы — для отсева тени у края листа в маске объекта.
    """

    origin: tuple[int, int]
    tone: np.ndarray
    texture: np.ndarray
    paper: float
    light_delta: float = LIGHT_TONE_DELTA
    light_texture: float = LIGHT_TEXTURE_THR
    plate_delta: float = PLATE_TONE_DELTA
    fine: np.ndarray | None = None
    page_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class SideModel:
    """Кривая стороны ``поперёк = f(вдоль)``: многочлен (``coef``) или кубический сплайн (``knots``, ``coef``, ``k``);
    за диапазоном точек ``[lo, hi]`` — продолжение касательной у конца (сплайн за краем не экстраполируется).

    Поля:
        coef: Коэффициенты многочлена (``np.polyval``) или B-сплайна.
        knots: Узлы B-сплайна (полный вектор) или None — многочлен.
        k: Степень сплайна.
        lo, hi: Диапазон точек вдоль, px.
    """

    coef: np.ndarray
    knots: np.ndarray | None = None
    k: int = 3
    lo: float = -np.inf
    hi: float = np.inf

    def _raw(self, a: np.ndarray) -> np.ndarray:
        """Значение без продолжения за краями."""
        if self.knots is None:
            return np.polyval(self.coef, a)
        from scipy.interpolate import splev

        return splev(a, (self.knots, self.coef, self.k))

    def __call__(self, a) -> np.ndarray:
        """Поперечная координата в точках ``a`` (скаляр или массив), px."""
        a = np.asarray(a, np.float64)
        if self.knots is None or not np.isfinite(self.lo):
            return self._raw(a)
        out = self._raw(np.clip(a, self.lo, self.hi))
        eps = max((self.hi - self.lo) * 1e-3, 1e-3)
        lo0, lo1, hi0, hi1 = self._raw(np.array([self.lo, self.lo + eps, self.hi - eps, self.hi]))
        # За краями — касательная у конца: значение конца плюс наклон × расстояние.
        out = np.where(a < self.lo, lo0 + (lo1 - lo0) / eps * (a - self.lo), out)
        return np.where(a > self.hi, hi1 + (hi1 - hi0) / eps * (a - self.hi), out)

    def to_json(self) -> dict:
        """Модель → JSON."""
        return {"coef": np.asarray(self.coef).tolist(), "knots": None if self.knots is None else self.knots.tolist(),
                "k": self.k, "lo": float(self.lo), "hi": float(self.hi)}  # fmt: skip


@dataclass(frozen=True)
class SideFit:
    """Подгонка одной стороны фото.

    Поля:
        side: Сторона.
        along: Вдоль-координаты точек края (N,), px полосы.
        across: Поперечные координаты точек края (N,), px полосы (y у верха/низа, x у боковых).
        contrast: Ступенька тона у каждой точки (N,).
        width: Ширина перехода у каждой точки (N,), px.
        inlier: Точка вошла в подгонку (N,).
        coef: Кривая стороны ``across = f(along)`` (:class:`SideModel`) или None — точек мало.
        span: Диапазон вдоль стороны, на котором ставились точки ``(a0, a1)``, px.
        samples: Сколько позиций вдоль стороны проверено (знаменатель покрытия).
        mm_per_px: Перевод px → мм.
    """

    side: Side
    along: np.ndarray
    across: np.ndarray
    contrast: np.ndarray
    width: np.ndarray
    inlier: np.ndarray
    coef: SideModel | None
    span: tuple[float, float]
    samples: int
    mm_per_px: float

    @property
    def cover(self) -> float:
        """Доля позиций вдоль стороны с годной точкой края (в подгонке)."""
        return float(self.inlier.sum()) / max(self.samples, 1)

    @property
    def resid_p90_mm(self) -> float:
        """Остаток подгонки по годным точкам, p90 модуля, мм; ``inf`` — подгонки нет."""
        if self.coef is None or not self.inlier.any():
            return float("inf")
        r = self.across[self.inlier] - self.coef(self.along[self.inlier])
        return float(np.percentile(np.abs(r), 90)) * self.mm_per_px

    @property
    def resid_rms_mm(self) -> float:
        """Остаток подгонки RMS по годным точкам, мм; ``inf`` — подгонки нет."""
        if self.coef is None or not self.inlier.any():
            return float("inf")
        r = self.across[self.inlier] - self.coef(self.along[self.inlier])
        return float(np.sqrt(np.mean(r**2))) * self.mm_per_px

    @property
    def contrast_median(self) -> float:
        """Медиана ступеньки по годным точкам; 0 — точек нет."""
        return float(np.median(self.contrast[self.inlier])) if self.inlier.any() else 0.0

    @property
    def width_median_mm(self) -> float:
        """Медиана ширины перехода по годным точкам, мм; ``inf`` — точек нет."""
        return float(np.median(self.width[self.inlier])) * self.mm_per_px if self.inlier.any() else float("inf")

    @property
    def bow_mm(self) -> float:
        """Прогиб подогнанной стороны: наибольшее отклонение кривой от хорды на диапазоне точек, мм."""
        if self.coef is None:
            return 0.0
        a = np.linspace(*self.span, 50)
        c = self.coef(a)
        chord = c[0] + (c[-1] - c[0]) * (a - a[0]) / max(a[-1] - a[0], 1e-9)
        return float(np.abs(c - chord).max()) * self.mm_per_px

    def sharp(
        self,
        min_cover: float = SHARP_MIN_COVER,
        max_resid_mm: float = SHARP_MAX_RESID_MM,
        max_width_mm: float = SHARP_MAX_WIDTH_MM,
    ) -> bool:
        """Чёткая ли сторона: покрытие, остаток и ширина перехода в порогах.

        Args:
            min_cover: Наименьшая доля длины с точками края.
            max_resid_mm: Наибольший остаток p90, мм.
            max_width_mm: Наибольшая медиана ширины перехода, мм.

        Returns:
            True — сторона чёткая.
        """
        return (
            self.coef is not None
            and self.cover >= min_cover
            and self.resid_p90_mm <= max_resid_mm
            and self.width_median_mm <= max_width_mm
        )

    def curve(self, a0: float, a1: float, n: int = 64) -> np.ndarray:
        """Точки подогнанной стороны ``(x, y)`` на участке вдоль ``[a0, a1]``.

        Args:
            a0, a1: Концы участка вдоль стороны, px.
            n: Число точек.

        Returns:
            Точки (n, 2), px полосы.
        """
        a = np.linspace(a0, a1, n)
        c = self.coef(a)
        return np.stack([a, c], 1) if self.side.horizontal else np.stack([c, a], 1)


@dataclass(frozen=True)
class PhotoBound:
    """Найденное фото: осевая рамка, маска и подгонка сторон.

    Поля:
        box: Уточнённая рамка ``(x0, y0, x1, y1)``, px полосы (300 dpi стенда).
        mask_box: Рамка по маске (до уточнения краем), px.
        seed: Затравка, px.
        sides: Подгонки сторон: верх, низ, левая, правая.
        corners: Углы кривого четырёхугольника (лв, пв, пн, лн) или None — правило четырёх сторон не выполнено
            (:func:`quad_check`): тогда ни одна сторона в опоры выпрямления не идёт, а область фото — рамка.
        dpi: Разрешение полосы.
        sharp_sides: Чёткость сторон (верх, низ, левая, правая) — для мер; в опоры идут только при ``corners``.
        quad_reason: Почему четырёхугольника нет (пусто — есть).
        outside_mm2: Объект за рамкой по сторонам (``top``/``bottom``/``left``/``right``/``total``), мм² — после
            расширения рамки до охвата объекта (:func:`object_outside`).
    """

    box: tuple[float, float, float, float]
    mask_box: tuple[int, int, int, int]
    seed: tuple[int, int, int, int]
    sides: tuple[SideFit, ...]
    corners: np.ndarray | None
    dpi: float
    sharp_sides: tuple[bool, ...] = field(default=())
    quad_reason: str = ""
    outside_mm2: dict = field(default_factory=dict)

    @property
    def edge_class(self) -> EdgeClass:
        """Класс чёткости: ``SHARP`` — четырёхугольник по правилу четырёх сторон, иначе по числу чётких сторон."""
        n = sum(self.sharp_sides)
        return EdgeClass.SHARP if self.corners is not None else EdgeClass.PARTIAL if n else EdgeClass.SOFT

    def side(self, which: Side) -> SideFit:
        """Подгонка стороны."""
        return next(s for s in self.sides if s.side is which)

    def polygon(self, n: int = 48) -> np.ndarray:
        """Контур фото: кривой четырёхугольник по сторонам (там, где угол найден) или осевая рамка.

        Args:
            n: Точек на сторону.

        Returns:
            Многоугольник (M, 2), px полосы, по часовой: верх слева направо, правая сверху вниз, низ справа налево,
            левая снизу вверх.
        """
        if self.corners is None:
            x0, y0, x1, y1 = self.box
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
        tl, tr, br, bl = self.corners
        top = self.side(Side.TOP).curve(tl[0], tr[0], n)
        right = self.side(Side.RIGHT).curve(tr[1], br[1], n)
        bottom = self.side(Side.BOTTOM).curve(br[0], bl[0], n)
        left = self.side(Side.LEFT).curve(bl[1], tl[1], n)
        return np.concatenate([top, right, bottom, left])


def mm(value: float, dpi: float) -> float:
    """мм → px (float) при ``dpi``."""
    return mm_to_px_float(value, dpi)


def page_maps(
    gray: np.ndarray, window: tuple[int, int, int, int], dpi: float, inner: tuple[int, int, int, int] | None = None
) -> Maps:
    """Карты тона и текстуры в окне полосы и местные пороги бумаги.

    Бумага — по кольцу окна вне ``inner`` (затравки): светлая половина тонов кольца (тёмная — текст, снимок), уровень —
    её медиана, разброс вниз — медиана минус p10. Пятнистая бумага (1968/11 IMG_0078_1L: тон 239–251) иначе уходит
    в светлый растр при пороге «бумага − 12», и рамка растёт в поле.

    Args:
        gray: Серая полоса.
        window: Окно ``(x0, y0, x1, y1)``, px (обрезается по полосе).
        dpi: Разрешение полосы.
        inner: Рамка, которая в оценку бумаги не идёт (затравка), px полосы; None — всё окно.

    Returns:
        Карты окна с уровнем бумаги и местными порогами светлого растра.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = max(window[0], 0), max(window[1], 0), min(window[2], w), min(window[3], h)
    part = gray[y0:y1, x0:x1].astype(np.float32)
    tone = cv2.GaussianBlur(part, (0, 0), mm(TONE_SIGMA_MM, dpi))
    k = max(3, int(round(mm(TEXTURE_WIN_MM, dpi))) | 1)
    mean = cv2.blur(part, (k, k))
    sq = cv2.blur(part * part, (k, k))
    texture = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    ring = np.ones(tone.shape, bool)
    if inner is not None:
        ix0, iy0 = max(inner[0] - x0, 0), max(inner[1] - y0, 0)
        ring[iy0 : max(inner[3] - y0, 0), ix0 : max(inner[2] - x0, 0)] = False
    values = tone[ring] if ring.sum() > 100 else tone.ravel()
    bright_sel = values >= np.median(values)
    bright = values[bright_sel]
    paper = float(np.median(bright))
    spread = paper - float(np.percentile(bright, 10))
    tex = (texture[ring] if ring.sum() > 100 else texture.ravel())[bright_sel]
    light_delta = max(LIGHT_TONE_DELTA, PAPER_SPREAD_FACTOR * spread + 4.0)
    light_texture = max(LIGHT_TEXTURE_THR, float(np.percentile(tex, 90)) + 1.5)
    plate_delta = max(PLATE_TONE_DELTA, 1.5 * spread + 2.0)
    fine = cv2.GaussianBlur(part, (0, 0), mm(OBJECT_SIGMA_MM, dpi))
    return Maps((x0, y0), tone, texture, paper, light_delta, light_texture, plate_delta, fine, (h, w))


def photo_mask(maps: Maps, dpi: float) -> np.ndarray:
    """Маска «снимок» в окне: тонированный или текстурный пиксель → замыкание → заливка дыр → размыкание.

    Args:
        maps: Карты окна.
        dpi: Разрешение полосы.

    Returns:
        Маска uint8 0/1 окна.
    """
    raw = ((maps.tone < maps.paper - CORE_TONE_DELTA) | (maps.texture > CORE_TEXTURE_THR)).astype(np.uint8)
    # Тень и пятнистая бумага у края листа в ядро снимка не идут (1972/12 IMG_0154_1L: ядро росло по тени у края до
    # верха страницы); снимок у самого края кадра возвращается на затравку правилом запасной стороны.
    raw = page_edge_cut(raw, maps, dpi)
    kc = max(1, int(round(mm(CLOSE_MM, dpi))))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kc, kc)))
    filled = fill_holes(closed)
    ko = max(1, int(round(mm(OPEN_MM, dpi))))
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko)))


def page_edge_cut(mask: np.ndarray, maps: Maps, dpi: float) -> np.ndarray:
    """Обнуляет маску ближе :data:`OBJECT_PAGE_EDGE_MM` к краю полосы (тень и неровная бумага у края листа).

    Args:
        mask: Маска окна.
        maps: Карты окна (начало, размер полосы).
        dpi: Разрешение.

    Returns:
        Маска без полосы у края полосы.
    """
    if maps.page_shape is None:
        return mask
    out = mask.copy()
    ox, oy = maps.origin
    e = int(round(mm(OBJECT_PAGE_EDGE_MM, dpi)))
    ph, pw = maps.page_shape
    out[: max(e - oy, 0)] = 0
    out[:, : max(e - ox, 0)] = 0
    out[max(ph - e - oy, 0) :] = 0
    out[:, max(pw - e - ox, 0) :] = 0
    return out


def letters_in(maps: Maps, exclude, dpi: float, delta: float = TEXT_INK_DELTA, on_paper: bool = True) -> np.ndarray:
    """Маска букв строк (сильная краска, компоненты размером с букву с соседями в ряду) вне рамки ``exclude``.

    Args:
        maps: Карты окна.
        exclude: Рамка, буквы внутри которой не ищутся (снимок), px полосы.
        dpi: Разрешение.
        delta: Насколько краска темнее бумаги, уровней (меньше — ловит и просвет букв оборота).
        on_paper: Только буквы на бумаге (рядом есть тон бумаги): тёмные детали в ряд внутри светлого неба снимка
            (машины под небом МТС 1990/05 IMG_1273) — не буквы. False — и на плашке (заголовок на тонированной плашке).

    Returns:
        Маска bool окна.
    """
    tone = maps.fine if maps.fine is not None else maps.tone
    ink = (tone < maps.paper - delta).astype(np.uint8)
    labels, text = text_letters(ink, maps, exclude, dpi)
    letters = text[labels]
    if on_paper and letters.any():
        # Бледные «буквы» — только на бумаге: самый светлый тон в округе LETTER_PAPER_MM не темнее бумаги на плашечный
        # порог. Пятна растра в светлом небе снимка (МТС 1990/05 IMG_1273: небо 230–238 при бумаге 253) — не буквы.
        k = max(3, int(round(2 * mm(LETTER_PAPER_MM, dpi))) | 1)
        bright = cv2.dilate(maps.tone, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        n = len(text)
        on_paper = np.zeros(n, bool)
        lab = labels[letters]
        peak = np.zeros(n)
        np.maximum.at(peak, lab, bright[letters])
        on_paper[peak >= maps.paper - maps.plate_delta] = True
        letters &= on_paper[labels]
    return letters


def letter_count(letters: np.ndarray) -> int:
    """Число букв (компонент) в маске."""
    if not letters.any():
        return 0
    return cv2.connectedComponents(letters.astype(np.uint8), connectivity=8)[0] - 1


def trim_text_plates(maps: Maps, box: tuple[int, int, int, int], dpi: float) -> tuple[int, int, int, int]:
    """Отрезает от рамки компоненты снимка полосы-плашки с текстом у краёв (см. :data:`PLATE_TEXTURE_SHARE`).

    Args:
        maps: Карты окна.
        box: Рамка компоненты, px полосы.
        dpi: Разрешение.

    Returns:
        Рамка без плашек с текстом.
    """
    ox, oy = maps.origin
    x0, y0, x1, y1 = (int(v) for v in box)
    tex = maps.texture > maps.light_texture
    letters = letters_in(maps, (0, 0, 0, 0), dpi, on_paper=False)
    body = max(2, int(round(mm(EDGE_BODY_MM, dpi))))
    out = [x0, y0, x1, y1]
    for horizontal, start, step in ((True, 1, 1), (True, 3, -1), (False, 0, 1), (False, 2, -1)):
        if horizontal:
            share = tex[y0 - oy : y1 - oy, x0 - ox : x1 - ox].mean(1)
        else:
            share = tex[y0 - oy : y1 - oy, x0 - ox : x1 - ox].mean(0)
        good = share >= PLATE_TEXTURE_SHARE
        seq = good if step > 0 else good[::-1]
        # Первая полоса снимка: ``body`` строк подряд с текстурой.
        run, cut = 0, None
        for i, g in enumerate(seq):
            run = run + 1 if g else 0
            if run >= body:
                cut = i - body + 1
                break
        if not cut:
            continue
        if horizontal:
            a, b = (y0, y0 + cut) if step > 0 else (y1 - cut, y1)
            band = letters[a - oy : b - oy, x0 - ox : x1 - ox]
        else:
            a, b = (x0, x0 + cut) if step > 0 else (x1 - cut, x1)
            band = letters[y0 - oy : y1 - oy, a - ox : b - ox]
        tone = (maps.tone[a - oy : b - oy, x0 - ox : x1 - ox] if horizontal
                else maps.tone[y0 - oy : y1 - oy, a - ox : b - ox])  # fmt: skip
        toned = tone.size > 0 and float(tone.mean()) < maps.paper - maps.plate_delta
        # Плашка однородна поперёк: участков сплошной текстуры (снимок рядом с подписью в коллаже 1969/11 IMG_0094_1L)
        # в ней почти нет — доля столбцов (строк) полосы, сплошь покрытых растром, меньше PLATE_PHOTO_SHARE.
        btex = (tex[a - oy : b - oy, x0 - ox : x1 - ox] if horizontal
                else tex[y0 - oy : y1 - oy, a - ox : b - ox].T)  # fmt: skip
        plain = btex.size > 0 and float((btex.mean(0) >= PLATE_TEXTURE_SHARE).mean()) < PLATE_PHOTO_SHARE
        if toned and plain and letter_count(band) >= TEXT_GUARD_LETTERS:
            out[start] = b if step > 0 else a
    return tuple(out)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заливка дыр маски: всё, что не достижимо заливкой фона от края окна, — внутри.

    Args:
        mask: Маска uint8 0/1.

    Returns:
        Маска без дыр.
    """
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return (mask.astype(bool) | holes).astype(np.uint8)


@dataclass(frozen=True)
class Component:
    """Компонента маски: рамка в px полосы, площадь, доля бумажного тона."""

    box: tuple[int, int, int, int]
    area: int
    paper_share: float


def photo_components(mask: np.ndarray, maps: Maps, seed: tuple[int, int, int, int], dpi: float) -> list[Component]:
    """Компоненты маски, похожие на снимок и задевающие затравку.

    Отсев: компонента больше чем наполовину вне затравки (сосед — текст колонки рядом, другой снимок), сторона
    меньше :data:`MIN_PHOTO_SIDE_MM`, доля бумажного тона больше :data:`TEXT_PAPER_SHARE` (строки текста, подпись).

    Args:
        mask: Маска окна.
        maps: Карты окна.
        seed: Затравка ``(x0, y0, x1, y1)``, px полосы.
        dpi: Разрешение полосы.

    Returns:
        Компоненты-снимки, рамки в px полосы.
    """
    ox, oy = maps.origin
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_side = mm(MIN_PHOTO_SIDE_MM, dpi)
    papery = maps.tone >= maps.paper - PAPER_TOL
    sx0, sy0, sx1, sy1 = seed[0] - ox, seed[1] - oy, seed[2] - ox, seed[3] - oy
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_side or h < min_side:
            continue
        comp = labels[y : y + h, x : x + w] == i
        # Охват без тонких отростков: тень у края листа, черта, задетая буква тянули рамку по маске на 12 мм
        # (1975/08 IMG_0055_1L — полоса тени справа подняла верх рамки до заголовка).
        bx0, by0, bx1, by1 = robust_extent(comp, ROBUST_EXTENT_SHARE, int(round(mm(EXTENT_GAP_MM, dpi))))
        x, y, w, h = x + bx0, y + by0, bx1 - bx0, by1 - by0
        comp = comp[by0:by1, bx0:bx1]
        tx0, ty0, tx1, ty1 = trim_text_plates(maps, (x + ox, y + oy, x + w + ox, y + h + oy), dpi)
        comp = comp[ty0 - oy - y : ty1 - oy - y, tx0 - ox - x : tx1 - ox - x]
        x, y, w, h = tx0 - ox, ty0 - oy, tx1 - tx0, ty1 - ty0
        if w < min_side or h < min_side:
            continue
        ix = max(0, min(x + w, sx1) - max(x, sx0))
        iy = max(0, min(y + h, sy1) - max(y, sy0))
        if ix * iy < 0.5 * w * h:
            continue
        share = float(papery[y : y + h, x : x + w][comp].mean())
        if share > TEXT_PAPER_SHARE:
            continue
        # Плотный набор (подпись в 2–4 строки, склеенная замыканием): между строками — светлые ряды по всей ширине.
        # У снимка таких рядов нет (белое небо светлее бумаги не бывает: растр есть и там).
        if looks_like_text_block(maps.tone[y : y + h, x : x + w], maps.paper, dpi):
            continue
        out.append(Component((x + ox, y + oy, x + w + ox, y + h + oy), int(area), share))
    return out


def robust_extent(comp: np.ndarray, share: float = ROBUST_EXTENT_SHARE, gap: int = 0) -> tuple[int, int, int, int]:
    """Охват компоненты без отростков: самая длинная полоса строк (затем столбцов), заполненных компонентой не меньше
    чем на долю ``share`` медианной заполненности, с разрывами не длиннее ``gap`` строк.

    Отросток — тень у края листа, черта, задетая буква — и подпись, сцепленная со снимком через такую тень (1975/10
    IMG_0014_1L: полоса тени шириной 30 % снимка на 5 мм, под ней жирная подпись) — отделены от снимка строками с малой
    заполненностью и в самую длинную полосу не входят.

    Args:
        comp: Маска компоненты bool (вырезка по её рамке).
        share: Доля медианной заполненности строки (столбца), с которой строка (столбец) входит в полосу.
        gap: Допустимый разрыв полосы, строк.

    Returns:
        ``(x0, y0, x1, y1)`` в px вырезки (правая и нижняя границы — исключительно).
    """
    rows = comp.sum(1)
    if not rows.any():
        return 0, 0, comp.shape[1], comp.shape[0]
    y0, y1 = longest_run(rows >= share * np.median(rows[rows > 0]), gap)
    cols = comp[y0:y1].sum(0)
    if not cols.any():
        return 0, y0, comp.shape[1], y1
    x0, x1 = longest_run(cols >= share * np.median(cols[cols > 0]), gap)
    return x0, y0, x1, y1


def longest_run(flags: np.ndarray, gap: int) -> tuple[int, int]:
    """Самая длинная полоса ``True`` с разрывами не длиннее ``gap``.

    Args:
        flags: Флаги по строкам (столбцам).
        gap: Допустимый разрыв.

    Returns:
        ``(начало, конец)`` (конец — исключительно); пустые флаги — весь диапазон.
    """
    idx = np.flatnonzero(flags)
    if not len(idx):
        return 0, len(flags)
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(idx) - 1]])
    k = int(np.argmax(idx[ends] - idx[starts]))
    return int(idx[starts[k]]), int(idx[ends[k]]) + 1


def looks_like_text_block(tone: np.ndarray, paper: float, dpi: float) -> bool:
    """Плотный набор, а не снимок: светлые ряды (средний тон не темнее бумаги на :data:`EMPTY_ROW_TOL`) короткими
    просветами — межстрочными (не длиннее :data:`INTERLINE_MAX_MM`), не меньше двух, с долей рядов больше
    :data:`TEXT_EMPTY_ROWS`. Белое небо или светлый фон снимка дают один длинный светлый участок, а не просветы.

    Args:
        tone: Тон в рамке компоненты.
        paper: Уровень бумаги.
        dpi: Разрешение.

    Returns:
        True — похоже на набор.
    """
    empty = tone.mean(1) >= paper - EMPTY_ROW_TOL
    limit = mm(INTERLINE_MAX_MM, dpi)
    runs, start = [], None
    for i, e in enumerate(np.append(empty, False)):
        if e and start is None:
            start = i
        elif not e and start is not None:
            runs.append(i - start)
            start = None
    short = [r for r in runs if r <= limit]
    return len(short) >= 2 and sum(short) > TEXT_EMPTY_ROWS * len(empty)


def group_components(components: list[Component], dpi: float) -> list[tuple[int, int, int, int]]:
    """Сводит компоненты одного снимка (разорванного светлым участком) в рамки: рамки ближе :data:`GROUP_GAP_MM`.

    Args:
        components: Компоненты-снимки.
        dpi: Разрешение.

    Returns:
        Рамки групп, px полосы.
    """
    gap = mm(GROUP_GAP_MM, dpi)
    boxes = [list(c.box) for c in components]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


def light_map(maps: Maps) -> np.ndarray:
    """Светлый растр окна: тон ниже бумаги на :data:`LIGHT_TONE_DELTA` или разброс выше :data:`LIGHT_TEXTURE_THR`.

    Args:
        maps: Карты окна.

    Returns:
        Маска bool окна.
    """
    return (maps.tone < maps.paper - maps.light_delta) | (maps.texture > maps.light_texture)


def grow_box(light: np.ndarray, maps: Maps, box: tuple[int, int, int, int], dpi: float) -> tuple[int, int, int, int]:
    """Рост рамки ядра на светлые края снимка в два шага: светлый растр, затем ровная светлая плашка (небо).

    1. Каждая сторона идёт наружу шагом :data:`GROW_STEP_MM`, пока полоска покрыта растром снимка не меньше чем на
       :data:`GROW_COVER`: разброс выше местного порога или тон темнее бумаги на :data:`CORE_TONE_DELTA`. Гладкий
       светлый тон без растра (тень у края листа ~20 уровней ниже бумаги, 1972/07 IMG_0020_1L) растром не считается.
    2. Затем по ровной плашке (:data:`PLATE_COVER`), но сторона, выросшая по плашке, принимается, только если за плашкой
       бумага (средний тон следующей полоски не темнее бумаги на половину плашечного порога): небо снимка кончается
       ступенькой к бумаге (МТС 1990/09 IMG_0310), а тень у края листа и сор — нет.

    Полоска с буквами строк (по сильной и по слабой краске — просвет оборота) рост останавливает.

    Args:
        light: Светлый растр окна (:func:`light_map`; для шага 1 ужесточается, см. выше).
        maps: Карты окна (начало окна).
        box: Рамка ядра, px полосы.
        dpi: Разрешение.

    Returns:
        Рамка после роста, px полосы (не дальше окна).
    """
    ox, oy = maps.origin
    h, w = light.shape
    x0, y0, x1, y1 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    step = max(1, int(round(mm(GROW_STEP_MM, dpi))))
    raster = light & ((maps.texture > maps.light_texture) | (maps.tone < maps.paper - CORE_TONE_DELTA))
    # Плашка — только ровный светлый тон: не растр и не краска (буквы подписи с ореолом размытия давали 85 % «плашки»
    # вместе со светлым растром, и рамка уходила на подпись — 1975/10 IMG_0014_1L).
    plate = ((maps.tone < maps.paper - maps.plate_delta) & (maps.tone > maps.paper - PLATE_MAX_DELTA)
             & (maps.texture <= PLATE_MAX_TEXTURE))  # fmt: skip
    # Строки текста рост не проходит (строка «чает сосчитывать…» над снимком 1969/12 IMG_0141_1L; просвет букв оборота
    # слева от снимков 1972/07 IMG_0020_1L и IMG_0011_1L — по слабой краске).
    letters = letters_in(maps, box, dpi) | letters_in(maps, box, dpi, GROW_LETTER_DELTA)
    letter_labels = cv2.connectedComponents(letters.astype(np.uint8), connectivity=8)[1]
    raster = page_edge_cut(raster.astype(np.uint8), maps, dpi) > 0
    plate = page_edge_cut(plate.astype(np.uint8), maps, dpi) > 0
    paper_line = maps.paper - maps.plate_delta / 2.0

    def strip(side: int, xs: tuple[int, int, int, int], d: int) -> tuple[slice, slice] | None:
        """Полоска шага ``d`` наружу от стороны ``side`` (0 — верх, 1 — низ, 2 — лево, 3 — право) или None."""
        a0, b0, a1, b1 = xs
        if side == 0:
            return None if b0 - d < 0 else (slice(b0 - d, b0 - d + step), slice(a0, a1))
        if side == 1:
            return None if b1 + d > h else (slice(b1 + d - step, b1 + d), slice(a0, a1))
        if side == 2:
            return None if a0 - d < 0 else (slice(b0, b1), slice(a0 - d, a0 - d + step))
        return None if a1 + d > w else (slice(b0, b1), slice(a1 + d - step, a1 + d))

    def cover(mask: np.ndarray, sl) -> float:
        part = mask[sl]
        return float(part.mean()) if part.size else 0.0

    edges = [x0, y0, x1, y1]  # лево, верх, право, низ — в порядке рамки
    order = {0: 1, 1: 3, 2: 0, 3: 2}  # сторона → индекс в рамке
    sign = {0: -1, 1: 1, 2: -1, 3: 1}
    for mask, cover_thr, need_paper in ((raster, GROW_COVER, False), (plate, PLATE_COVER, True)):
        for side in (0, 1, 2, 3):
            start = edges[order[side]]
            d = step
            while True:
                xs = (edges[0], edges[1], edges[2], edges[3])
                sl = strip(side, xs, d)
                if sl is None or len(np.unique(letter_labels[sl])) - 1 >= GROW_STOP_LETTERS or not (cover(mask | raster, sl) >= cover_thr
                                                           or cover(raster, sl) >= GROW_COVER):  # fmt: skip
                    break
                d += step
            grown = d - step
            if not grown:
                continue
            if need_paper:
                # За плашкой должна быть бумага: следующая полоска (или две) светлая и ровная.
                nxt = strip(side, (edges[0], edges[1], edges[2], edges[3]), grown + 2 * step)
                if nxt is None or float(maps.tone[nxt].mean()) < paper_line:
                    continue
            edges[order[side]] = start + sign[side] * grown
    x0, y0, x1, y1 = edges
    return x0 + ox, y0 + oy, x1 + ox, y1 + oy


def grow_parts(maps: Maps, box, sharp: tuple[bool, ...], dpi: float):
    """Частичный рост итоговой рамки у нечётких сторон (:func:`grow_partial`): часть снимка уже стороны за ней.

    Идёт после подгонки сторон и охвата объекта, чтобы не сдвигать рамку по маске: сдвиг на 3–6 px менял точки края
    и откат на затравку (1975/08 IMG_0055_1L — снимок в полосе у края кадра, 1972/03 IMG_0116_1L — левая сторона
    теряла чёткость). Растр — растровые точки (локальный контраст, :func:`halftone_map`) или заметная краска, а не
    разброс: шум пятнистой бумаги и тень у края листа дают разброс у порога светлого растра и тянули сторону на
    3–47 мм (1972/12 IMG_0154_1L, 1972/07 IMG_0020_1L, прогон стенда v14); тень темнее бумаги на 20–30, до краски
    (:func:`object_ink`) не дотягивает.

    Args:
        maps: Карты окна.
        box: Итоговая рамка, px полосы.
        sharp: Чёткость сторон (верх, низ, левая, правая); чёткая сторона не растёт.
        dpi: Разрешение.

    Returns:
        Рамка после роста, px полосы.
    """
    ox, oy = maps.origin
    h, w = maps.tone.shape
    halftone, dots = halftone_map(maps, dpi)
    raster = page_edge_cut((halftone | object_ink(maps)).astype(np.uint8), maps, dpi) > 0
    dots = page_edge_cut(dots.astype(np.uint8), maps, dpi) > 0
    # Буквы — вне рамки, сжатой на OBJECT_CORE_INSET_MM, но больше чем наполовину снаружи рамки: буквы подписи,
    # стоящие на самой линии рамки (подпись «Фото 6» у кромки снимка коллажа 1969/11 IMG_0094_1L), останавливают рост,
    # а пятна растра у кромки самого снимка — нет.
    inset = mm(OBJECT_CORE_INSET_MM, dpi)
    inner = (box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset)
    strong = letters_in(maps, inner, dpi)
    letter_labels = outside_letters(strong | letters_in(maps, inner, dpi, GROW_LETTER_DELTA), box, maps.origin)
    # Добивка по слабым точкам останавливается только сильной краской: точки бледной части по слабой краске
    # складываются в «буквы» ~1 мм.
    strong_labels = outside_letters(strong, box, maps.origin)
    edges = [int(np.clip(round(box[0] - ox), 0, w)), int(np.clip(round(box[1] - oy), 0, h)),
             int(np.clip(round(box[2] - ox), 0, w)), int(np.clip(round(box[3] - oy), 0, h))]  # fmt: skip
    step = max(1, int(round(mm(GROW_STEP_MM, dpi))))
    order = {0: 1, 1: 3, 2: 0, 3: 2}  # сторона → индекс в рамке
    out = list(box)
    for side in (0, 1, 2, 3):
        if sharp and sharp[side]:
            continue
        pos, active = grow_partial(raster, letter_labels, edges, side, step, dpi)
        if pos != edges[order[side]]:
            pos = dot_tip(dots, strong_labels, edges, side, pos, active, step, dpi)
            out[order[side]] = float(pos + (ox if side in (2, 3) else oy))
    return tuple(out)


def grow_partial(raster: np.ndarray, letter_labels: np.ndarray, edges: list, side: int, step: int, dpi: float) -> int:
    """Частичный рост одной стороны: наружу, пока за ней есть растр снимка, сцепленный с растром предыдущей полоски,
    участками от :data:`PART_RUN_MM` и всего от :data:`PART_MIN_MM` вдоль стороны.

    Полный рост (:func:`grow_box`) требует растра на :data:`GROW_COVER` ширины стороны и останавливается у части снимка
    уже стороны: верх кепки над белым небом (1975/11 IMG_0054_1L — 3 мм кепки над рамкой, кепка на трети ширины).
    Бумажный зазор (подпись, соседняя колонка) сцепку рвёт; полоску с :data:`GROW_STOP_LETTERS` буквами рост не проходит.

    Args:
        raster: Растр снимка окна (bool; без полосы у края кадра).
        letter_labels: Метки букв строк окна (0 — не буква).
        edges: Рамка в координатах окна ``[x0, y0, x1, y1]``.
        side: Сторона: 0 — верх, 1 — низ, 2 — лево, 3 — право.
        step: Шаг роста, px.
        dpi: Разрешение.

    Returns:
        Новая координата стороны в окне.
    """
    h, w = raster.shape
    x0, y0, x1, y1 = (int(v) for v in edges)
    run_px = max(1, int(round(mm(PART_RUN_MM, dpi))))
    min_px = mm(PART_MIN_MM, dpi)
    kd = max(1, int(round(mm(PART_LINK_MM, dpi))))
    horizontal = side in (0, 1)
    outward = -1 if side in (0, 2) else 1
    base = (y0 if side == 0 else y1) if horizontal else (x0 if side == 2 else x1)
    limit = h if horizontal else w

    def columns(lo: int, hi: int) -> np.ndarray:
        """Доля растра в каждом столбце (строке) полоски ``[lo, hi)`` поперёк стороны ≥ 0.5."""
        part = raster[lo:hi, x0:x1] if horizontal else raster[y0:y1, lo:hi].T
        return part.mean(0) >= 0.5 if part.size else np.zeros(0, bool)

    # Затравка — растр в полоске шага у самой стороны изнутри.
    lo, hi = (base, base + step) if outward < 0 else (base - step, base)
    prev = columns(max(lo, 0), min(hi, limit))
    pos = base
    while prev.any():
        lo, hi = (pos - step, pos) if outward < 0 else (pos, pos + step)
        if lo < 0 or hi > limit:
            break
        cand = columns(lo, hi)
        linked = cv2.dilate(prev.astype(np.uint8)[None], np.ones((1, 2 * kd + 1), np.uint8))[0] > 0
        active = drop_short_runs(cand & linked, run_px)
        # Мало растра — часть кончилась; растр почти во всю ширину — плашка или поле, которые решает полный рост
        # с проверкой плашек (тонированная плашка с заголовком над снимком 1975/03 IMG_0103_1L).
        if active.sum() < min_px or active.mean() >= GROW_COVER:
            break
        letters = letter_labels[lo:hi, x0:x1] if horizontal else letter_labels[y0:y1, lo:hi]
        if len(np.unique(letters)) - 1 >= GROW_STOP_LETTERS:
            break
        prev, pos = active, pos + outward * step
    return pos, prev


def outside_letters(letters: np.ndarray, box, origin: tuple[int, int]) -> np.ndarray:
    """Метки букв, лежащих больше чем наполовину снаружи рамки (остальные — 0): буква на самой линии рамки в счёт идёт,
    пятно растра у кромки снимка внутри рамки — нет.

    Args:
        letters: Маска букв окна (bool).
        box: Рамка, px полосы.
        origin: Начало окна на полосе.

    Returns:
        Метки букв окна (int32), 0 — не буква.
    """
    n, labels = cv2.connectedComponents(letters.astype(np.uint8), connectivity=8)
    if n <= 1:
        return labels
    ox, oy = origin
    h, w = labels.shape
    inside = np.zeros((h, w), bool)
    x0, y0 = int(np.clip(round(box[0] - ox), 0, w)), int(np.clip(round(box[1] - oy), 0, h))
    x1, y1 = int(np.clip(round(box[2] - ox), 0, w)), int(np.clip(round(box[3] - oy), 0, h))
    inside[y0:y1, x0:x1] = True
    total = np.bincount(labels.ravel(), minlength=n)
    within = np.bincount(labels[inside], minlength=n)
    keep = within <= 0.5 * total
    keep[0] = False
    return np.where(keep[labels], labels, 0)


def dot_tip(
    dots: np.ndarray,
    letter_labels: np.ndarray,
    edges: list,
    side: int,
    pos: int,
    active: np.ndarray,
    step: int,
    dpi: float,
) -> int:
    """Добивка выросшей стороны по самим растровым точкам: наружу не дальше :data:`TIP_REACH_MM` (растр по плотности
    сжат на полуокно и у бледного края теряет ещё до окна, :func:`halftone_map`), пока в столбцах растущей части полоски шага есть точки — доля от
    :data:`TIP_DOT_SHARE`. Бледная макушка кепки (1975/11 IMG_0054_1L) — редкие точки за краем сжатого растра;
    у резкой кромки снимка с чистой бумагой за ней точек нет, сторона не сдвигается.

    Args:
        dots: Растровые точки окна (bool).
        letter_labels: Метки букв строк окна (0 — не буква): полоска с :data:`GROW_STOP_LETTERS` буквами добивку
            останавливает.
        edges: Рамка в координатах окна ``[x0, y0, x1, y1]`` (до роста — диапазон вдоль стороны).
        side: Сторона: 0 — верх, 1 — низ, 2 — лево, 3 — право.
        pos: Координата стороны после роста.
        active: Столбцы растущей части в последней полоске (вдоль стороны).
        step: Шаг, px.
        dpi: Разрешение.

    Returns:
        Координата стороны после добивки.
    """
    h, w = dots.shape
    x0, y0, x1, y1 = (int(v) for v in edges)
    horizontal = side in (0, 1)
    outward = -1 if side in (0, 2) else 1
    limit = h if horizontal else w
    reach = int(round(mm(TIP_REACH_MM, dpi)))
    cols = cv2.dilate(active.astype(np.uint8)[None], np.ones((1, 2 * reach + 1), np.uint8))[0] > 0
    best, cur = pos, pos
    while abs(cur - pos) < reach:
        lo, hi = (cur - step, cur) if outward < 0 else (cur, cur + step)
        if lo < 0 or hi > limit:
            break
        part = dots[lo:hi, x0:x1] if horizontal else dots[y0:y1, lo:hi].T
        if not part.size or not cols.any():
            break
        letters = letter_labels[lo:hi, x0:x1] if horizontal else letter_labels[y0:y1, lo:hi]
        if len(np.unique(letters)) - 1 >= GROW_STOP_LETTERS:
            break
        cur = cur + outward * step
        if float(part[:, cols].mean()) >= TIP_DOT_SHARE:
            best = cur
    return best


def halftone_map(maps: Maps, dpi: float) -> tuple[np.ndarray, np.ndarray]:
    """Бледный растр снимка по растровым точкам: пиксель слабо размытого тона (``Maps.fine``) темнее светлого фона в
    окне :data:`DOT_WIN_MM` на :data:`DOT_CONTRAST` при разбросе выше местного порога светлого растра; растр — где
    таких точек в окне :data:`DOT_DENSITY_MM` не меньше :data:`DOT_SHARE`.

    Локальный контраст точки не зависит от уровня бумаги: бледный верх кепки (тон 240–250 при бумаге 254, 1975/11
    IMG_0054_1L) — 5–6 % точек в полосе роста, а тень у края листа и пятнистая бумага, у которых разброс тот же, —
    0–1 % (1972/07 IMG_0020_1L, 1972/12 IMG_0154_1L, 1968/10 IMG_0037_1L, 1976/08 IMG_0091_1L).

    Args:
        maps: Карты окна.
        dpi: Разрешение.

    Returns:
        ``(растр, точки)`` — маски bool окна: растр по плотности точек (сжатый на полуокно) и слабые точки для добивки
        края (контраст от :data:`TIP_DOT_CONTRAST`).
    """
    fine = maps.fine if maps.fine is not None else maps.tone
    kw = max(3, int(round(mm(DOT_WIN_MM, dpi))) | 1)
    bright = cv2.dilate(fine, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw)))
    textured = maps.texture > maps.light_texture
    dots = ((bright - fine > DOT_CONTRAST) & textured).astype(np.float32)
    # Для добивки края (:func:`dot_tip`) — точки послабее: у макушки бледного снимка контраст точек 8–12.
    tips = (bright - fine > TIP_DOT_CONTRAST) & textured
    kd = max(3, int(round(mm(DOT_DENSITY_MM, dpi))) | 1)
    dense = (cv2.blur(dots, (kd, kd)) >= DOT_SHARE).astype(np.uint8)
    # Окно плотности размазывает растр на полуокно за кромку снимка — сжатие на полуокно возвращает его к краю точек
    # (без него нечёткая сторона уходила в чистую бумагу на 0.5–1.3 мм: 1972/07 IMG_0020_1L, 1970/01 IMG_0026_1L).
    return cv2.erode(dense, cv2.getStructuringElement(cv2.MORPH_RECT, (kd, kd))) > 0, tips


def drop_short_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
    """Флаги без участков короче ``min_len`` подряд.

    Args:
        flags: Флаги bool (N,).
        min_len: Наименьшая длина участка.

    Returns:
        Флаги bool (N,).
    """
    out = flags.copy()
    n = len(flags)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        if j - i < min_len:
            out[i:j] = False
        i = j
    return out


def edge_points(maps: Maps, box: tuple[int, int, int, int], side: Side, dpi: float) -> tuple[np.ndarray, ...]:
    """Точки края вдоль одной стороны рамки: профиль тона поперёк, внешняя ступенька «снимок → бумага».

    На каждой позиции вдоль стороны (шаг :data:`EDGE_STEP_MM`, без :data:`CORNER_SKIP` у углов) берётся профиль тона
    от ``PROFILE_IN_MM`` внутрь до ``PROFILE_OUT_MM`` наружу, усреднённый по полосе :data:`PROFILE_BAND_MM` вдоль
    стороны. Край — самый внешний максимум производной «светлеет наружу», за которым бумага (тон не ниже бумаги на
    :data:`OUTSIDE_PAPER_TOL` в окне ``[SIDE_NEAR_MM, SIDE_FAR_MM]``) и у которого ступенька не меньше
    :data:`EDGE_MIN_CONTRAST`. Положение — с параболической доводкой.

    Args:
        maps: Карты окна.
        box: Рамка по маске, px полосы.
        side: Сторона.
        dpi: Разрешение.

    Returns:
        ``(вдоль, поперёк, ступенька, ширина, всего_позиций, a0, a1)``: координаты годных точек в px полосы, их
        ступенька и ширина перехода (px), число проверенных позиций и диапазон вдоль.
    """
    ox, oy = maps.origin
    tone = maps.tone
    h, w = tone.shape
    x0, y0, x1, y1 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    if side.horizontal:
        a_lo, a_hi, base = x0, x1, (y0 if side is Side.TOP else y1 - 1)
    else:
        a_lo, a_hi, base = y0, y1, (x0 if side is Side.LEFT else x1 - 1)
    length = a_hi - a_lo
    a0, a1 = a_lo + CORNER_SKIP * length, a_hi - CORNER_SKIP * length
    step = mm(EDGE_STEP_MM, dpi)
    positions = np.arange(a0, a1 + 1e-6, step)
    n_in, n_out = int(round(mm(PROFILE_IN_MM, dpi))), int(round(mm(PROFILE_OUT_MM, dpi)))
    band = max(1, int(round(mm(PROFILE_BAND_MM, dpi) / 2)))
    near, far = mm(SIDE_NEAR_MM, dpi), mm(SIDE_FAR_MM, dpi)
    # Поперечная координата s: 0 — сторона рамки, положительная — наружу.
    s = np.arange(-n_in, n_out + 1)
    across = base + side.outward * s
    ok = (across >= 0) & (across < (h if side.horizontal else w))
    s, across = s[ok], across[ok]
    out_a, out_c, out_k, out_wd = [], [], [], []
    for a in positions:
        lo, hi = int(round(a)) - band, int(round(a)) + band + 1
        lo, hi = max(lo, 0), min(hi, w if side.horizontal else h)
        if hi <= lo or len(s) < 5:
            continue
        prof = tone[across, lo:hi].mean(1) if side.horizontal else tone[lo:hi, across].mean(0)
        tex = maps.texture[across, lo:hi].mean(1) if side.horizontal else maps.texture[lo:hi, across].mean(0)
        found = step_on_profile(prof, s, maps.paper, near, far, maps.light_delta, tex, maps.light_texture)
        if found is None:
            continue
        pos, contrast, width = found
        out_a.append(a + (ox if side.horizontal else oy))
        out_c.append(base + side.outward * pos + (oy if side.horizontal else ox))
        out_k.append(contrast)
        out_wd.append(width)
    shift = ox if side.horizontal else oy
    return (np.array(out_a), np.array(out_c), np.array(out_k), np.array(out_wd), len(positions), a0 + shift,
            a1 + shift)  # fmt: skip


def step_on_profile(
    prof: np.ndarray,
    s: np.ndarray,
    paper: float,
    near: float,
    far: float,
    paper_tol: float = 12.0,
    texture: np.ndarray | None = None,
    light_texture: float = LIGHT_TEXTURE_THR,
) -> tuple[float, float, float] | None:
    """Внешняя ступенька «темно внутри → бумага снаружи» на профиле тона.

    С профилем разброса (``texture``) край годен, только если внутри растр снимка (см. :data:`SHADE_PAPER_TOL`):
    снаружи допускается и затенённая гладкая бумага, а внешний край гладкого серого клина тени краем не считается.

    Args:
        prof: Тон вдоль поперечной координаты.
        s: Поперечная координата профиля, px (наружу — больше).
        paper: Уровень бумаги.
        near, far: Окно тона по обе стороны края ``[near, far]`` px.
        paper_tol: Снаружи края тон не ниже бумаги больше чем на столько (местный порог бумаги окна).
        texture: Разброс яркости вдоль того же профиля или None — проверки растра внутри и тени снаружи нет
            (прежнее поведение).
        light_texture: Местный порог разброса светлого растра (``Maps.light_texture``).

    Returns:
        ``(положение s, ступенька, ширина перехода px)`` или None — годной ступеньки нет.
    """
    grad = np.gradient(prof)
    n = len(prof)
    best = None
    # Кандидаты — локальные максимумы производной (светлеет наружу), от внешнего края профиля внутрь.
    for i in range(n - 2, 0, -1):
        if not (grad[i] > 0 and grad[i] >= grad[i - 1] and grad[i] >= grad[i + 1]):
            continue
        out_sel = (s >= s[i] + near) & (s <= s[i] + far)
        in_sel = (s <= s[i] - near) & (s >= s[i] - far)
        if out_sel.sum() < 2 or in_sel.sum() < 2:
            continue
        # Снаружи — наибольший тон окна: бумага хоть где-то в окне есть, даже если в 1–1.5 мм от края начинается
        # текст соседней колонки или подпись.
        outside, inside = float(prof[out_sel].max()), float(prof[in_sel].mean())
        contrast = outside - inside
        if contrast < EDGE_MIN_CONTRAST:
            continue
        if outside < paper - max(paper_tol, OUTSIDE_PAPER_TOL):
            # Снаружи не бумага окна — годится только затенённая гладкая бумага при сильной ступеньке.
            if texture is None or outside < paper - SHADE_PAPER_TOL or contrast < SHADE_MIN_CONTRAST:
                continue
            # Гладкость — по медиане: внешняя ступенька узкого клина сама даёт разброс 7–11 на краю окна.
            if float(np.median(texture[out_sel])) > light_texture:
                continue
        if texture is not None:
            # Внутри — растр снимка: разброс по всему окну (наименьший — разброс у кромки снимка за клином шириной
            # 1–1.5 мм заходит в окно краем и поднимает среднее) или ступенька заметной краски.
            if float(np.median(texture[in_sel])) <= light_texture and contrast < OBJECT_SMOOTH_DELTA:
                continue
        # Параболическая доводка максимума производной.
        d = grad[i - 1] - 2 * grad[i] + grad[i + 1]
        frac = 0.5 * (grad[i - 1] - grad[i + 1]) / d if d < 0 else 0.0
        # Ширина перехода: ступенька, делённая на наибольшую производную (у размытой ступеньки — σ·√(2π)).
        width = contrast / max(float(grad[i]), 1e-6)
        best = (float(s[i] + np.clip(frac, -0.5, 0.5) * (s[i + 1] - s[i])), contrast, width)
        break
    return best


def robust_poly(
    along: np.ndarray, across: np.ndarray, degree: int, huber: float, outlier: float, passes: int = FIT_PASSES
) -> tuple[np.ndarray | None, np.ndarray]:
    """Робастная подгонка ``across = poly(along)``: IRLS с весами Хьюбера и отсевом выбросов.

    Степень снижается, если точек мало (не меньше ``3 × (степень + 1)``). Сначала прямая по медианам (устойчивый
    старт), затем проходы IRLS.

    Args:
        along, across: Точки (N,).
        degree: Степень многочлена.
        huber: Порог Хьюбера, px.
        outlier: Отсев: остаток больше — вес 0, px.
        passes: Проходов IRLS.

    Returns:
        ``(коэффициенты или None, маска годных точек)``.
    """
    n = len(along)
    inlier = np.zeros(n, bool)
    if n < 4:
        return None, inlier
    degree = min(degree, max(1, n // 3 - 1))
    # Устойчивый старт: прямая Тейла–Сена по подвыборке пар.
    coef = theil_sen(along, across)
    if degree > 1:
        coef = np.concatenate([np.zeros(degree - 1), coef])
    for _ in range(passes):
        r = across - np.polyval(coef, along)
        absr = np.abs(r)
        weight = np.where(absr <= huber, 1.0, huber / np.maximum(absr, 1e-9))
        weight[absr > outlier] = 0.0
        if (weight > 0).sum() < degree + 2:
            return None, inlier
        coef = np.polyfit(along, across, degree, w=np.sqrt(weight))
    inlier = np.abs(across - np.polyval(coef, along)) <= outlier
    return coef, inlier


def theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 4000) -> np.ndarray:
    """Прямая Тейла–Сена: медиана наклонов пар и медиана сдвигов.

    Args:
        x, y: Точки (N,), N ≥ 2.
        max_pairs: Предел числа пар (берутся пары через равный шаг).

    Returns:
        ``[наклон, сдвиг]`` (порядок ``np.polyval``).
    """
    n = len(x)
    i, j = np.triu_indices(n, 1)
    if len(i) > max_pairs:
        sel = np.linspace(0, len(i) - 1, max_pairs).astype(int)
        i, j = i[sel], j[sel]
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else 0.0
    return np.array([slope, float(np.median(y - slope * x))])


def fit_side(maps: Maps, box: tuple[int, int, int, int], side: Side, dpi: float) -> SideFit:
    """Точки края стороны и их робастная подгонка многочленом :data:`SIDE_DEGREE`.

    Args:
        maps: Карты окна.
        box: Рамка по маске, px полосы.
        side: Сторона.
        dpi: Разрешение.

    Returns:
        Подгонка стороны.
    """
    along, across, contrast, width, total, a0, a1 = edge_points(maps, box, side, dpi)
    huber, outlier = mm(FIT_HUBER_MM, dpi), mm(FIT_OUTLIER_MM, dpi)
    coef, inlier = robust_poly(along, across, SIDE_DEGREE, huber, outlier)
    model = None if coef is None else SideModel(np.asarray(coef))
    if model is not None:
        model, inlier = robust_spline(along, across, model, inlier, dpi, huber, outlier)
    return SideFit(side, along, across, contrast, width, inlier, model, (a0, a1), total, 25.4 / dpi)


def robust_spline(
    along: np.ndarray,
    across: np.ndarray,
    start: SideModel,
    inlier: np.ndarray,
    dpi: float,
    huber: float,
    outlier: float,
) -> tuple[SideModel, np.ndarray]:
    """Кубический сплайн стороны с узлом на :data:`SIDE_KNOT_MM` поверх робастного многочлена: IRLS Хьюбера от его
    остатков, выбросы (дальше ``outlier``) — с весом 0.

    Args:
        along, across: Точки края.
        start: Многочлен-старт.
        inlier: Годные точки старта.
        dpi: Разрешение.
        huber, outlier: Пороги Хьюбера и отсева, px.

    Returns:
        ``(модель, годные точки)``; сторона короче двух узлов или неудачный сплайн — старт как есть.
    """
    from scipy.interpolate import LSQUnivariateSpline

    order = np.argsort(along)
    a, c = along[order], across[order]
    good = inlier[order]
    if good.sum() < 8:
        return start, inlier
    lo, hi = float(a[good].min()), float(a[good].max())
    n_int = int(min(SIDE_MAX_KNOTS, np.floor((hi - lo) / mm(SIDE_KNOT_MM, dpi)) - 1))
    if n_int < 1:
        return start, inlier
    knots = np.linspace(lo, hi, n_int + 2)[1:-1]
    r = np.abs(c - start(a))
    model = start
    for _ in range(FIT_PASSES):
        weight = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-9))
        weight[r > outlier] = 0.0
        sel = (weight > 0) & (a >= lo) & (a <= hi)
        # Между узлами должны быть точки (условие Шёнберга–Уитни), иначе сплайн не строится — тогда старт.
        if sel.sum() < n_int + 8 or np.any(np.histogram(a[sel], np.concatenate([[lo], knots, [hi]]))[0] < 2):
            return model, np.abs(across - model(along)) <= outlier
        try:
            spline = LSQUnivariateSpline(a[sel], c[sel], knots, w=np.sqrt(weight[sel]), k=3)
        except ValueError:
            return model, np.abs(across - model(along)) <= outlier
        t, coef, k = spline._eval_args
        model = SideModel(np.asarray(coef), np.asarray(t), k, lo, hi)
        r = np.abs(c - model(a))
    return model, np.abs(across - model(along)) <= outlier


def corners_of(sides: dict[Side, SideFit], box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Углы кривого четырёхугольника: пересечения соседних сторон (простая итерация от углов рамки).

    Args:
        sides: Подгонки сторон.
        box: Рамка — начальные приближения углов.

    Returns:
        Углы (4, 2) — лв, пв, пн, лн — или None, если у какой-то стороны нет подгонки или итерация не сошлась.
    """
    if any(sides[s].coef is None for s in Side):
        return None
    x0, y0, x1, y1 = box
    pairs = ((Side.TOP, Side.LEFT, (x0, y0)), (Side.TOP, Side.RIGHT, (x1, y0)),
             (Side.BOTTOM, Side.RIGHT, (x1, y1)), (Side.BOTTOM, Side.LEFT, (x0, y1)))  # fmt: skip
    # Угол дальше половины большей стороны рамки от угла рамки — итерация ушла (крутая подгонка стороны по сору).
    reach = 0.5 * max(x1 - x0, y1 - y0)
    out = []
    with np.errstate(over="ignore", invalid="ignore"):
        for horiz, vert, (x, y) in pairs:
            start = (x, y)
            for _ in range(20):
                y = float(sides[horiz].coef(x))
                x = float(sides[vert].coef(y))
                if not (np.isfinite(x) and np.isfinite(y)):
                    return None
            if max(abs(x - start[0]), abs(y - start[1])) > reach:
                return None
            out.append((x, y))
    return np.array(out)


def refine_photo(
    maps: Maps, box: tuple[int, int, int, int], seed: tuple[int, int, int, int], dpi: float, sole: bool = True
) -> PhotoBound:
    """Уточнение одного фото: подгонка четырёх сторон, углы, рамка (чёткие стороны — по кривой, прочие — по маске).

    Рамка по чёткой стороне — самая внешняя точка кривой между углами: рамка накрывает снимок целиком. Нечёткая
    сторона, ушедшая внутрь затравки дальше :data:`SOFT_INSET_MM`, возвращается на сторону затравки (если снимок
    в затравке один): у края снимка белое небо или светлая стена без растра, и маска их не видит (1970/11
    IMG_0054_1L — верх на 24 мм ниже, 1971/06 IMG_0125_1L — на 10 мм).

    Args:
        maps: Карты окна.
        box: Рамка по маске, px полосы.
        seed: Затравка.
        dpi: Разрешение.
        sole: Снимок в затравке один (затравку можно брать запасной стороной).

    Returns:
        Фото.
    """
    sides = {s: fit_side(maps, box, s, dpi) for s in Side}
    sharp = {s: sides[s].sharp() for s in Side}
    corners = corners_of(sides, box)
    x0, y0, x1, y1 = (float(v) for v in box)
    if corners is not None:
        tl, tr, br, bl = corners
        spans = {Side.TOP: (tl[0], tr[0]), Side.BOTTOM: (bl[0], br[0]),
                 Side.LEFT: (tl[1], bl[1]), Side.RIGHT: (tr[1], br[1])}  # fmt: skip
        for s, (lo, hi) in spans.items():
            if not sharp[s] or hi - lo < 2:
                continue
            c = sides[s].curve(lo, hi)[:, 1 if s.horizontal else 0]
            if not np.all(np.isfinite(c)):
                continue
            if s is Side.TOP:
                y0 = float(c.min())
            elif s is Side.BOTTOM:
                y1 = float(c.max())
            elif s is Side.LEFT:
                x0 = float(c.min())
            else:
                x1 = float(c.max())
    if sole:
        # Запасная сторона — сторона затравки, если между ней и рамкой нет строк текста (подпись сбоку от снимка
        # внутри затравки детектора, 1969/11 IMG_0094_1L, в рамку не идёт).
        inset = mm(SOFT_INSET_MM, dpi)
        letters = letters_in(maps, (x0, y0, x1, y1), dpi)
        ox, oy = maps.origin

        def free(a0: float, a1: float, b0: float, b1: float) -> bool:
            band = letters[int(max(b0 - oy, 0)) : int(max(b1 - oy, 0)), int(max(a0 - ox, 0)) : int(max(a1 - ox, 0))]
            return letter_count(band) < TEXT_GUARD_LETTERS

        soft = {s: not sharp[s] for s in Side}
        if soft[Side.TOP] and y0 - seed[1] > inset and free(x0, x1, seed[1], y0):
            y0 = float(seed[1])
        if soft[Side.BOTTOM] and seed[3] - y1 > inset and free(x0, x1, y1, seed[3]):
            y1 = float(seed[3])
        if soft[Side.LEFT] and x0 - seed[0] > inset and free(seed[0], x0, y0, y1):
            x0 = float(seed[0])
        if soft[Side.RIGHT] and seed[2] - x1 > inset and free(x1, seed[2], y0, y1):
            x1 = float(seed[2])
    order = (Side.TOP, Side.BOTTOM, Side.LEFT, Side.RIGHT)
    # Кривой четырёхугольник — только по правилу четырёх сторон: все чёткие и сходятся в правдоподобных углах.
    reason = "" if all(sharp.values()) else "не все стороны чёткие"
    if not reason:
        reason = quad_check(sides, corners, box, dpi)
    quad = corners if not reason else None
    return PhotoBound((x0, y0, x1, y1), box, seed, tuple(sides[s] for s in order), quad, dpi,
                      tuple(sharp[s] for s in order), reason)  # fmt: skip


def quad_check(
    sides: dict[Side, SideFit], corners: np.ndarray | None, box: tuple[int, int, int, int], dpi: float
) -> str:
    """Правило четырёх сторон для чётких сторон: сходятся ли они в связный правдоподобный четырёхугольник.

    Args:
        sides: Подгонки сторон.
        corners: Углы (лв, пв, пн, лн) или None.
        box: Рамка по маске, px.
        dpi: Разрешение.

    Returns:
        Пусто — четырёхугольник годен; иначе причина отказа.
    """
    if corners is None:
        return "нет углов"
    tl, tr, br, bl = corners
    spans = {Side.TOP: (tl[0], tr[0]), Side.BOTTOM: (bl[0], br[0]), Side.LEFT: (tl[1], bl[1]),
             Side.RIGHT: (tr[1], br[1])}  # fmt: skip
    for side, (lo, hi) in spans.items():
        gap = max(mm(CORNER_GAP_MM, dpi), CORNER_GAP_SHARE * (hi - lo))
        fit = sides[side]
        a = fit.along[fit.inlier]
        if hi - lo < mm(MIN_PHOTO_SIDE_MM, dpi) or not len(a):
            return f"сторона {side.value} короче {MIN_PHOTO_SIDE_MM:g} мм"
        skip = CORNER_SKIP * (hi - lo)
        if a.min() - lo - skip > gap or hi - a.max() - skip > gap:
            return f"точки края {side.value} не доходят до угла"
    for i in range(4):
        v1, v2 = corners[i - 1] - corners[i], corners[(i + 1) % 4] - corners[i]
        cos = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9))
        if abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90.0) > CORNER_ANGLE_DEG:
            return "угол не прямой"
    if not cv2.isContourConvex(np.round(corners).astype(np.int32).reshape(-1, 1, 2)):
        return "невыпуклый"
    qbox = (corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max())
    if max(abs(float(q) - float(b)) for q, b in zip(qbox, box)) > mm(QUAD_BOX_MM, dpi):
        return "охват далеко от рамки по маске"
    return ""


def object_mask(maps: Maps, core: tuple[float, float, float, float], dpi: float, others: tuple = ()) -> np.ndarray:
    """Объект снимка в окне: всё темнее местной бумаги на плашечный порог (краска, полутон, светлая растушёвка),
    сцепленное замыканием :data:`OBJECT_CLOSE_MM` и задевающее ядро (рамку снимка, сжатую на
    :data:`OBJECT_CORE_INSET_MM`). Подпись и соседний текст в ядро не входят и с объектом не сцеплены.

    Args:
        maps: Карты окна.
        core: Рамка снимка, px полосы.
        dpi: Разрешение.
        others: Рамки других снимков полосы, px: их пиксели из маски вычитаются (соседний снимок в 1 мм — не объект
            этого; 1971/01 IMG_0004_1L, два снимка друг над другом).

    Returns:
        Маска bool окна.
    """
    ox, oy = maps.origin
    tone = maps.fine if maps.fine is not None else maps.tone
    dark = tone < maps.paper - max(maps.plate_delta, OBJECT_TONE_DELTA)
    # Гладкий светлый тон без растра (тень у края листа, сор) — не объект; краска, растр и растушёвка — объект.
    dark &= (tone < maps.paper - OBJECT_SMOOTH_DELTA) | (maps.texture > maps.light_texture)
    dark = drop_text_and_edge(dark.astype(np.uint8), maps, core, dpi)
    k = max(1, int(round(mm(OBJECT_CLOSE_MM, dpi))))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    gap = int(round(mm(OBJECT_CORE_INSET_MM, dpi)))
    for b in others:
        # Соседний снимок вычитается с полем: связь через кромку между снимками тоже рвётся.
        bx0, by0 = int(max(b[0] - ox - gap, 0)), int(max(b[1] - oy - gap, 0))
        bx1, by1 = int(max(b[2] - ox + gap, 0)), int(max(b[3] - oy + gap, 0))
        dark[by0:by1, bx0:bx1] = 0
    n, labels = cv2.connectedComponents(dark, connectivity=8)
    inset = mm(OBJECT_CORE_INSET_MM, dpi)
    h, w = dark.shape
    x0, y0 = int(max(core[0] - ox + inset, 0)), int(max(core[1] - oy + inset, 0))
    x1, y1 = int(min(core[2] - ox - inset, w)), int(min(core[3] - oy - inset, h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros_like(dark, bool)
    touching = np.unique(labels[y0:y1, x0:x1])
    obj = np.isin(labels, touching[touching > 0])
    # Строки текста, сцепленные с объектом через сор или грязь бумаги, — не объект: буквы по сильной краске с ореолом
    # TEXT_HALO_MM вычитаются (внутри рамки снимка ничего не вычитается — там буквы «строк» сами часть снимка).
    # Буквы — по сильной и по слабой краске (просвет оборота), кандидаты — вне рамки снимка, сжатой на
    # OBJECT_CORE_INSET_MM (строка вплотную над снимком заходит в его рамку выносными, 1969/12 IMG_0141_1L).
    inset = mm(OBJECT_CORE_INSET_MM, dpi)
    inner = (core[0] + inset, core[1] + inset, core[2] - inset, core[3] - inset)
    letters = (letters_in(maps, inner, dpi) | letters_in(maps, inner, dpi, GROW_LETTER_DELTA)).astype(np.uint8)
    if letters.any():
        kh = max(1, int(round(2 * mm(TEXT_HALO_MM, dpi)))) | 1
        halo = cv2.dilate(letters, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kh))) > 0
        halo[y0:y1, x0:x1] = False
        obj &= ~halo
    return obj


def text_letters(dark: np.ndarray, maps: Maps, core, dpi: float) -> tuple[np.ndarray, np.ndarray]:
    """Буквы строк вне рамки снимка: компоненты размером с букву, у которых в ряду не меньше двух соседей того же кегля
    (:data:`TEXT_MIN_MM`).

    Args:
        dark: Маска краски окна uint8.
        maps: Карты окна.
        core: Рамка снимка, px полосы.
        dpi: Разрешение.

    Returns:
        ``(метки компонент окна, флаги «буква» по меткам)``.
    """
    from scipy.spatial import cKDTree

    ox, oy = maps.origin
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    text = np.zeros(n, bool)
    if n <= 1:
        return labels, text
    x, y, w, h = stats[1:, 0] + ox, stats[1:, 1] + oy, stats[1:, 2], stats[1:, 3]
    cand = ((h >= mm(TEXT_MIN_MM, dpi)) & (h <= mm(TEXT_MAX_MM, dpi)) & (w <= mm(TEXT_MAX_W_MM, dpi))
            & ((x + w <= core[0]) | (x >= core[2]) | (y + h <= core[1]) | (y >= core[3])))  # fmt: skip
    idx = np.flatnonzero(cand)
    if len(idx) < 3:
        return labels, text
    cy = (y + h / 2.0)[idx]
    xl, xr, hh = x[idx], (x + w)[idx], h[idx].astype(float)
    centres = np.stack([(xl + xr) / 2.0, cy], 1)
    gap = mm(TEXT_ROW_GAP_MM, dpi)
    for i, near in enumerate(cKDTree(centres).query_ball_point(centres, gap + mm(TEXT_MAX_W_MM, dpi))):
        same = [j for j in near if j != i and abs(cy[j] - cy[i]) <= 0.5 * max(hh[i], hh[j])
                and 0.5 <= hh[j] / hh[i] <= 2.0 and max(xl[j] - xr[i], xl[i] - xr[j]) <= gap]  # fmt: skip
        if len(same) >= 2:
            text[idx[i] + 1] = True
    return labels, text


def drop_text_and_edge(dark: np.ndarray, maps: Maps, core, dpi: float) -> np.ndarray:
    """Вырезает из маски краски буквы строк вне рамки снимка и полосу у края листа (см. :data:`TEXT_MIN_MM`).

    Args:
        dark: Маска краски окна uint8.
        maps: Карты окна (начало, размер полосы).
        core: Рамка снимка, px полосы.
        dpi: Разрешение.

    Returns:
        Маска без букв и края листа.
    """
    ox, oy = maps.origin
    out = dark.copy()
    if maps.page_shape is not None:
        e = int(round(mm(OBJECT_PAGE_EDGE_MM, dpi)))
        ph, pw = maps.page_shape
        h, w = out.shape
        out[: max(e - oy, 0)] = 0
        out[:, : max(e - ox, 0)] = 0
        out[max(ph - e - oy, 0) :] = 0
        out[:, max(pw - e - ox, 0) :] = 0
    labels, text = text_letters(out, maps, core, dpi)
    out[text[labels]] = 0
    return out


def object_outside(
    obj: np.ndarray, origin: tuple[int, int], box, dpi: float, margin_mm: float = None, ink: np.ndarray | None = None
) -> dict:
    """Объект за рамкой: площадь маски объекта дальше :data:`OBJECT_MARGIN_MM` за рамкой, всего и по сторонам
    (полосы за каждой стороной), мм². Поле — против кромки самого края: точки растра у края снимка выступают за
    середину ступеньки на 1–3 px (по периметру снимка 80×60 мм это ~40 мм²).

    Args:
        obj: Маска объекта окна (:func:`object_mask`).
        origin: Начало окна на полосе.
        box: Рамка, px полосы.
        dpi: Разрешение.
        margin_mm: Поле за рамкой, мм; None — :data:`OBJECT_MARGIN_MM`.
        ink: Маска заметной краски (:func:`object_ink`) для отсева бледных частей или None.

    Returns:
        ``{"top", "bottom", "left", "right", "total"}`` → мм².
    """
    obj = outside_parts(obj, origin, box, dpi, margin_mm, ink)
    m = mm(OBJECT_MARGIN_MM if margin_mm is None else margin_mm, dpi)
    box = (box[0] - m, box[1] - m, box[2] + m, box[3] + m)
    ox, oy = origin
    h, w = obj.shape
    x0, y0 = int(np.clip(round(box[0] - ox), 0, w)), int(np.clip(round(box[1] - oy), 0, h))
    x1, y1 = int(np.clip(round(box[2] - ox), 0, w)), int(np.clip(round(box[3] - oy), 0, h))
    px2 = (25.4 / dpi) ** 2
    out = {
        "top": float(obj[:y0].sum()) * px2,
        "bottom": float(obj[y1:].sum()) * px2,
        "left": float(obj[y0:y1, :x0].sum()) * px2,
        "right": float(obj[y0:y1, x1:].sum()) * px2,
    }
    out["total"] = sum(out.values())
    return {k: round(v, 2) for k, v in out.items()}


def object_ink(maps: Maps) -> np.ndarray:
    """Заметная краска окна: тон (слабо размытый) темнее бумаги на :data:`OBJECT_SMOOTH_DELTA`."""
    tone = maps.fine if maps.fine is not None else maps.tone
    return tone < maps.paper - OBJECT_SMOOTH_DELTA


def outside_parts(
    obj: np.ndarray, origin: tuple[int, int], box, dpi: float, margin_mm: float = None, ink: np.ndarray | None = None
) -> np.ndarray:
    """Части объекта за рамкой (с полем :data:`OBJECT_MARGIN_MM`) площадью от :data:`OBJECT_PART_MM2` — стрела, люлька,
    облако растушёвки; крошки сора и обрывки просвета у края снимка (1972/07 IMG_0020_1L, 1969/12 IMG_0141_1L — пятна
    по 0.5–2 мм², тянувшие рамку на 2–5 мм) не в счёт. Внутри рамки маска сохраняется как есть.

    Args:
        obj: Маска объекта окна.
        origin: Начало окна на полосе.
        box: Рамка, px полосы.
        dpi: Разрешение.
        margin_mm: Поле за рамкой, мм; None — :data:`OBJECT_MARGIN_MM`.
        ink: Маска заметной краски окна (темнее бумаги на :data:`OBJECT_SMOOTH_DELTA`) или None; часть за рамкой, в
            которой такой краски меньше :data:`OBJECT_PART_INK`, — просвет оборота или сор (1972/07 IMG_0020_1L,
            IMG_0011_1L), а не стрела или растушёвка.

    Returns:
        Маска объекта без мелких и бледных частей за рамкой.
    """
    m = mm(OBJECT_MARGIN_MM if margin_mm is None else margin_mm, dpi)
    ox, oy = origin
    h, w = obj.shape
    x0, y0 = int(np.clip(round(box[0] - m - ox), 0, w)), int(np.clip(round(box[1] - m - oy), 0, h))
    x1, y1 = int(np.clip(round(box[2] + m - ox), 0, w)), int(np.clip(round(box[3] + m - oy), 0, h))
    out = obj.copy()
    out[y0:y1, x0:x1] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] * (25.4 / dpi) ** 2 < OBJECT_PART_MM2
    # Кайма вдоль стороны (полоса тени или грязи шириной до OBJECT_FRINGE_MM, прижатая к снимку по всей длине — 1974/03
    # IMG_0140_1L, 1975/02 IMG_0060_1L: 0.9–1.3 мм × 22 мм слева) — не объект: стрела и люлька уходят от снимка дальше.
    fringe = mm(OBJECT_FRINGE_MM, dpi)
    px, py, pw, ph = (stats[1:, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    reach = np.maximum.reduce([x0 - px, (px + pw) - x1, y0 - py, (py + ph) - y1])
    small[1:] |= reach <= fringe
    if ink is not None and n > 1:
        share = np.bincount(labels.ravel(), weights=ink.ravel().astype(float), minlength=n) / np.maximum(
            np.bincount(labels.ravel(), minlength=n), 1
        )
        small[1:] |= share[1:] < OBJECT_PART_INK
    result = obj.copy()
    result[small[labels]] = False
    return result


def cover_object(photo: PhotoBound, maps: Maps, dpi: float, others: tuple = ()) -> PhotoBound:
    """Рамка, охватывающая весь объект снимка: если объекта за рамкой больше :data:`OBJECT_OUT_MM2`, рамка
    расширяется до охвата маски объекта, а четырёхугольника нет (вырезка без края, растушёвка, стрела крана за краем).

    Args:
        photo: Фото после уточнения краем.
        maps: Карты окна.
        dpi: Разрешение.
        others: Рамки других снимков полосы (:func:`object_mask`).

    Returns:
        Фото с рамкой по объекту и мерой «объект за рамкой» после расширения.
    """
    from dataclasses import replace

    obj = object_mask(maps, photo.box, dpi, others)
    ink = object_ink(maps)
    outside = object_outside(obj, maps.origin, photo.box, dpi, ink=ink)
    if outside["total"] <= OBJECT_OUT_MM2 or not obj.any():
        if photo.corners is not None or not obj.any():
            return replace(photo, outside_mm2=outside)
        # Части за рамкой отсеяны как кайма или бледные, но тёмная краска объекта может сплошь выходить за сторону
        # (низ колеса вместе со светлой тенью под машиной) — рамка накрывает и её.
        box = no_text_extension(maps, photo.box, ink_overhang(obj, ink, maps.origin, photo.box, dpi), dpi)
        box = grow_parts(maps, box, photo.sharp_sides, dpi)
        if box != photo.box:
            obj = object_mask(maps, box, dpi, others)
            outside = object_outside(obj, maps.origin, box, dpi, ink=ink)
        return replace(photo, box=box, outside_mm2=outside)
    first = outside["total"]
    ox, oy = maps.origin
    box = photo.box
    # Расширение — до неподвижной точки: с большим ядром к объекту прицепляются новые компоненты (полоска неба над
    # снимком 1975/01 IMG_0004_1L цепляется только к расширенной рамке).
    for _ in range(OBJECT_PASSES):
        ys, xs = np.nonzero(outside_parts(obj, maps.origin, box, dpi, ink=ink))
        x0, y0, x1, y1 = box
        box = (min(x0, float(xs.min() + ox)), min(y0, float(ys.min() + oy)),
               max(x1, float(xs.max() + 1 + ox)), max(y1, float(ys.max() + 1 + oy)))  # fmt: skip
        obj = object_mask(maps, box, dpi, others)
        outside = object_outside(obj, maps.origin, box, dpi, ink=ink)
        if outside["total"] <= OBJECT_OUT_MM2:
            break
    box = no_text_extension(maps, photo.box, ink_overhang(obj, ink, maps.origin, box, dpi), dpi)
    box = grow_parts(maps, box, photo.sharp_sides, dpi)
    obj = object_mask(maps, box, dpi, others)
    outside = object_outside(obj, maps.origin, box, dpi, ink=ink)
    reason = photo.quad_reason or f"объект за рамкой {first:.1f} мм²"
    return replace(photo, box=box, corners=None, quad_reason=reason, outside_mm2=outside)


def ink_overhang(obj: np.ndarray, ink: np.ndarray, origin: tuple[int, int], box, dpi: float):
    """Расширение рамки до тёмной краски объекта, сплошь выходящей за сторону: компонента краски объекта
    (:func:`object_ink` внутри маски объекта), которая лежит и в полосе :data:`INK_CROSS_MM` внутри рамки, и дальше
    :data:`OBJECT_MARGIN_MM` за ней площадью от :data:`INK_OVERHANG_MM2` и больше буквы (:data:`TEXT_MAX_W_MM` ×
    :data:`TEXT_MAX_MM`). Это продолжение снимка за рамкой (низ колеса
    крана МТС 1991/01 IMG_0615 — 0.9 мм ниже рамки; :func:`outside_parts` отсеивает такую часть как кайму и как бледную
    вместе со светлой тенью под машиной). Тень у края листа и просвет оборота тёмной краски не имеют.

    Args:
        obj: Маска объекта окна (:func:`object_mask`).
        ink: Маска заметной краски окна (:func:`object_ink`).
        origin: Начало окна на полосе.
        box: Рамка, px полосы.
        dpi: Разрешение.

    Returns:
        Рамка, накрывающая выходящую краску (или прежняя), px полосы.
    """
    ox, oy = origin
    h, w = obj.shape
    dark = (obj & ink).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return box
    # Компонента размером с букву — буква подписи, стоящая на линии рамки (подпись «Фото 6» у кромки снимка коллажа
    # 1969/11 IMG_0094_1L), а не продолжение снимка: отсев букв в маске объекта берёт только буквы целиком за рамкой.
    letter_sized = (stats[:, cv2.CC_STAT_WIDTH] <= mm(TEXT_MAX_W_MM, dpi)) & (
        stats[:, cv2.CC_STAT_HEIGHT] <= mm(TEXT_MAX_MM, dpi)
    )
    m, cross = mm(OBJECT_MARGIN_MM, dpi), mm(INK_CROSS_MM, dpi)
    x0, y0 = int(np.clip(round(box[0] - ox), 0, w)), int(np.clip(round(box[1] - oy), 0, h))
    x1, y1 = int(np.clip(round(box[2] - ox), 0, w)), int(np.clip(round(box[3] - oy), 0, h))
    # Полоса внутри рамки у сторон: компонента краски, задевающая её, — часть снимка у самого края.
    band = np.zeros((h, w), bool)
    band[y0:y1, x0:x1] = True
    c = int(round(cross))
    band[min(y0 + c, y1) : max(y1 - c, y0), min(x0 + c, x1) : max(x1 - c, x0)] = False
    inside_labels = np.unique(labels[band & (labels > 0)])
    # За рамкой с полем.
    far = np.ones((h, w), bool)
    mx0, my0 = int(np.clip(round(box[0] - m - ox), 0, w)), int(np.clip(round(box[1] - m - oy), 0, h))
    mx1, my1 = int(np.clip(round(box[2] + m - ox), 0, w)), int(np.clip(round(box[3] + m - oy), 0, h))
    far[my0:my1, mx0:mx1] = False
    px2 = (25.4 / dpi) ** 2
    out = list(box)
    for lab in inside_labels:
        if letter_sized[lab]:
            continue
        ys, xs = np.nonzero((labels == lab) & far)
        if len(ys) * px2 < INK_OVERHANG_MM2:
            continue
        # Рамка — до выходящих пикселей этой компоненты.
        out = [min(out[0], float(xs.min() + ox)), min(out[1], float(ys.min() + oy)),
               max(out[2], float(xs.max() + 1 + ox)), max(out[3], float(ys.max() + 1 + oy))]  # fmt: skip
    return tuple(out)


def no_text_extension(maps: Maps, old, new, dpi: float):
    """Возврат стороны рамки, расширенной по объекту, если в полосе расширения лежат строки текста (не меньше
    :data:`TEXT_GUARD_LETTERS` букв, :func:`text_letters` по сильной краске): объект сцепился с подписью через сор или
    грязь бумаги (1975/10 IMG_0014_1L — серое пятно под снимком, накрывшее подпись). Стрела крана или растушёвка букв
    строк в полосе не имеет и расширения не теряет.

    Args:
        maps: Карты окна.
        old: Рамка до расширения, px.
        new: Рамка после расширения, px.
        dpi: Разрешение.

    Returns:
        Рамка с возвращёнными сторонами.
    """
    tone = maps.fine if maps.fine is not None else maps.tone
    ink = (tone < maps.paper - TEXT_INK_DELTA).astype(np.uint8)
    labels, text = text_letters(ink, maps, old, dpi)
    ox, oy = maps.origin
    letters = text[labels]
    out = list(new)
    bands = {
        0: (new[0], old[0], old[1], old[3], True),
        2: (old[2], new[2], old[1], old[3], True),
        1: (new[1], old[1], new[0], new[2], False),
        3: (old[3], new[3], new[0], new[2], False),
    }
    for k, (a0, a1, b0, b1, vertical) in bands.items():
        if a1 - a0 < 1:
            continue
        if vertical:
            band = letters[int(max(b0 - oy, 0)) : int(max(b1 - oy, 0)), int(max(a0 - ox, 0)) : int(max(a1 - ox, 0))]
        else:
            band = letters[int(max(a0 - oy, 0)) : int(max(a1 - oy, 0)), int(max(b0 - ox, 0)) : int(max(b1 - ox, 0))]
        if band.size:
            count = cv2.connectedComponents(band.astype(np.uint8), connectivity=8)[0] - 1
            if count >= TEXT_GUARD_LETTERS:
                out[k] = old[k]
    return tuple(out)


def find_photos(gray: np.ndarray, seeds: list[tuple[int, int, int, int]], dpi: float) -> list[PhotoBound]:
    """Фото полосы по затравкам (рамки растрового детектора): маска в окне, компоненты, группы, уточнение каждого.

    Затравка без годной компоненты (штрих, пустая рамка) остаётся как есть: рамка = затравка, сторон нет
    (класс ``SOFT``) — чтобы прежняя находка не терялась.

    Args:
        gray: Серая полоса.
        seeds: Затравки ``(x0, y0, x1, y1)``, px полосы.
        dpi: Разрешение полосы.

    Returns:
        Фото по всем затравкам.
    """
    pad = int(round(mm(SEARCH_MM, dpi)))
    # Первый проход — снимки каждой затравки; второй — охват объекта с вычетом всех прочих снимков полосы.
    found = []
    for seed in seeds:
        window = (seed[0] - pad, seed[1] - pad, seed[2] + pad, seed[3] + pad)
        maps = page_maps(gray, window, dpi, seed)
        mask = photo_mask(maps, dpi)
        groups = group_components(photo_components(mask, maps, seed, dpi), dpi)
        if not groups:
            found.append((None, PhotoBound(tuple(float(v) for v in seed), seed, seed, (), None, dpi, (),
                                           "нет снимка в затравке")))  # fmt: skip
            continue
        light = light_map(maps)
        for box in groups:
            found.append((maps, refine_photo(maps, grow_box(light, maps, box, dpi), seed, dpi, len(groups) == 1)))
    out = []
    for i, (maps, photo) in enumerate(found):
        others = tuple(p.box for j, (_, p) in enumerate(found) if j != i)
        out.append(photo if maps is None else cover_object(photo, maps, dpi, others))
    return out
