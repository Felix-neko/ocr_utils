"""Восстановление контраста текста в зоне, закрытой матовым пятном.

ЗАДАЧА. На отдельных кадрах часть страницы закрыта матовым пятном: текст под ним
читается, но выглядит бледно-серым (иногда с синевой), тогда как рядом, на той же
странице, тот же набор чёрный. Нужно поднять контраст ТОЛЬКО в этой зоне и не тронуть
остальной кадр.

ФИЗИКА, ИЗ КОТОРОЙ СЛЕДУЕТ ВЕСЬ АЛГОРИТМ. Матовое пятно ведёт себя как вуаль:
наблюдаемая яркость ``I = a*I_ист + b`` (часть света поглощается, часть рассеивается
обратно). Пересчитаем это в ГЛУБИНУ КРАСКИ ``D = 1 - I/bg``, где ``bg`` — гладкий
уровень бумаги (``ocr_utils.paper``). Уровень бумаги под вуалью тоже равен
``bg = a*bg_ист + b``, поэтому

    D = (bg - I)/bg = a*(bg_ист - I_ист)/bg = k * D_ист,   k = a*bg_ист/bg < 1.

То есть в координатах глубины краски вуаль — это ПРОСТО МНОЖИТЕЛЬ, одинаковый для всех
штрихов под пятном. Значит, восстановление — умножение ``D`` на ``1/k``, а ``1/k``
измеряется прямо по кадру: сравнением глубины краски под пятном с глубиной такого же
набора рядом. Никакого подбора «на глаз» и никакой генеративной дорисовки: возвращается
ровно та амплитуда, которая была бы без вуали.

ПОЧЕМУ ПОКАНАЛЬНО. Вуаль рассеивает синий сильнее красного, поэтому текст под пятном
уходит в синеву. Если считать усиление по яркости и применять его ко всем каналам
одинаково, синева усилится вместе с текстом. Поэтому ``k`` меряется и снимается в
каждом канале СВОИМ: каждый канал возвращается к своей же норме, и цветовой сдвиг
уходит сам собой, без отдельной коррекции баланса.

ТРИ ЭТАПА И ЗАЧЕМ ОНИ РАЗДЕЛЕНЫ:

1. ГРАНИЦЫ ЗОНЫ (:func:`find_zone`) — где вообще разрешено править.
2. АМПЛИТУДА (:func:`amplify`) — насколько поднять глубину краски в каждой точке.
3. РАЗМЫТИЕ (:func:`sharpen_veil`) — снять рассеяние, которое усилением не лечится.

Второй и третий этапы разделены не для порядка, а потому что вуаль повреждает набор
ДВУМЯ разными способами: гасит контраст (это множитель, снимается усилением) и
рассеивает свет (это свёртка, снимается деконволюцией). После одного лишь усиления
сердцевины штрихов чернеют, но остаётся широкая серая кайма, и текст всё ещё читается
бледнее соседнего — замер по доле глубокой краски в :func:`sharpen_veil`.

Маска НЕ задаёт величину правки, она задаёт только область, где правка вообще
разрешена: усиление считается по локальной глубине краски и на здоровом наборе само
равно единице. Поэтому маска может быть щедрой (выпуклая оболочка, растушёвка) — лишний
захват здорового текста ничего не портит. Строгой маска обязана быть в другом: за
пределами листа (фон стола, пальцы, тень у корешка) оценка «нормы» бессмысленна, и
усиление там разнесло бы шум. Отсюда и порядок отбора тайлов ниже.

РАЗМЕРЫ — В ПИКСЕЛЯХ И ЗАМЕРЕНЫ ПО КАДРУ, а не выведены долей от него: толщина штриха
~5 px, шаг строк ~72 px (кадр 5692x4362, разворот «Плановое хозяйство» 1929). Для
другого пака их надо перемерить и передать опциями, а не полагаться на умолчания.
"""

import json
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image

from ocr_utils.paper import disk, paper_level

Image.MAX_IMAGE_PIXELS = None

# --- Размеры, замеренные на IMG_0056.jpg (5692x4362) ----------------------------------
# Раздутие светлого при оценке бумаги — пара толщин штриха, чтобы буквы ушли из оценки
# целиком. Окно размытия — крупнее буквы, но мельче неровности света: ~1.5 шага строк.
PAPER_DILATE_PX = 12
PAPER_BLUR_PX = 120

# Сетка тайлов. Сторона тайла — примерно полтора шага строк: в тайл гарантированно
# попадает хотя бы одна строка набора, иначе «глубина краски в тайле» меряется по
# межстрочному пробелу. Шаг вчетверо мельче стороны — чтобы граница пятна получилась
# не ступенчатой.
TILE_PX = 96
STRIDE_PX = 24

# Порог «здесь есть краска» и минимальная доля таких пикселей в тайле. Замерено: на
# чистой бумаге 99-й перцентиль D = 0.04, так что 0.15 — заведомо краска, а не шум.
# Доля мелкая нарочно: в таблице строка цифр занимает единицы процентов тайла, и при
# щедром пороге покрытия строки «112,8» и «158,9» выпадали из замера нормы вовсе.
INK_D = 0.15
COVER_MIN = 0.02

# Уровень бумаги ниже этого — не лист. Замерено на кадре: бумага 247, палец 200,
# фон стола 153. Порог отсекает и фон, и пальцы, не задевая лист.
PAPER_MIN = 215.0

# Насколько бумага может быть темнее окрестной, чтобы её всё ещё считать ровной. Тень у
# корешка и на изгибе — не краска, но по глубине выглядит как бледный штрих (D = 0.25 на
# линии корешка этого кадра), и усиление вгоняло её в чёрно-бурую полосу. Замеры провала
# относительно локального максимума в окне ~500 px: ровная бумага 4.9, зона под пятном
# 1.7-7.8, корешок 28.2, палец 64.1. Порог посередине широкого зазора.
PAPER_DIP = 12.0
PAPER_PEAK_TILES = 21

# Глубина краски здорового набора и порог «бледно». Замерено по кадру: у 70% текстовых
# тайлов 99-й перцентиль D выше 0.95, у бледных — 0.25-0.60. Провал между модами широк,
# порог посередине.
FAINT_D = 0.85

# Шум бумаги в единицах D (99-й перцентиль на чистом межколоннике — 0.042). Всё, что
# ниже, усиливать нельзя: это зерно JPEG и фактура бумаги, а не краска. См. :func:`ink_gate`.
NOISE_D = 0.05

# Куда тянуть глубину краски под пятном. Медиана здорового набора — 0.996, то есть
# «в чёрное»; берём чуть ниже, чтобы не выбивать восстановленные штрихи в клиппинг.
TARGET_D = 0.92

# Потолок усиления. Самая бледная измеренная глубина в пятне — 0.15-0.20 (единица «112,8»),
# и чтобы вытянуть её до цели, нужно около шести. Потолок сам по себе безопасен: шум
# бумаги гасится воротами (:func:`ink_gate`), и при шестикратном усилении зерно на чистой
# бумаге остаётся ниже порога различимости — проверено на пустых участках внутри зоны.
GAIN_MAX = 6.0


def tile_stats(values: np.ndarray, tile_px: int, stride_px: int, reducer) -> np.ndarray:
    """Свёртка кадра в сетку тайлов заданной функцией.

    Args:
        values: Кадр (H, W), float32.
        tile_px: Сторона тайла в пикселях.
        stride_px: Шаг сетки в пикселях.
        reducer: Функция, сворачивающая тайл в число.

    Returns:
        Массив (ny, nx) со значением на тайл.
    """
    height, width = values.shape
    ny = (height - tile_px) // stride_px + 1
    nx = (width - tile_px) // stride_px + 1
    out = np.zeros((ny, nx), np.float32)
    for j in range(ny):
        band = values[j * stride_px : j * stride_px + tile_px]
        for i in range(nx):
            out[j, i] = reducer(band[:, i * stride_px : i * stride_px + tile_px])
    return out


def ink_body(depth: np.ndarray) -> np.ndarray:
    """Глубина ТЕЛА штриха: медиана 5x5 по глубине краски.

    Норма меряется перцентилем, а одиночная чёрная крапина (соринка, точка типографского
    брака) перцентиль уводит. Замер на строке «112,8»: 99-й перцентиль по тайлу дал
    норму 0.77 при реальной глубине цифр 0.35, то есть усиление обнулилось ровно там,
    где было нужнее всего. Медиана 5x5 крапину убирает, а тело штриха (толщина ~5 px)
    переживает.

    Args:
        depth: Глубина краски.

    Returns:
        Глубина краски без одиночных тёмных точек.
    """
    return cv2.medianBlur(depth, 5)


def local_norm(depth: np.ndarray, tile_px: int, stride_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Локальная норма глубины краски по тайлам и доля краски в тайле.

    Норма — 90-й перцентиль глубины ПО ПИКСЕЛЯМ КРАСКИ, а не по всем пикселям тайла.
    Разница принципиальная: доля краски в тайле гуляет от 2% (строка цифр в таблице) до
    20% (плотный абзац), и один и тот же перцентиль по всему тайлу означает в этих двух
    случаях совершенно разные вещи — в первом он попадает на бумагу, во втором на штрих.
    По пикселям краски величина сравнима везде: это «насколько тёмен здешний штрих».

    Args:
        depth: Глубина краски (уже без крапин, см. :func:`ink_body`).
        tile_px: Сторона тайла.
        stride_px: Шаг сетки.

    Returns:
        Пара (норма по тайлам, доля краски по тайлам). Норма равна нулю там, где краски
        для замера не хватило.
    """
    min_pixels = COVER_MIN * tile_px * tile_px

    def reducer(tile: np.ndarray) -> float:
        ink = tile[tile > INK_D]
        return float(np.percentile(ink, 90)) if ink.size >= min_pixels else 0.0

    norm = tile_stats(depth, tile_px, stride_px, reducer)
    cover = tile_stats(depth, tile_px, stride_px, lambda t: float((t > INK_D).mean()))
    return norm, cover


def veil_field(norm: np.ndarray, cover: np.ndarray, erode_tiles: int) -> np.ndarray:
    """Гладкое поле «во сколько раз здесь придавлена краска».

    Два шага, и оба нужны.

    ЗАПОЛНЕНИЕ. Норма определена только там, где есть краска; в межстрочных пробелах и
    на пустых участках зоны её нет, а усиление нужно и там — иначе на пробелах оно
    провалится к единице и текст пойдёт полосами. Дырки заращиваются от ближайших
    тайлов с краской.

    МИНИМУМ ПО ОКРЕСТНОСТИ. Норма тайла — оценка СМЕЩЁННАЯ ВВЕРХ и только вверх: если в
    окно попал здоровый набор рядом с пятном, норма подскакивает, а вниз её сдвинуть
    нечему (крапины уже убраны медианой). Поэтому по окрестности берётся минимум, а не
    среднее: вуаль меняется плавно, и самая тёмная оценка вокруг — самая честная. Замер
    на строке «112,8», где рядом стоит чёрная шапка таблицы: усреднение давало норму
    0.52 и усиление 1.9, минимум даёт 0.26 и усиление 3.6 — то, что и требуется.

    Args:
        norm: Норма по тайлам, ноль там, где не измерена.
        cover: Доля краски по тайлам.
        erode_tiles: Радиус минимума по окрестности, тайлы.

    Returns:
        Поле нормы по тайлам, определённое везде и гладкое.
    """
    known = (cover > COVER_MIN).astype(np.uint8)
    filled = norm.copy()
    square = np.ones((3, 3), np.uint8)
    for _ in range(64):
        if known.all():
            break
        grown = cv2.dilate(known, square)
        spread = cv2.dilate(filled, square)
        filled = np.where(grown > known, spread, filled)
        known = grown
    filled = cv2.erode(filled, disk(erode_tiles))
    # Сглаживание — всего полтора тайла: окна тайлов и так перекрываются вчетверо.
    return cv2.GaussianBlur(filled, (0, 0), 1.2)


def ink_depth(channel: np.ndarray, dilate_px: int, blur_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Уровень бумаги и глубина краски канала.

    Args:
        channel: Канал изображения, uint8.
        dilate_px: Радиус раздутия светлого при оценке бумаги.
        blur_px: Сторона окна размытия при оценке бумаги.

    Returns:
        Пара (уровень бумаги, глубина краски ``D = 1 - I/bg`` в [0, 1]).
    """
    background = np.maximum(paper_level(channel, dilate_px, blur_px), 1.0)
    depth = np.clip(1.0 - channel.astype(np.float32) / background, 0.0, 1.0)
    return background, depth


def find_zone(gray: np.ndarray, tile_px: int, stride_px: int, faint_d: float) -> tuple[np.ndarray, dict]:
    """Границы зоны, накрытой матовым пятном.

    Ищем не «светлое место» (светлых мест на кадре полно — поля, межколонник, фон), а
    место, где НАБОР ЕСТЬ, А КРАСКА МЕЛКАЯ. Отсюда три условия на тайл, и каждое из них
    отсекает свой класс ложных срабатываний:

    * уровень бумаги выше порога — отсекает всё, что не лист: фон стола, пальцы, тень
      у корешка. Там «норма глубины краски» не определена, и усиливать нечего;
    * доля краски выше порога — отсекает пустые поля и межстрочные пробелы: у чистой
      бумаги глубина краски мелкая по той же причине, что у бледного текста, и без
      этого условия маска расползлась бы по всем полям;
    * глубина краски ниже порога — собственно бледность.

    Разрозненные тайлы собираются в ОДНО пятно двумя шагами. Сначала близкие компоненты
    склеиваются раздутием: таблица и абзац под одним и тем же пятном разделены пустой
    полосой (заголовок, межабзацный пробел) и связными компонентами не соединяются, хотя
    физически это одно повреждение. Потом от полученной группы берётся выпуклая оболочка
    — модель пятна как связного мазка, а не как россыпи строк; заодно в зону попадают
    пробелы между строками, и поле усиления внутри пятна выходит непрерывным.

    Args:
        gray: Полутоновый кадр, uint8.
        tile_px: Сторона тайла.
        stride_px: Шаг сетки.
        faint_d: Порог бледности по глубине краски.

    Returns:
        Пара (маска зоны в тайлах, словарь замеров для отчёта).
    """
    background, depth = ink_depth(gray, PAPER_DILATE_PX, PAPER_BLUR_PX)
    tile_depth, tile_cover = local_norm(ink_body(depth), tile_px, stride_px)
    tile_paper = tile_stats(background, tile_px, stride_px, np.median)

    # «Ровная бумага»: не только светлая сама по себе, но и не проваленная относительно
    # окрестной. Второе условие снимает тени — у корешка и на изгибе листа, — которые по
    # глубине неотличимы от бледного штриха и которые усиление превращало в тёмную полосу.
    dip = cv2.dilate(tile_paper, disk(PAPER_PEAK_TILES)) - tile_paper
    on_sheet = (tile_paper > PAPER_MIN) & (dip < PAPER_DIP)
    has_ink = tile_cover > COVER_MIN
    faint = (on_sheet & has_ink & (tile_depth < faint_d)).astype(np.uint8)

    # Замыкание на пару тайлов убирает дыры от межстрочных пробелов, открытие — одиночные
    # тайлы на краю набора, где в окно попала половина буквы.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    faint = cv2.morphologyEx(faint, cv2.MORPH_CLOSE, kernel, iterations=2)
    faint = cv2.morphologyEx(faint, cv2.MORPH_OPEN, kernel, iterations=1)

    # Склейка близких компонент: радиус раздутия ~350 px кадра — больше межабзацного
    # разрыва, но меньше расстояния до противоположного края листа.
    glue_tiles = max(1, int(round(350 / stride_px)))
    glued = cv2.dilate(faint, disk(glue_tiles))
    count, labels = cv2.connectedComponents(glued, 8)

    # Из групп выбираем самую «бледную по массе», а не самую большую по площади: масса
    # учитывает и размер, и глубину провала, поэтому крупная еле-еле бледная область на
    # изгибе страницы не перебивает настоящее пятно.
    deficit = np.clip(faint_d - tile_depth, 0.0, None) * faint
    best, best_mass = 0, -1.0
    for label in range(1, count):
        mass = float(deficit[labels == label].sum())
        if mass > best_mass:
            best, best_mass = label, mass
    group = (faint > 0) & (labels == best)

    points = np.column_stack(np.nonzero(group))[:, ::-1].astype(np.int32)
    zone = np.zeros_like(faint)
    cv2.fillConvexPoly(zone, cv2.convexHull(points), 1)
    # Оболочка выпуклая и потому перехлёстывает через корешок на соседнюю полосу.
    # Пересечение с ровной бумагой возвращает её в границы листа.
    zone &= on_sheet.astype(np.uint8)

    ys, xs = np.nonzero(zone)
    report = {
        "тайлов_на_листе": int(on_sheet.sum()),
        "тайлов_с_краской": int(has_ink.sum()),
        "тайлов_бледных": int(faint.sum()),
        "тайлов_в_группе": int(group.sum()),
        "тайлов_в_оболочке": int(zone.sum()),
        "масса_бледности": round(best_mass, 1),
        "bbox_px": [
            int(xs.min() * stride_px),
            int(ys.min() * stride_px),
            int((xs.max() + 1) * stride_px + tile_px),
            int((ys.max() + 1) * stride_px + tile_px),
        ],
        "глубина_в_зоне_p10_p50": [
            round(float(np.percentile(tile_depth[group], 10)), 3),
            round(float(np.percentile(tile_depth[group], 50)), 3),
        ],
        "глубина_здорового_набора_p50": round(
            float(np.percentile(tile_depth[has_ink & on_sheet & (tile_depth >= faint_d)], 50)), 3
        ),
    }
    return zone, report


def upscale(tile_map: np.ndarray, shape: tuple[int, int], tile_px: int, stride_px: int) -> np.ndarray:
    """Разворачивает карту тайлов обратно в кадр.

    Тайл отвечает за свой ЦЕНТР, поэтому карта сначала растягивается билинейно по шагу
    сетки, а потом сдвигается на полтайла: иначе поле усиления уезжает влево-вверх на
    полтайла и на границе пятна получается ступенька.

    Args:
        tile_map: Карта по тайлам, float32.
        shape: Форма кадра (H, W).
        tile_px: Сторона тайла.
        stride_px: Шаг сетки.

    Returns:
        Кадр (H, W), float32.
    """
    ny, nx = tile_map.shape
    big = cv2.resize(tile_map, (nx * stride_px, ny * stride_px), interpolation=cv2.INTER_LINEAR)
    pad = tile_px // 2
    big = cv2.copyMakeBorder(big, pad, shape[0], pad, shape[1], cv2.BORDER_REPLICATE)
    return big[: shape[0], : shape[1]]


def ink_gate(depth: np.ndarray, noise_d: float) -> np.ndarray:
    """Вес «это краска, а не шум бумаги»: ``D^4 / (D^4 + (2*noise)^4)``.

    Усиление в пять-шесть раз обязано не трогать чистую бумагу, иначе внутри зоны
    вылезает и зерно, и — что заметнее — ЦВЕТНОЕ пятно: каналы усиливаются по-своему, и
    разница в пару уровней на бумаге читается как жёлтый налёт. Первая версия гасила шум
    мягкой формулой ``D^2/(D+noise)``, и этого не хватило: на бумаге с ``D = 0.03`` она
    давала на выходе 0.056, то есть почти двукратный подъём, и цветной след был виден.

    Четвёртая степень даёт резкий, но гладкий переход: половина веса приходится на
    удвоенный шум бумаги, ниже вес падает почти отвесно, выше — почти единица. Замеры на
    этом кадре (шум бумаги 0.05): при ``D = 0.03`` вес 0.02 — бумага меняется меньше чем
    на уровень яркости; при ``D = 0.12`` (край штриха) вес 0.67; при ``D = 0.30`` (бледный
    штрих под пятном) вес 0.995 — усиление работает в полную силу.

    Args:
        depth: Глубина краски.
        noise_d: Уровень шума бумаги в единицах глубины.

    Returns:
        Вес в [0, 1] той же формы.
    """
    quad = depth**4
    return quad / (quad + (2.0 * noise_d) ** 4)


def blend(channel: np.ndarray, background: np.ndarray, depth: np.ndarray, new_depth: np.ndarray) -> np.ndarray:
    """Собирает канал кадра по новой глубине краски.

    Кадр собирается ПРИБАВКОЙ к исходным пикселям, а не пересчётом из уровня бумаги.
    Разница не косметическая: глубина краски обрезана снизу нулём, поэтому у бликов ярче
    оценки бумаги обратный пересчёт ``bg * (1 - D)`` вернул бы не исходную яркость, а
    уровень бумаги, и поехал бы весь кадр, а не только зона (в первом прогоне так и
    вышло: 88% изменённых пикселей оказались вне зоны). С прибавкой за пределами зоны
    получается побайтно исходник, потому что там ``new_depth - depth`` равно нулю.

    Args:
        channel: Исходный канал, uint8.
        background: Уровень бумаги канала.
        depth: Исходная глубина краски.
        new_depth: Новая глубина краски, уже взвешенная маской зоны.

    Returns:
        Канал uint8.
    """
    return np.clip(channel.astype(np.float32) - background * (new_depth - depth), 0, 255).astype(np.uint8)


def amplify(
    image: np.ndarray,
    alpha: np.ndarray,
    tile_px: int,
    stride_px: int,
    target_d: float,
    gain_max: float,
    erode_tiles: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Возвращает краске в зоне её исходную ГЛУБИНУ (амплитуду).

    Усиление НЕ константа по зоне: пятно неоднородно, к краю вуаль слабее. В каждой
    точке берётся своя локальная норма (:func:`veil_field`), и усиление считается как
    «сколько не хватает до цели». На здоровом наборе норма уже равна цели, усиление
    выходит единичным, и попавший в маску здоровый текст остаётся нетронутым — это и
    позволяет держать маску щедрой.

    Усиление пропускается через ворота :func:`ink_gate`: иначе шестикратный коэффициент
    поднял бы вместе с текстом и зерно JPEG на чистой бумаге внутри пятна.

    Args:
        image: Кадр BGR, uint8.
        alpha: Растушёванная маска зоны в пикселях кадра.
        tile_px: Сторона тайла.
        stride_px: Шаг сетки.
        target_d: Целевая глубина краски.
        gain_max: Потолок усиления.
        erode_tiles: Радиус минимума при сборке поля вуали, тайлы.

    Returns:
        Тройка (кадр BGR uint8, поле усиления по яркости, словарь замеров).
    """
    height, width = image.shape[:2]
    out = np.empty_like(image)
    gains, luma_gain = {}, None
    for index, name in enumerate(("B", "G", "R")):
        channel = image[:, :, index]
        background, depth = ink_depth(channel, PAPER_DILATE_PX, PAPER_BLUR_PX)

        norm, cover = local_norm(ink_body(depth), tile_px, stride_px)
        filled = veil_field(norm, cover, erode_tiles)

        gain = np.clip(target_d / np.maximum(filled, 1e-3), 1.0, gain_max)
        gain_px = upscale(gain, (height, width), tile_px, stride_px)

        # Усиление применяется через ворота: краска умножается на gain целиком, бумага
        # не трогается вовсе. Норма меряется по пикселям краски, где ворота уже открыты,
        # поэтому gain = цель/норма попадает в цель без всякой поправки на ворота.
        boosted = np.clip(depth * (1.0 + ink_gate(depth, NOISE_D) * (gain_px - 1.0)), 0.0, 1.0)
        out[:, :, index] = blend(channel, background, depth, depth + alpha * (boosted - depth))

        luma_gain = gain_px if luma_gain is None else np.minimum(luma_gain, gain_px)
        inside = gain_px[alpha > 0.5]
        gains[name] = [round(float(np.percentile(inside, p)), 2) for p in (50, 90, 99)]

    return out, luma_gain, {"усиление_p50_p90_p99": gains}


def deconvolve(depth: np.ndarray, sigma: float, iterations: int) -> np.ndarray:
    """Ричардсон — Люси по глубине краски, ядро — гаусс.

    Считается именно по глубине краски, а не по яркости: RL требует неотрицательного
    сигнала на нулевом фоне, и ``D`` (бумага 0, штрих около 1) — ровно такой, тогда как
    яркость устроена наоборот и на ней метод разваливается.

    Args:
        depth: Глубина краски с закрытыми на бумаге воротами (см. :func:`ink_gate`).
        sigma: Сигма ядра рассеяния, пикс.
        iterations: Число итераций.

    Returns:
        Глубина краски после деконволюции, в [0, 1].
    """
    estimate = depth.copy()
    for _ in range(iterations):
        convolved = cv2.GaussianBlur(estimate, (0, 0), sigma)
        estimate = estimate * cv2.GaussianBlur(depth / (convolved + 1e-3), (0, 0), sigma)
        estimate = np.clip(estimate, 0.0, 1.0)
    return estimate


def sharpen_veil(image: np.ndarray, weight: np.ndarray, sigma: float, iterations: int) -> tuple[np.ndarray, dict]:
    """Снимает РАЗМЫТИЕ, оставшееся после подъёма амплитуды.

    ЗАЧЕМ ОТДЕЛЬНЫЙ ЭТАП. Матовое пятно не только гасит контраст, но и рассеивает свет,
    а это два разных повреждения, и усилением лечится только первое. После подъёма
    амплитуды сердцевины штрихов под пятном становятся чёрными, но у них остаётся
    широкая серая кайма, и текст по-прежнему читается бледнее соседнего. Признак,
    который это ловит, — доля ГЛУБОКОЙ краски (``D > 0.8``) среди пикселей краски:
    на здоровом наборе этой полосы 0.50-0.57, в зоне после одного лишь усиления 0.25
    (таблица) и 0.38 (абзац). Гистограмма в зоне размазана по середине — подпись
    размытия, а не недобора яркости.

    СКОЛЬКО РАЗМЫТИЯ СНИМАТЬ. Сигма подобрана по кадру, а не на глаз: здоровый участок
    размывался гауссом с разной сигмой, пока гистограмма глубины краски не совпадала с
    зонной. Минимум расхождения — 0.8-1.0 px для абзаца и 1.0-1.5 px для таблицы.
    Число итераций выбрано по тому же признаку: при sigma=1.2 и 20 итерациях доля
    глубокой краски выходит 0.52 в таблице и 0.55 в абзаце против 0.50 и 0.57 у
    здорового набора рядом. Дальше гнать нельзя — RL начинает выедать штрих в толщину.

    ПОЧЕМУ ВЕС — ПО СИЛЕ ВУАЛИ, А НЕ ПО МАСКЕ. Рассеяние и поглощение идут от одного и
    того же налёта, поэтому размытие там, где вуаль сильнее. Здоровый текст, попавший в
    щедрую выпуклую оболочку, усиления не потребовал — и деконволюции не получает тоже,
    иначе он вышел бы перешарпленным на фоне остальной полосы.

    Args:
        image: Кадр BGR, uint8.
        weight: Вес правки в пикселях кадра, [0, 1].
        sigma: Сигма ядра рассеяния, пикс.
        iterations: Число итераций RL.

    Returns:
        Пара (кадр BGR uint8, словарь замеров).
    """
    # RL — самая дорогая часть прогона (две свёртки на итерацию на канал), а нужен он
    # только внутри зоны. Считаем в окне вокруг неё с полем в PAPER_BLUR_PX, чтобы оценка
    # уровня бумаги у края окна была той же, что на полном кадре.
    rows, cols = np.nonzero(weight > 0.0)
    if rows.size == 0:
        return image, {"деконволюция_sigma_итераций": [sigma, iterations], "площадь_деконволюции_px": 0}
    y0, y1 = max(0, rows.min() - PAPER_BLUR_PX), min(image.shape[0], rows.max() + 1 + PAPER_BLUR_PX)
    x0, x1 = max(0, cols.min() - PAPER_BLUR_PX), min(image.shape[1], cols.max() + 1 + PAPER_BLUR_PX)
    window, window_weight = image[y0:y1, x0:x1], weight[y0:y1, x0:x1]

    out = image.copy()
    for index in range(3):
        channel = window[:, :, index]
        background, depth = ink_depth(channel, PAPER_DILATE_PX, PAPER_BLUR_PX)
        # RL считается по краске с закрытыми на бумаге воротами, и результат через те же
        # ворота возвращается: на бумаге правка строго нулевая. Без этого деконволюция
        # выбелила бы фактуру бумаги внутри зоны, и пятно стало бы видно как более чистый
        # прямоугольник.
        gate = ink_gate(depth, NOISE_D)
        base = depth * gate
        sharp = deconvolve(base, sigma, iterations)
        new_depth = depth + window_weight * gate * (sharp - base)
        out[y0:y1, x0:x1, index] = blend(channel, background, depth, new_depth)
    return out, {
        "деконволюция_sigma_итераций": [sigma, iterations],
        "площадь_деконволюции_px": int((weight > 0.5).sum()),
    }


def restore(
    image: np.ndarray,
    zone: np.ndarray,
    tile_px: int,
    stride_px: int,
    target_d: float,
    gain_max: float,
    erode_tiles: int,
    feather_px: int,
    passes: int,
    deblur_sigma: float,
    deblur_iters: int,
) -> tuple[np.ndarray, dict]:
    """Полный цикл правки зоны: подъём амплитуды, потом снятие размытия.

    ПОЧЕМУ ПОДЪЁМ АМПЛИТУДЫ ДЕЛАЕТСЯ НЕСКОЛЬКО РАЗ. Норма меряется по тайлу, а тайл в
    разреженной таблице ловит и здоровые соседние знаки, так что за один проход зона
    вытягивается не до конца: после первого прохода норма в столбце «СССР» осталась
    0.7-0.9 при 0.98 у здорового набора. Второй проход меряет норму уже по исправленному
    кадру и добирает остаток. Процесс сходится сам: как только норма дошла до цели,
    усиление становится единичным и следующий проход ничего не меняет.

    ГРАНИЦЫ ЗОНЫ БЕРУТСЯ ОДИН РАЗ, по исходному кадру, и на проходах не пересчитываются:
    пятно — физический дефект, оно никуда не переезжает, а вот признак бледности после
    первого прохода частично исчезает, и повторный поиск сузил бы зону до недоделанного
    остатка.

    Args:
        image: Кадр BGR, uint8.
        zone: Маска зоны в тайлах.
        tile_px: Сторона тайла.
        stride_px: Шаг сетки.
        target_d: Целевая глубина краски.
        gain_max: Потолок усиления.
        erode_tiles: Радиус минимума при сборке поля вуали, тайлы.
        feather_px: Радиус растушёвки границы зоны, пикс.
        passes: Сколько раз поднимать амплитуду.
        deblur_sigma: Сигма ядра рассеяния (0 — не снимать размытие).
        deblur_iters: Число итераций RL.

    Returns:
        Пара (восстановленный кадр BGR uint8, словарь замеров для отчёта).
    """
    height, width = image.shape[:2]

    alpha = upscale(zone.astype(np.float32), (height, width), tile_px, stride_px)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, feather_px / 3.0))
    alpha = np.clip(alpha, 0.0, 1.0)

    out, report, veil_gain = image, {"площадь_зоны_px": int((alpha > 0.5).sum())}, None
    for number in range(1, passes + 1):
        out, gain_px, step = amplify(out, alpha, tile_px, stride_px, target_d, gain_max, erode_tiles)
        report[f"проход_{number}"] = step
        if veil_gain is None:
            veil_gain = gain_px

    if deblur_sigma > 0:
        # Вес деконволюции — по усилению ПЕРВОГО прохода: именно оно измеряет вуаль на
        # неиспорченном кадре. Усиление 1.0 (здоровый набор) даёт вес 0, усиление вдвое
        # и выше — полный вес.
        weight = alpha * np.clip(veil_gain - 1.0, 0.0, 1.0)
        out, step = sharpen_veil(out, weight, deblur_sigma, deblur_iters)
        report.update(step)

    return out, report


@click.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dst", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Куда записать результат.")
@click.option("--tile-px", default=TILE_PX, show_default=True, help="Сторона тайла, пикс.")
@click.option("--stride-px", default=STRIDE_PX, show_default=True, help="Шаг сетки тайлов, пикс.")
@click.option("--faint-d", default=FAINT_D, show_default=True, help="Порог бледности по глубине краски.")
@click.option("--target-d", default=TARGET_D, show_default=True, help="Целевая глубина краски.")
@click.option("--gain-max", default=GAIN_MAX, show_default=True, help="Потолок усиления.")
@click.option("--erode-tiles", default=3, show_default=True, help="Радиус минимума поля вуали, тайлы.")
@click.option("--feather-px", default=120, show_default=True, help="Растушёвка границы зоны, пикс.")
@click.option("--passes", default=2, show_default=True, help="Сколько раз поднимать амплитуду.")
@click.option("--deblur-sigma", default=1.2, show_default=True, help="Сигма ядра рассеяния (0 — не снимать).")
@click.option("--deblur-iters", default=20, show_default=True, help="Итераций Ричардсона — Люси.")
@click.option("--quality", default=97, show_default=True, help="Качество JPEG.")
@click.option("--debug-dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="Куда класть карты.")
def main(
    src,
    dst,
    tile_px,
    stride_px,
    faint_d,
    target_d,
    gain_max,
    erode_tiles,
    feather_px,
    passes,
    deblur_sigma,
    deblur_iters,
    quality,
    debug_dir,
):
    """Поднимает контраст текста в зоне, закрытой матовым пятном."""
    with Image.open(src) as pil:
        pil = pil.convert("RGB")
        exif = pil.info.get("exif")
        dpi = pil.info.get("dpi")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    zone, zone_report = find_zone(gray, tile_px, stride_px, faint_d)
    out, fix_report = restore(
        image, zone, tile_px, stride_px, target_d, gain_max, erode_tiles, feather_px, passes, deblur_sigma, deblur_iters
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    result = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    save_kwargs = {"quality": quality, "subsampling": 0}
    if exif:
        save_kwargs["exif"] = exif
    if dpi:
        save_kwargs["dpi"] = dpi
    result.save(dst, **save_kwargs)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        alpha = upscale(zone.astype(np.float32), gray.shape, tile_px, stride_px)
        overlay = image.copy()
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] + 70 * alpha, 0, 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "zone_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])

    click.echo(json.dumps({**zone_report, **fix_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
