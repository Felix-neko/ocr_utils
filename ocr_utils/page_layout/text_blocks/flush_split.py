"""Разделение блоков по смене выключки: подпись авторов вправо и «г. Город» влево не срастаются с абзацем по формату."""

from __future__ import annotations

from enum import Enum

import numpy as np

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.blocks import Row, pitch_of, row_gap, rows_of


class FlushSplit(str, Enum):
    """Способ разделения по смене выключки (переключатель :func:`flush_change` и :func:`split_far_pieces`).

    * ``OFF`` — прежний ход: склейка ложных разрезов (``regroup.rejoin``) смотрит только на совпадение крайних
      краёв групп, ряды из далёких кусков на одной высоте не режутся;
    * ``ALIGN`` — смена стороны выключки на стыке групп сама по себе запрещает склейку;
    * ``ALIGN_GAP`` — смена стороны выключки плюс хотя бы одно подкрепление: увеличенный промежуток до соседней
      строки (``FLUSH_GAP_PITCHES`` шага строк) или смена набора (кегль или жирность, ``FLUSH_TYPE_RATIO``).

    У ``ALIGN`` и ``ALIGN_GAP`` ряд из двух далёких кусков на одной высоте (``юрисконсульт`` справа и
    ``г. Ереван`` слева) режется на два ряда, если соседняя строка дотягивается только до одного куска.
    """

    OFF = "off"
    ALIGN = "align"
    ALIGN_GAP = "align_gap"
    # Смена выключки режет блок без подкреплений, а решает её первая строка: строка, сменившая выключку с «влево / по
    # формату» на «вправо», — первая строка нового блока (подпись авторов, ХиП 1991/04 IMG_0198 L: промежуток 1.12
    # шага, кегль 1.12, жирность по медиане 1.02 — у ALIGN_GAP не дотягивали до 1.15); строка «влево» или абзац точно
    # у левого края под подписью — тоже («г. Город», следующая цитата под автором предыдущей). Внутри подписи (регалии
    # лесенкой, второй автор) разрезов нет, и разный набор склейке не мешает (:func:`signature_continues`).
    # Множество ХиП 1991 (`research/flush_split/sets`): подписи 132 → 150 из 160, ложные разрезы внутри подписи 43 → 18.
    ALIGN_RIGHT = "align_right"
    # Объединение ``ALIGN_RIGHT`` и ``ALIGN_GAP``: разрез, если его ставит любой из двух. ``ALIGN_RIGHT`` не надмножество
    # ``ALIGN_GAP`` (на множестве ХиП 1991 у 9 стыков прав только ``ALIGN_GAP``). Разрезы ``ALIGN_GAP`` внутри подписи
    # (фамилия и регалии, :func:`signature_continues`) не ставятся: это главный источник его ложных разрезов.
    ALIGN_RIGHT_GAP = "align_right_gap"


class FlushChange(str, Enum):
    """Вид смены стороны выключки на стыке двух групп."""

    TO_RIGHT = "to_right"  # абзац по формату (или влево) → строки вправо: начало подписи
    FROM_RIGHT = "from_right"  # строки вправо → абзац или строки влево: «г. Город», абзац после подписи
    # строки вправо → абзац, первая строка которого с абзацным отступом (у ALIGN_RIGHT — с подкреплением)
    FROM_RIGHT_INDENT = "from_right_indent"
    # абзац → строки на общем центре вдали от левого края: подпись по центру правой части колонки (1973/10 IMG_0173_2R)
    TO_CENTRE = "to_centre"
    # строки на общем центре → абзац или строки влево
    FROM_CENTRE = "from_centre"


class FlushSide(str, Enum):
    """Сторона, по которой выключена группа рядов внутри общего охвата двух групп."""

    LEFT = "left"  # левые края у общего левого края, правые далеко от общего правого
    RIGHT = "right"  # правые края у общего правого края, левые далеко от общего левого


# Способ по умолчанию; читается при каждом вызове, поэтому стенд может переключить его присваиванием.
DEFAULT_FLUSH_SPLIT = FlushSplit.ALIGN_RIGHT_GAP

# Край ряда «у общего края» — не дальше стольких мм (как допуск совпадения краёв у ``regroup.REJOIN_EDGE_MM``).
FLUSH_TOL_MM = 3.0
# Край ряда «далеко от общего края» — дальше стольких мм: больше абзацного отступа (4–5 мм в ХиП 1991) и
# как ``blocks.OFFSET_MM`` (подпись, отодвинутая от обеих кромок колонки).
FLUSH_OFFSET_MM = 8.0
# ...и дальше такой доли общего охвата: ряд «вправо» или «влево» короче трёх четвертей охвата (строки подписи ХиП
# 1991/01 IMG_1008 — 0.17–0.44 ширины колонки).
FLUSH_SHORT_SHARE = 0.25
# Группа «вправо» — из строк текста: медианная ширина её рядов не меньше стольких мм. Колонка номеров страниц
# оглавления (цифры по 3–5 мм у правого края, ХиП 1991/03 IMG_0057_R) — не подпись, и от строк оглавления она
# не отделяется (иначе ложилась отдельным блоком поверх блока оглавления). Подпись: «юрист» — 8 мм, фамилии — от 15.
FLUSH_MIN_WIDTH_MM = 7.0
# Группа — абзац: не меньше стольких её рядов шириной не меньше такой доли общего охвата.
FLUSH_MIN_FULL_ROWS = 2
FLUSH_FULL_SHARE = 0.8
# Группа «держится левого края», если у него начинаются не меньше такой доли её рядов.
ANCHOR_SHARE = 0.5
# Увеличенный промежуток: расстояние между осями стыковых рядов не меньше стольких шагов строк. Разброс шага
# внутри абзаца ХиП 1991/01 IMG_1008 — 21.0–23.6 px при медиане 22 (до 1.07); подпись отбита на 27.1 px (1.23).
FLUSH_GAP_PITCHES = 1.15
# Смена набора: отношение медиан высоты глифа или жирности (``Row.weight``) групп не меньше этого. У IMG_1008:
# глиф 10.0 против 8.5 px (1.18), жирность 2.68 против 1.95 (1.37).
FLUSH_TYPE_RATIO = 1.15
# Группа «на общем центре» (подпись по центру под абзацем, :func:`_centred`): не меньше стольких рядов вдали от левого
# края (как у «вправо»: дальше ``FLUSH_OFFSET_MM`` и ``FLUSH_SHORT_SHARE`` охвата), центры которых отходят от их медианы
# не дальше ``FLUSH_CENTRE_TOL_MM``. Подпись 1973/10 IMG_0173_2R: пять строк с центрами x 674–675 px при левых краях
# 543–592 и правых 756–808 (до правого края колонки 19–71 px — не «вправо»). Одна строка по центру сюда не входит: её
# не отличить от строки формулы или звёздочек.
FLUSH_CENTRE_MIN_ROWS = 2
FLUSH_CENTRE_TOL_MM = 3.0
# Разброс центров группы на общем центре — не больше такой доли меньшего из разбросов левых и правых краёв.
FLUSH_CENTRE_SPREAD_SHARE = 0.5
# Далёкие куски одного ряда: просвет между осями шире стольких мм...
FAR_GAP_MM = 15.0
# ...и соседние строки ищутся не дальше стольких высот строки по вертикали (как ``regroup.REACH_HEIGHTS``).
FAR_REACH_HEIGHTS = 2.5
# Соседняя строка — не та же самая: оси разнесены по вертикали больше чем на такую долю высоты.
FAR_SAME_ROW_HEIGHTS = 0.5
# Просвет пуст: краска текста (отточия, тире, обломки букв) — не больше чем в такой доле его столбцов в полосе ряда.
# Строки оглавления «Название ......... 28» рвутся на оси до отточия и номер за ним; отточие в просвете — краска, и
# номер страницы от своей строки не отрезается (ХиП 1991/03 IMG_0057_R: без этой проверки номера уходили в
# отдельный блок поверх блока оглавления).
FAR_GAP_INK_SHARE = 0.05


def _mode(mode: FlushSplit | None) -> FlushSplit:
    """Способ разделения: явный или модульный по умолчанию (``DEFAULT_FLUSH_SPLIT``)."""
    return DEFAULT_FLUSH_SPLIT if mode is None else FlushSplit(mode)


def row_side(row: Row, cx0: float, cx1: float, dpi: float) -> FlushSide | None:
    """Сторона выключки одного ряда в охвате ``[cx0, cx1]``.

    Args:
        row: Ряд.
        cx0, cx1: Общий охват по x (пиксели рабочей копии).
        dpi: Разрешение рабочей копии.

    Returns:
        ``LEFT`` — ряд стоит у левого края и не доходит до правого дальше ``FLUSH_OFFSET_MM`` и дальше
        ``FLUSH_SHORT_SHARE`` охвата; ``RIGHT`` — наоборот; ``None`` — ряд во всю ширину, по центру или в середине.
    """
    # Отступ — не меньше доли охвата: первая строка реплики диалога («— Нельзя ли вкратце…», ХиП 1991/01 IMG_0992,
    # отступ с тире 8.5 мм) идёт до правого края, и по одному отступу в мм она выглядела строкой «вправо».
    tol = mm_to_px(FLUSH_TOL_MM, dpi)
    offset = max(mm_to_px(FLUSH_OFFSET_MM, dpi), FLUSH_SHORT_SHARE * (cx1 - cx0))
    if row.x0 - cx0 <= tol and cx1 - row.x1 > offset:
        return FlushSide.LEFT
    if cx1 - row.x1 <= tol and row.x0 - cx0 > offset:
        return FlushSide.RIGHT
    return None


def _detached(row: Row, cx0: float, cx1: float, dpi: float) -> bool:
    """Стоит ли ряд вдали от левого края охвата: дальше ``FLUSH_OFFSET_MM`` и ``FLUSH_SHORT_SHARE`` охвата (как у «вправо»).

    Args:
        row: Ряд.
        cx0, cx1: Общий охват по x.
        dpi: Разрешение рабочей копии.

    Returns:
        ``True`` — ряд отодвинут от левого края дальше абзацного отступа и висячей метки.
    """
    return row.x0 - cx0 > max(mm_to_px(FLUSH_OFFSET_MM, dpi), FLUSH_SHORT_SHARE * (cx1 - cx0))


def _detached_run(rows: list[Row], cx0: float, cx1: float, dpi: float) -> list[Row]:
    """Начальная серия рядов ``rows``, стоящих вдали от левого края (:func:`_detached`)."""
    run: list[Row] = []
    for row in rows:
        if not _detached(row, cx0, cx1, dpi):
            break
        run.append(row)
    return run


def _centred(rows: list[Row], cx0: float, cx1: float, dpi: float) -> bool:
    """Группа строк на общем центре вдали от левого края охвата — подпись или заголовок по центру своей оси.

    Args:
        rows: Ряды группы (серия сверху вниз).
        cx0, cx1: Общий охват по x.
        dpi: Разрешение рабочей копии.

    Returns:
        ``True`` — не меньше ``FLUSH_CENTRE_MIN_ROWS`` рядов, все вдали от левого края (:func:`_detached`), это строки
        текста (:func:`_text_wide`), центры отходят от медианы не дальше ``FLUSH_CENTRE_TOL_MM``, а края рваные с обеих
        сторон и разбегаются вдвое сильнее центров (``FLUSH_CENTRE_SPREAD_SHARE``).
    """
    if len(rows) < FLUSH_CENTRE_MIN_ROWS or not _text_wide(rows, dpi):
        return False
    if not all(_detached(row, cx0, cx1, dpi) for row in rows):
        return False
    tol = mm_to_px(FLUSH_CENTRE_TOL_MM, dpi)
    centres = np.array([(row.x0 + row.x1) / 2.0 for row in rows])
    if not np.all(np.abs(centres - np.median(centres)) <= tol):
        return False
    # Центровка, а не выключка к краю: и левые, и правые края рваные (разброс не меньше допуска), а центры разбегаются
    # меньше половины разброса краёв. У подписи вправо правые края совпадают, и две её строки близкой длины иначе
    # сходили за строки на общем центре (регалии лесенкой, ХиП 1991/01 IMG_0982_L). У 1973/10 IMG_0173_2R: края
    # разбегаются на 49 и 52 px, центры — на 1 px.
    spread = min(float(np.ptp([row.x0 for row in rows])), float(np.ptp([row.x1 for row in rows])))
    return spread >= tol and float(np.ptp(centres)) <= FLUSH_CENTRE_SPREAD_SHARE * spread


def group_side(rows: list[Row], cx0: float, cx1: float, dpi: float) -> FlushSide | None:
    """Сторона выключки группы: у ВСЕХ её рядов одна и та же сторона (:func:`row_side`), иначе ``None``.

    Args:
        rows: Ряды группы.
        cx0, cx1: Общий охват по x двух сравниваемых групп.
        dpi: Разрешение рабочей копии.

    Returns:
        Общая сторона рядов или ``None`` (абзац по формату, смешанные края, центр).
    """
    sides = {row_side(row, cx0, cx1, dpi) for row in rows}
    return next(iter(sides)) if len(sides) == 1 else None


def _starts_left(row: Row, cx0: float, dpi: float) -> bool:
    """Начинается ли ряд у левого края охвата (с абзацным отступом включительно): не дальше ``FLUSH_OFFSET_MM``."""
    return row.x0 - cx0 <= mm_to_px(FLUSH_OFFSET_MM, dpi)


def left_anchored(rows: list[Row], cx0: float, dpi: float) -> bool:
    """Держится ли группа левого края охвата: не меньше ``ANCHOR_SHARE`` её рядов начинаются у него (:func:`_starts_left`).

    Абзац по формату и «г. Ереван» держатся левого края; подпись авторов вправо — нет, хотя её самый длинный ряд
    («председатель производственного») и задаёт левый край охвата, если сравнивать подпись саму с собой
    (фамилия над регалиями). Без этой меры фамилия «М. АКОПЯН» отрывалась от своих регалий.

    Args:
        rows: Ряды группы.
        cx0: Левый край общего охвата.
        dpi: Разрешение рабочей копии.

    Returns:
        ``True`` — группа выключена влево или по формату.
    """
    return float(np.mean([_starts_left(row, cx0, dpi) for row in rows])) >= ANCHOR_SHARE


def _text_wide(rows: list[Row], dpi: float) -> bool:
    """Строки ли текста в группе: медианная ширина рядов не меньше ``FLUSH_MIN_WIDTH_MM`` (не колонка номеров)."""
    return float(np.median([row.x1 - row.x0 for row in rows])) >= mm_to_px(FLUSH_MIN_WIDTH_MM, dpi)


def _paragraph(rows: list[Row], cx0: float, cx1: float, dpi: float) -> bool:
    """Абзац ли группа: не меньше ``FLUSH_MIN_FULL_ROWS`` рядов во всю ширину охвата (``FLUSH_FULL_SHARE``) и ни одного
    ряда «вправо» (:func:`row_side`).

    Второе условие — от подписи из двух авторов, где регалии длинные: «главный специалист Государственной» и
    «комиссии Совета Министров СССР» во всю ширину подписи выглядели абзацем, и подпись рвалась перед второй фамилией
    (ХиП 1991/01 IMG_0990_R). В абзаце строк «вправо» нет; в подписи — фамилии.
    """
    full = sum(row.x1 - row.x0 >= FLUSH_FULL_SHARE * (cx1 - cx0) for row in rows)
    right = any(row_side(row, cx0, cx1, dpi) is FlushSide.RIGHT for row in rows)
    return full >= FLUSH_MIN_FULL_ROWS and not right


def side_change(upper: list[Row], lower: list[Row], dpi: float) -> bool:
    """Меняется ли сторона выключки на стыке двух групп, одна под другой (:func:`change_kind` не None)."""
    return change_kind(upper, lower, dpi) is not None


# Куски одной строки: оси разнесены по вертикали меньше чем на такую долю большей высоты ряда. Правый кусок строки
# на изогнутой странице стоит выше остатка на 0.8 высоты (ХиП 1991/03 IMG_0082_L «ст. 214»: 10.4 px при высоте 13).
SAME_LINE_HEIGHTS = 1.0
# Наибольшее налезание куска на остаток строки по x, доля высоты ряда.
PIECE_OVERLAP_HEIGHTS = 0.75


def _line_piece(row: Row, rows: list[Row], dpi: float) -> bool:
    """Кусок строки: в ``rows`` есть другой ряд на той же высоте (:data:`SAME_LINE_HEIGHTS`) рядом по x — не перекрыт
    с ним (налезание до :data:`PIECE_OVERLAP_HEIGHTS` высоты) и отделён просветом уже ``FAR_GAP_MM``.

    Сегментация рвёт строку по широкому пробелу выключки по формату («…(ч. 1 ст. 214», ХиП 1991/03 IMG_0082_L,
    просвет 8 мм; «…и другие остаются», 04/IMG_0239_R), и правый кусок стоит у правого края с большим отступом слева —
    как строка вправо. Такой кусок сменой выключки не считается. Ряды абзаца над и под ним перекрыты с ним по x;
    короткая концовка абзаца над подписью отделена просветом в десятки мм (05/IMG_0332_R «…голос.» → «Плутарх»).

    Args:
        row: Проверяемый ряд.
        rows: Ряды вокруг (группы стыка и ряды страницы в их охвате).
        dpi: Разрешение рабочей копии.

    Returns:
        ``True`` — ряд — кусок строки.
    """
    far = mm_to_px(FAR_GAP_MM, dpi)
    for other in rows:
        if other is row or abs(other.y - row.y) >= SAME_LINE_HEIGHTS * max(other.height, row.height):
            continue
        overlap = min(other.x1, row.x1) - max(other.x0, row.x0)
        # Налезание по x — не больше PIECE_OVERLAP_HEIGHTS высоты: края кусков налезают на букву (04/IMG_0239_R
        # «остаются»: 10.5 px при высоте 14), а подпись под короткой концовкой абзаца перекрыта с ней сильнее («града?»
        # и «Н. Кривоусова (г. Москва)», ХиП 1991/06 IMG_0475_R: 16.5 px при высоте 17).
        if -far < overlap <= PIECE_OVERLAP_HEIGHTS * max(other.height, row.height):
            return True
    return False


def _short_paragraph(rows: list[Row], cx0: float, cx1: float, dpi: float) -> bool:
    """Короткий абзац для способа ``ALIGN_RIGHT``: хотя бы один ряд во всю ширину охвата, ни одного ряда «вправо»,
    все ряды, кроме первого, начинаются точно у левого края (``FLUSH_TOL_MM``), первый — у края или с абзацным
    отступом (:func:`_starts_left`).

    Цитата в две строки (полная и концовка у левого края) над автором вправо — абзац, хотя полных рядов в ней один
    (ХиП 1991/05 IMG_0332_R, «Ленивые всегда бывают…» / «было роде.» над «Ф. Вольтер»). Регалии подписи сюда не
    подходят: их строки стоят лесенкой, не у одного левого края.
    """
    full = any(row.x1 - row.x0 >= FLUSH_FULL_SHARE * (cx1 - cx0) for row in rows)
    right = any(row_side(row, cx0, cx1, dpi) is FlushSide.RIGHT for row in rows)
    tol = mm_to_px(FLUSH_TOL_MM, dpi)
    flush_left = _starts_left(rows[0], cx0, dpi) and all(row.x0 - cx0 <= tol for row in rows[1:])
    return full and not right and flush_left


def change_kind(
    upper: list[Row], lower: list[Row], dpi: float, lead: bool = False, page_rows: list[Row] | None = None
) -> FlushChange | None:
    """Вид смены стороны выключки на стыке двух групп, одна под другой.

    Случаи (охват — общий у обеих групп):

    * нижняя группа целиком выключена вправо с рваным левым краем (:func:`group_side`), а верхняя держится левого
      края (:func:`left_anchored`) и её последний ряд сам не «вправо» (абзац по формату → подпись авторов, ХиП
      1991/01 IMG_1008);
    * верхняя группа выключена вправо, а нижняя держится левого края и её первый ряд начинается у него (подпись →
      «г. Ереван», подпись или дата над абзацем).

    Строки одного абзаца так не стоят: следующий ряд абзаца (и первый ряд нового абзаца с отступом) начинается
    у левого края и идёт до правого.

    Args:
        upper: Ряды верхней группы сверху вниз.
        lower: Ряды нижней группы сверху вниз.
        dpi: Разрешение рабочей копии.
        lead: Смену решает первая строка нижней группы (способ ``ALIGN_RIGHT``): абзац → первая строка вправо —
            ``TO_RIGHT``, даже если ниже в группе регалия почти во всю ширину или следующая цитата (подпись
            «В. АНОХИН» над регалией «главный государственный арбитр…», ХиП 1991/04 IMG_0207_L; автор афоризма
            «Плутарх» над следующей цитатой, 05/IMG_0332_R); подпись (группа, начатая строкой вправо) → первая строка
            у левого края во всю ширину или влево — ``FROM_RIGHT``.
        page_rows: Все ряды страницы или None: кусок строки (:func:`_line_piece`) ищется и среди рядов других групп —
            правый кусок строки, разорванной широким пробелом, бывает отдельной группой (03/IMG_0082_L «ст. 214»).

    Returns:
        ``TO_RIGHT`` — абзац → строки вправо; ``FROM_RIGHT`` — строки вправо → абзац или влево;
        ``FROM_RIGHT_INDENT`` — строки вправо → абзац с отступом (только с ``lead``); ``None`` — смены нет.
    """
    rows = [*upper, *lower]
    cx0, cx1 = min(row.x0 for row in rows), max(row.x1 for row in rows)
    upper_side, lower_side = group_side(upper, cx0, cx1, dpi), group_side(lower, cx0, cx1, dpi)
    # Группа «вправо» из одних номеров (колонка номеров оглавления) — не подпись: стороной не считается.
    if upper_side is FlushSide.RIGHT and not _text_wide(upper, dpi):
        upper_side = None
    if lower_side is FlushSide.RIGHT and not _text_wide(lower, dpi):
        lower_side = None
    # Последний ряд верхней группы сам не «вправо» (иначе это продолжение подписи); начинаться точно у левого края он
    # не обязан: концовку абзаца на изогнутой странице сегментация режет, и ряд начинается с середины строки
    # (ХиП 1991/03 IMG_0116_R, «…на современном этапе.» с 17.6 мм).
    last_side = row_side(upper[-1], cx0, cx1, dpi)
    # Сторона «влево» или «по формату» должна быть настоящей: абзацем (не меньше ``FLUSH_MIN_FULL_ROWS`` рядов во всю
    # ширину охвата) или, у нижней группы, строками «влево» («г. Город»). Иначе две половины одной подписи (фамилия и
    # регалии одного автора — над фамилией и регалиями второго, ХиП 1991/01 IMG_1042_R) выглядели абзацем и подписью:
    # самая длинная регалия задаёт левый край охвата, и «держится левого края».
    # Первая строка нижней группы — строка вправо из текста (не номер оглавления) и не кусок строки (у ``lead`` кусок
    # не начинает новый блок, даже если он — целая группа «вправо»).
    # Соседи по строке — только внутри охвата двух групп: строки соседней колонки на той же высоте не в счёт.
    tol = mm_to_px(FLUSH_TOL_MM, dpi)
    inside = [row for row in page_rows or () if row.x0 >= cx0 - tol and row.x1 <= cx1 + tol]
    piece = lead and _line_piece(lower[0], rows + inside, dpi)
    lead_right = (
        lead and row_side(lower[0], cx0, cx1, dpi) is FlushSide.RIGHT and _text_wide(lower[:1], dpi) and not piece
    )
    if (
        (lower_side is FlushSide.RIGHT or lead_right)
        and not piece
        and (_paragraph(upper, cx0, cx1, dpi) or (lead and _short_paragraph(upper, cx0, cx1, dpi)))
        and left_anchored(upper, cx0, dpi)
        and last_side is not FlushSide.RIGHT
    ):
        return FlushChange.TO_RIGHT
    # Абзац → строки на общем центре вдали от левого края (подпись по центру своей оси, не у правого края). Серия — по
    # первым рядам нижней группы: у ``lead`` нижняя группа — весь остаток части, и под подписью бывает следующий абзац.
    if (
        not piece
        and _centred(_detached_run(lower, cx0, cx1, dpi), cx0, cx1, dpi)
        and (_paragraph(upper, cx0, cx1, dpi) or (lead and _short_paragraph(upper, cx0, cx1, dpi)))
        and left_anchored(upper, cx0, dpi)
        and not _detached(upper[-1], cx0, cx1, dpi)
    ):
        return FlushChange.TO_CENTRE
    # Строки на общем центре → абзац или строки влево, первая строка которых начинается у левого края.
    if (
        _centred(upper, cx0, cx1, dpi)
        and _starts_left(lower[0], cx0, dpi)
        and left_anchored(lower, cx0, dpi)
        and (_paragraph(lower, cx0, cx1, dpi) or lower_side is FlushSide.LEFT or lead)
    ):
        return FlushChange.FROM_CENTRE
    lower_left = _paragraph(lower, cx0, cx1, dpi) or lower_side is FlushSide.LEFT
    if lead:
        # Подпись — группа, начатая строкой вправо (регалии под фамилией бывают почти во всю ширину и сами не
        # «вправо»); возврат к абзацу — первая строка нижней группы «влево» или у левого края во всю ширину.
        first = row_side(lower[0], cx0, cx1, dpi)
        full = lower[0].x1 - lower[0].x0 >= FLUSH_FULL_SHARE * (cx1 - cx0)
        if row_side(upper[0], cx0, cx1, dpi) is FlushSide.RIGHT and _text_wide(upper[:1], dpi):
            upper_side = FlushSide.RIGHT
        # Строка во всю ширину у левого края — начало абзаца, только если она и следующая строка стоят у левого края
        # точно (``FLUSH_TOL_MM``, без абзацного отступа и лесенки регалий 3–6 мм) и следующая не «вправо»: внутри
        # подписи самая длинная регалия сама задаёт левый край охвата («материально-технического снабжения ВИПКИСа»,
        # ХиП 1991/01 IMG_0982_L; «главный специалист Государственной», IMG_0990_R), а строка под
        # ней уже отступает; у цитаты под автором предыдущей вторая строка снова у левого края (05/IMG_0332_R).
        tol = mm_to_px(FLUSH_TOL_MM, dpi)
        second_left = (
            len(lower) > 1
            and lower[0].x0 - cx0 <= tol
            and lower[1].x0 - cx0 <= tol
            and row_side(lower[1], cx0, cx1, dpi) is not FlushSide.RIGHT
        )
        # «Абзац» по охвату одной подписи не в счёт: две длинные регалии под фамилией выглядят абзацем, и подпись
        # рвалась на фамилию и регалии (смена жирности подтверждала разрез: ХиП 1991/01 IMG_1087_R «Н. БОДЕРСКОВА»,
        # 04/IMG_0182_R «В. САФОНОВ»).
        lower_left = lower_side is FlushSide.LEFT or first is FlushSide.LEFT or (first is None and full and second_left)
        # Первая строка абзаца с отступом, вторая во всю ширину точно у левого края — тоже возврат к абзацу (ответ
        # редакции под подписью читателя: «Согласно ст. 23 Основ…» под «(г. Завитинск, Амурская область)», ХиП 1991/01
        # IMG_1075_L), но с подкреплением: лесенка регалий под фамилией выглядит так же (отступ 3–6 мм, ниже — самая
        # длинная регалия у левого края охвата подписи).
        indented = (
            not lower_left
            and upper_side is FlushSide.RIGHT
            and first is None
            and full
            and _starts_left(lower[0], cx0, dpi)
            and len(lower) > 1
            and lower[1].x0 - cx0 <= tol
            and lower[1].x1 - lower[1].x0 >= FLUSH_FULL_SHARE * (cx1 - cx0)
        )
        if indented:
            # Под строкой с отступом ещё две полные строки точно у левого края — настоящий абзац, без подкреплений:
            # лесенка регалий двух таких строк не даёт («В. Бессонов (г. Тверь)» над «В целях улучшения…», ХиП 1991/06
            # IMG_0476_R).
            body = lower[1:3]
            if len(body) == 2 and all(
                row.x0 - cx0 <= tol and row.x1 - row.x0 >= FLUSH_FULL_SHARE * (cx1 - cx0) for row in body
            ):
                return FlushChange.FROM_RIGHT
            return FlushChange.FROM_RIGHT_INDENT
    if (
        upper_side is FlushSide.RIGHT
        and lower_left
        and left_anchored(lower, cx0, dpi)
        and _starts_left(lower[0], cx0, dpi)
    ):
        return FlushChange.FROM_RIGHT
    return None


def signature_continues(upper: list[Row], lower: list[Row], dpi: float, mode: FlushSplit | None = None) -> bool:
    """Продолжение подписи: нижняя группа — регалии или следующий автор той же подписи (способ ``ALIGN_RIGHT``).

    Подпись — верхняя группа, начатая строкой «вправо» из текста (:func:`row_side` в общем охвате); правые края групп
    совпадают (``FLUSH_TOL_MM``); возврата к абзацу или строке влево нет (:func:`change_kind` с ``lead`` — не
    ``FROM_RIGHT``). Такие группы склеиваются несмотря на разный набор: жирная фамилия и светлые регалии под ней —
    одна подпись (ХиП 1991/02 IMG_1174_R «М. РОЗЕНБЕРГ», 01/IMG_1030_R «В. ТИХИНЯ», «В. ГОЛОВАНОВ»).

    Args:
        upper: Ряды верхней группы сверху вниз.
        lower: Ряды нижней группы сверху вниз.
        dpi: Разрешение рабочей копии.
        mode: Способ (:class:`FlushSplit`); ``None`` — ``DEFAULT_FLUSH_SPLIT``; не ``ALIGN_RIGHT`` и не
            ``ALIGN_RIGHT_GAP`` — всегда ``False``.

    Returns:
        ``True`` — набор склейке не мешает.
    """
    if _mode(mode) not in (FlushSplit.ALIGN_RIGHT, FlushSplit.ALIGN_RIGHT_GAP):
        return False
    rows = [*upper, *lower]
    cx0, cx1 = min(row.x0 for row in rows), max(row.x1 for row in rows)
    if row_side(upper[0], cx0, cx1, dpi) is not FlushSide.RIGHT or not _text_wide(upper[:1], dpi):
        return False
    tol = mm_to_px(FLUSH_TOL_MM, dpi)
    if abs(max(row.x1 for row in upper) - max(row.x1 for row in lower)) > tol:
        return False
    # Абзац с отступом под подписью (``FROM_RIGHT_INDENT``) склейку здесь не запрещает: его отделяет промежуток
    # в :func:`flush_change`, а без промежутка это лесенка регалий (01/IMG_1030_R «начальник отдела…»).
    return change_kind(upper, lower, dpi, lead=True) in (None, FlushChange.FROM_RIGHT_INDENT)


def _median(rows: list[Row], name: str) -> float:
    """Медиана положительных значений поля ``name`` по рядам (0 — мера не набралась)."""
    values = [getattr(row, name) for row in rows if getattr(row, name) > 0]
    return float(np.median(values)) if values else 0.0


def type_change(upper: list[Row], lower: list[Row]) -> bool:
    """Сменился ли набор между группами: медиана высоты глифа или жирности отличается в ``FLUSH_TYPE_RATIO`` раз.

    Args:
        upper: Ряды верхней группы.
        lower: Ряды нижней группы.

    Returns:
        ``True`` — набор разный хотя бы по одной мере; молчащая мера (0 у одной из групп) ничего не решает.
    """
    for name in ("glyph_h", "weight"):
        values = sorted((_median(upper, name), _median(lower, name)))
        if values[0] > 0 and values[1] >= FLUSH_TYPE_RATIO * values[0]:
            return True
    return False


def flush_change(
    upper: list[Row],
    lower: list[Row],
    gap: float,
    pitch: float,
    dpi: float,
    mode: FlushSplit | None = None,
    page_rows: list[Row] | None = None,
) -> bool:
    """Запрет склейки двух групп по смене выключки (вызывается из ``regroup._rejoinable``).

    Args:
        upper: Ряды верхней группы сверху вниз.
        lower: Ряды нижней группы сверху вниз.
        gap: Расстояние между осями стыковых рядов вдоль наклона (``blocks.row_gap``), пиксели рабочей копии.
        pitch: Шаг строк групп (больший из двух, как в ``_rejoinable``).
        dpi: Разрешение рабочей копии.
        mode: Способ (:class:`FlushSplit`); ``None`` — ``DEFAULT_FLUSH_SPLIT``.
        page_rows: Все ряды страницы или None (поиск кусков строк, :func:`change_kind`).

    Returns:
        ``True`` — группы не склеивать: это разные блоки (абзац и подпись, подпись и город).
    """
    mode = _mode(mode)
    if mode is FlushSplit.OFF:
        return False
    if mode is FlushSplit.ALIGN_RIGHT_GAP:
        # Разрез любого из двух способов; разрез ``ALIGN_GAP`` — только не внутри подписи (фамилия и регалии).
        if flush_change(upper, lower, gap, pitch, dpi, FlushSplit.ALIGN_RIGHT, page_rows):
            return True
        if signature_continues(upper, lower, dpi, FlushSplit.ALIGN_RIGHT):
            return False
        return flush_change(upper, lower, gap, pitch, dpi, FlushSplit.ALIGN_GAP, page_rows)
    kind = change_kind(upper, lower, dpi, lead=mode is FlushSplit.ALIGN_RIGHT, page_rows=page_rows)
    if kind is None:
        return False
    if mode is FlushSplit.ALIGN or (mode is FlushSplit.ALIGN_RIGHT and kind is not FlushChange.FROM_RIGHT_INDENT):
        return True
    # Подкрепления: промежуток больше обычного интерлиньяжа или другой набор.
    wide = pitch > 0 and gap >= FLUSH_GAP_PITCHES * pitch
    if kind is FlushChange.FROM_RIGHT_INDENT:
        # Абзац с отступом под подписью — только по промежутку: смена набора внутри подписи есть всегда (жирная
        # фамилия над светлыми регалиями лесенкой, ХиП 1991/01 IMG_1030_R, IMG_1045_L).
        return wide
    return wide or type_change(upper, lower)


def split_by_flush(rows: list[Row], dpi: float, mode: FlushSplit | None = None) -> list[list[Row]]:
    """Разрезать группу рядов там, где внутри неё начинается серия строк «вправо» (подпись под абзацем).

    Склейка ложных разрезов (``regroup.rejoin``) проверяет смену выключки только на уже готовых стыках групп. Если
    подпись дотянулась до абзаца по x (фамилия правее короткой концовки, но перекрыта с ней: ХиП 1991/03 IMG_0116_R,
    «…на современном этапе.» и «Н. КАРГИНА,»), она с самого начала в одной группе с абзацем, и стыка нет. Здесь
    ищется серия рядов «вправо» в охвате группы (:func:`row_side`), идущая сразу за рядом, который начинается у левого
    края; на её границах ставятся разрезы, если :func:`flush_change` подтверждает смену выключки (с подкреплением
    промежутком или набором у ``ALIGN_GAP``).

    Args:
        rows: Ряды группы сверху вниз.
        dpi: Разрешение рабочей копии.
        mode: Способ (:class:`FlushSplit`); ``None`` — ``DEFAULT_FLUSH_SPLIT``; у ``OFF`` группа не режется.

    Returns:
        Части группы сверху вниз (одна — если резать нечего).
    """
    mode = _mode(mode)
    if mode is FlushSplit.OFF or len(rows) < 3:
        return [rows]
    if mode is FlushSplit.ALIGN_RIGHT:
        return _split_by_lead(rows, dpi)
    if mode is FlushSplit.ALIGN_RIGHT_GAP:
        # Сначала разрезы по первой строке смены выключки, затем внутри каждой части — по серии строк «вправо»
        # (:func:`flush_change` этого способа не режет внутри подписи).
        parts = _split_by_lead(rows, dpi)
        return [piece for part in parts for piece in (_split_by_side(part, dpi, mode) if len(part) >= 3 else [part])]
    return _split_by_side(rows, dpi, mode)


def _split_by_side(rows: list[Row], dpi: float, mode: FlushSplit) -> list[list[Row]]:
    """Разрезы группы способами ``ALIGN`` и ``ALIGN_GAP``: на границах серии рядов «вправо» под рядом у левого края.

    Args:
        rows: Ряды группы сверху вниз.
        dpi: Разрешение рабочей копии.
        mode: Способ (:class:`FlushSplit`), которым :func:`flush_change` подтверждает смену выключки.

    Returns:
        Части группы сверху вниз (одна — если резать нечего).
    """
    cx0, cx1 = min(row.x0 for row in rows), max(row.x1 for row in rows)
    sides = [row_side(row, cx0, cx1, dpi) for row in rows]
    cuts: list[int] = []
    index = 1
    while index < len(rows):
        if sides[index] is not FlushSide.RIGHT or sides[index - 1] is FlushSide.RIGHT:
            index += 1
            continue
        # Серия рядов «вправо», начиная с ``index``.
        end = index
        while end < len(rows) and sides[end] is FlushSide.RIGHT:
            end += 1
        start = cuts[-1] if cuts else 0
        upper, lower = rows[start:index], rows[index:end]
        # Шаг строк — по верхней части (абзацу); у одной строки — полторы её высоты.
        pitch = pitch_of(upper) if len(upper) > 1 else 1.6 * upper[0].height
        if flush_change(upper, lower, row_gap(upper[-1], lower[0]), pitch, dpi, mode):
            cuts.append(index)
            if end < len(rows):
                cuts.append(end)
        index = end
    if not cuts:
        return [rows]
    bounds = [0, *cuts, len(rows)]
    return [rows[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _split_by_lead(rows: list[Row], dpi: float) -> list[list[Row]]:
    """Разрезы группы способом ``ALIGN_RIGHT``: по первой строке смены выключки (:func:`change_kind` с ``lead``).

    Идёт по стыкам сверху вниз. Текущая часть — от последнего разреза. Разрез ставится:

    - перед строкой «вправо», если над ней в части нет строк «вправо» и :func:`flush_change` видит переход абзац →
      строка вправо (подпись, автор цитаты): без подкреплений;
    - перед строкой у левого края, если часть начата строкой «вправо» (подпись) и :func:`flush_change` видит возврат
      к абзацу или строке влево (следующая цитата, «г. Город») — тоже без подкреплений.

    Регалии под фамилией (почти во всю ширину, но не от левого края) подпись не рвут: они не «вправо» и не у левого
    края.

    Args:
        rows: Ряды группы сверху вниз.
        dpi: Разрешение рабочей копии.

    Returns:
        Части группы сверху вниз.
    """
    cx0, cx1 = min(row.x0 for row in rows), max(row.x1 for row in rows)
    sides = [row_side(row, cx0, cx1, dpi) for row in rows]
    cuts: list[int] = []
    for index in range(1, len(rows)):
        start = cuts[-1] if cuts else 0
        upper, lower = rows[start:index], rows[index:]
        right_part = sides[start] is FlushSide.RIGHT
        to_right = (
            sides[index] is FlushSide.RIGHT
            and not any(side is FlushSide.RIGHT for side in sides[start:index])
            and not _line_piece(rows[index], rows, dpi)
        )
        from_right = right_part and sides[index] is not FlushSide.RIGHT and _starts_left(rows[index], cx0, dpi)
        # То же для строк на общем центре (подпись по центру своей оси): начало серии под рядами у левого края и
        # возврат к левому краю после неё; подтверждает :func:`change_kind` (``TO_CENTRE`` / ``FROM_CENTRE``).
        to_centre = (
            _detached(rows[index], cx0, cx1, dpi)
            and not any(_detached(row, cx0, cx1, dpi) for row in rows[start:index])
            and not _line_piece(rows[index], rows, dpi)
        )
        from_centre = _detached(rows[start], cx0, cx1, dpi) and _starts_left(rows[index], cx0, dpi)
        if not (to_right or from_right or to_centre or from_centre):
            continue
        pitch = pitch_of(upper) if len(upper) > 1 else 1.6 * upper[0].height
        if flush_change(upper, lower, row_gap(upper[-1], lower[0]), pitch, dpi, FlushSplit.ALIGN_RIGHT):
            cuts.append(index)
    if not cuts:
        return [rows]
    bounds = [0, *cuts, len(rows)]
    return [rows[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _pieces(row: Row) -> tuple[tuple, tuple, float] | None:
    """Разбить оси ряда по самому широкому просвету по x на левый и правый куски.

    Args:
        row: Ряд из нескольких осей.

    Returns:
        ``(оси слева, оси справа, ширина просвета)`` или ``None`` — у ряда одна ось.
    """
    axes = sorted(row.axes, key=lambda axis: axis.x0)
    if len(axes) < 2:
        return None
    best, best_gap, reach = None, -np.inf, axes[0].x1
    for index in range(1, len(axes)):
        # Просвет — от правого края всего, что левее (оси могут налезать), до начала следующей оси.
        gap = axes[index].x0 - reach
        if gap > best_gap:
            best, best_gap = index, gap
        reach = max(reach, axes[index].x1)
    return tuple(axes[:best]), tuple(axes[best:]), float(best_gap)


def _covers(row: Row, axes: tuple) -> bool:
    """Перекрывает ли ряд по x хотя бы одну из осей куска."""
    return any(min(row.x1, axis.x1) - max(row.x0, axis.x0) > 0 for axis in axes)


def _one_sided(row: Row, left: tuple, right: tuple, rows: list[Row], skip: tuple[Row, ...] = ()) -> bool:
    """Дотягиваются ли соседние строки только до одного из кусков ряда и ни одна — до обоих.

    Args:
        row: Ряд из двух кусков (по нему берутся ордината и высота).
        left, right: Оси левого и правого кусков.
        rows: Все ряды куска колонки.
        skip: Ряды, которые соседями не считаются (сами куски, если они уже отдельные ряды).

    Returns:
        ``True`` — хотя бы одна соседняя строка (не дальше ``FAR_REACH_HEIGHTS`` высот) перекрыта только с одним
        куском, и ни одна не перекрыта с обоими (тогда просвет — пробел внутри строки таблицы или колонки).
    """
    one = False
    for other in rows:
        if other is row or any(other is item for item in skip):
            continue
        dy = abs(other.y - row.y)
        if dy <= FAR_SAME_ROW_HEIGHTS * min(row.height, other.height):
            continue
        if dy > FAR_REACH_HEIGHTS * max(row.height, other.height):
            continue
        on_left, on_right = _covers(other, left), _covers(other, right)
        if on_left and on_right:
            return False
        one = one or on_left or on_right
    return one


def far_pair(
    a: Row, b: Row, rows: list[Row], dpi: float, ink: np.ndarray | None = None, mode: FlushSplit | None = None
) -> bool:
    """Два ряда на одной высоте — далёкие куски, к которым соседние строки дотягиваются порознь.

    Тот же признак, что режет ряд в :func:`split_far_pieces`; нужен ``regroup.reach_components``, где ряды одной
    высоты иначе связываются всегда (половинки строки графы и числа за отточием) и разрезанный ряд срастался обратно.

    Args:
        a, b: Ряды на одной высоте.
        rows: Все ряды группы.
        dpi: Разрешение рабочей копии.
        ink: Краска текста рендера ``RENDER_DPI``: просвет с краской (отточие оглавления) — не пустота, и ряды
            остаются половинками одной строки; ``None`` — всегда ``False`` (без краски пустоту не проверить).
        mode: Способ (:class:`FlushSplit`); ``None`` — ``DEFAULT_FLUSH_SPLIT``; у ``OFF`` — всегда ``False``.

    Returns:
        ``True`` — ряды не связывать как половинки одной строки.
    """
    if _mode(mode) is FlushSplit.OFF or ink is None:
        return False
    left, right = (a, b) if a.x0 <= b.x0 else (b, a)
    if right.x0 - left.x1 <= mm_to_px(FAR_GAP_MM, dpi):
        return False
    if gap_ink_share(left, left.axes, right.axes, ink, dpi) > FAR_GAP_INK_SHARE:
        return False
    return _one_sided(left, left.axes, right.axes, rows, skip=(right,))


def gap_ink_share(row: Row, left: tuple, right: tuple, ink: np.ndarray, dpi: float) -> float:
    """Доля столбцов просвета между кусками ряда, где в полосе ряда есть краска текста.

    Args:
        row: Ряд (ордината и высота полосы).
        left, right: Оси левого и правого кусков.
        ink: Краска текста рендера ``RENDER_DPI`` (ненулевое — краска).
        dpi: Разрешение рабочей копии.

    Returns:
        Доля 0…1; 0 — просвет пуст (или пуст по размеру).
    """
    k = RENDER_DPI / dpi
    x0 = int(round(max(axis.x1 for axis in left) * k))
    x1 = int(round(min(axis.x0 for axis in right) * k))
    y0 = max(0, int(round((row.y - row.height / 2.0) * k)))
    y1 = min(ink.shape[0], int(round((row.y + row.height / 2.0) * k)) + 1)
    x0, x1 = max(0, x0), min(ink.shape[1], x1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.mean(np.any(ink[y0:y1, x0:x1] > 0, axis=0)))


def split_far_pieces(
    rows: list[Row],
    ink: np.ndarray,
    span: tuple[int, int],
    dpi: float,
    leaders: list | None = None,
    barriers=None,
    mode: FlushSplit | None = None,
) -> list[Row]:
    """Разрезать ряды из двух далёких кусков на одной высоте, к которым соседние строки дотягиваются порознь.

    ``blocks.rows_of`` сливает в ряд все оси одной высоты в колонке. ХиП 1991/01 IMG_1008, с. 59: «юрисконсульт»
    (последняя строка подписи, выключка вправо, x 529–631) и «г. Ереван» (выключка влево, x 36–102) стоят на одной
    высоте через просвет в 72 мм и становились одним рядом во всю ширину — он связывал подпись с абзацем над ней.
    Ряд режется, если просвет шире ``FAR_GAP_MM``, пуст (:func:`gap_ink_share`, отточие оглавления — не пустота) и
    соседние строки сверху или снизу дотягиваются только до одного куска (:func:`_one_sided`); каждый кусок становится своим рядом (края и меры — заново ``blocks.rows_of``).

    Args:
        rows: Ряды куска колонки.
        ink: Краска текста рендера ``RENDER_DPI``.
        span: Границы колонки по x.
        dpi: Разрешение рабочей копии.
        leaders: Отточия страницы (меры штриха их не берут).
        barriers: Линейки-барьеры.
        mode: Способ (:class:`FlushSplit`); ``None`` — ``DEFAULT_FLUSH_SPLIT``; у ``OFF`` ряды не режутся.

    Returns:
        Ряды сверху вниз.
    """
    if _mode(mode) is FlushSplit.OFF:
        return rows
    limit = mm_to_px(FAR_GAP_MM, dpi)
    out: list[Row] = []
    for row in rows:
        split = _pieces(row)
        if split is None or split[2] <= limit or not _one_sided(row, split[0], split[1], rows):
            out.append(row)
            continue
        if gap_ink_share(row, split[0], split[1], ink, dpi) > FAR_GAP_INK_SHARE:
            out.append(row)
            continue
        # Каждый кусок — свой ряд: края по краске, штрих и глифы мерятся заново по его полосе. Запасной режим
        # (``fallback``): границы колонки по межколонникам уже проверены при сборке исходного ряда.
        parts = [
            rows_of(list(axes), ink, span, dpi, leaders=leaders, fallback=True, barriers=barriers) for axes in split[:2]
        ]
        if all(parts):
            out.extend(piece for part in parts for piece in part)
        else:
            out.append(row)
    return sorted(out, key=lambda item: item.y)


__all__ = [
    "DEFAULT_FLUSH_SPLIT",
    "FlushChange",
    "FlushSide",
    "FlushSplit",
    "change_kind",
    "far_pair",
    "flush_change",
    "group_side",
    "row_side",
    "side_change",
    "signature_continues",
    "split_by_flush",
    "split_far_pieces",
    "type_change",
]
