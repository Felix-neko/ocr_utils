"""Добавление полей к экспортируемым из PDF изображениям.

Модуль решает три задачи:

1. **Цвет полей.** Поля заливаются не усреднённым цветом всей картинки (он «уехал» бы
   в сторону текста и иллюстраций), а оценённым цветом *бумаги* — см.
   :func:`estimate_paper_color`.

2. **Поля без перекодирования JPEG.** Если исходное изображение было JPEG, поля
   добавляются на уровне DCT-коэффициентов, без декодирования и повторного сжатия —
   см. :func:`pad_jpeg_lossless`. Это возможно только когда ширина поля кратна размеру
   MCU исходного файла (8 px без субдискретизации цветности, 16 px при 4:2:0), поэтому
   запрошенная ширина округляется вверх — см. :func:`align_padding_up`.

3. **Разрешение.** По запросу выходным файлам проставляется заданное DPI — для JPEG
   правкой JFIF APP0 прямо в байтах, без перекодирования (:func:`set_jpeg_dpi`). Это
   только тег: количество пикселей от него не меняется.

О точности «беспотерьного» варианта: DCT-коэффициенты всех блоков исходной картинки
переносятся в результат как есть, заново пережатых блоков нет. Без субдискретизации
цветности (4:4:4, градации серого) декодированный результат совпадает с оригиналом
побитово. При 4:2:0/4:2:2 отличается только внешний ряд пикселей шириной 1 px: декодер
интерполирует цветность по соседним отсчётам, а за краем картинки теперь лежит не
«продолжение края», а залитое поле. Всё, что внутри, совпадает пиксель в пиксель.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# --- Константы формата JPEG ---------------------------------------------------------

#: Размер DCT-блока JPEG. Минимально возможный шаг выравнивания полей — для картинок
#: без субдискретизации цветности (4:4:4) и для градаций серого.
JPEG_DCT_BLOCK_SIZE = 8

#: Максимальный размер MCU для распространённых схем субдискретизации (4:2:0 → 16x16).
#: Кратность этому числу гарантирует беспотерьную вставку для любой из них.
JPEG_MAX_MCU_SIZE = 16

#: Разрешение, которое обычно проставляют экспортируемым сканам. Значение по умолчанию
#: для CLI-опции --dpi; сама опция по умолчанию выключена.
TARGET_DPI = 300

#: Сколько раз подбирать цвет заливки под квантование JPEG (см. _compensate_fill_color).
FILL_COLOR_FIT_STEPS = 4

#: Схемы субдискретизации, которые умеет записывать Pillow: факторы (h1, v1, h2, v2, h3, v3)
#: из SOF → код для параметра subsampling. По порядку: 4:4:4, 4:2:2, 4:2:0.
_PIL_SUBSAMPLING = {(1, 1, 1, 1, 1, 1): 0, (2, 1, 1, 1, 1, 1): 1, (2, 2, 1, 1, 1, 1): 2}

# --- Константы оценки цвета бумаги --------------------------------------------------

#: До какого размера по большей стороне ужимается картинка перед анализом цвета бумаги.
PAPER_ANALYSIS_MAX_SIDE = 512

#: Окно локального СКО, по которому отбираются «гладкие» (не текст, не полутон) пиксели.
PAPER_FLATNESS_WINDOW = 5

#: Пиксель считается «гладким», если локальное СКО яркости меньше этого порога.
PAPER_FLATNESS_MAX_STD = 6.0

#: Полуширина окна вокруг найденного тона, из которого берутся пиксели бумаги.
PAPER_TONE_WINDOW = 12

#: Минимальное число пикселей в выборке, иначе отбор ослабляется.
PAPER_MIN_SAMPLE_PIXELS = 100

#: Цвет полей, если оценить не удалось (пустая картинка и т.п.).
PAPER_FALLBACK_COLOR_BGR = (255, 255, 255)


class LosslessPaddingError(RuntimeError):
    """Беспотерьное добавление полей к JPEG невозможно для этого файла."""


# --- Цвет бумаги --------------------------------------------------------------------


def estimate_paper_color(image_bgr: np.ndarray) -> tuple[int, int, int]:
    """Оценить цвет бумаги на скане.

    Простое усреднение по всей картинке даёт цвет, стянутый к тексту и иллюстрациям, —
    поля получились бы заметно темнее и «грязнее» листа. Поэтому цвет ищется так:

    1. Картинка ужимается до :data:`PAPER_ANALYSIS_MAX_SIDE` по большей стороне.
    2. Отбираются «гладкие» пиксели — с низким локальным СКО яркости. Текст, растр
       иллюстраций и края объектов дают высокое СКО и отсеиваются; бумага — гладкая.
    3. По гладким пикселям строится гистограмма яркости, и берётся её *самый массивный*
       тон — тот, вокруг которого сосредоточено больше всего пикселей. Бумага — самая
       большая гладкая область страницы, поэтому она и выигрывает.
    4. Цвет = поканальная медиана гладких пикселей, попавших в окно вокруг этого тона.
       Медиана устойчива к случайным вкраплениям.

    Раньше на шаге 3 брался самый *светлый* значимый пик, а не самый массивный. Это
    оказалось ошибкой: у сканов разворотов вдоль края почти всегда виден белый фон
    сканера, он ярче бумаги и совершенно гладкий — и выигрывал у неё. Поля заливались
    холодным почти-белым, тогда как страница тёплая и заметно темнее. Светлее бумаги
    может оказаться и блик, и вклеенный белый лист; массивнее бумаги — практически
    ничто, потому что она занимает страницу целиком.

    Из правила «побеждает наибольшая гладкая область» следует и его ограничение: на
    странице, залитой краской под обрез — цветной обложке, глубокой печати по чёрному
    фону — победит цвет краски, а не узкое белое поле по краю. Для заливки полей это
    скорее хорошо: поле сливается со страницей, а не обводит её светлой рамкой.

    Args:
        image_bgr: Картинка в BGR (uint8) или в градациях серого

    Returns:
        Цвет бумаги как кортеж (B, G, R) значений 0..255
    """
    if image_bgr is None or image_bgr.size == 0:
        return PAPER_FALLBACK_COLOR_BGR

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    height, width = image_bgr.shape[:2]
    scale = PAPER_ANALYSIS_MAX_SIDE / max(height, width)
    if scale < 1.0:
        small = cv2.resize(
            image_bgr, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
    else:
        small = image_bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Локальное СКО яркости: sqrt(E[x^2] - E[x]^2) по окну PAPER_FLATNESS_WINDOW.
    window = (PAPER_FLATNESS_WINDOW, PAPER_FLATNESS_WINDOW)
    gray_f = gray.astype(np.float32)
    local_mean = cv2.blur(gray_f, window)
    local_mean_sq = cv2.blur(gray_f * gray_f, window)
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0.0))
    flat_mask = local_std < PAPER_FLATNESS_MAX_STD

    if int(flat_mask.sum()) < PAPER_MIN_SAMPLE_PIXELS:
        # Скан целиком «шумный» (мелкий текст на всю площадь) — анализируем всё подряд.
        flat_mask = np.ones_like(flat_mask)

    paper_tone = _dominant_tone(gray[flat_mask])
    if paper_tone is None:
        return PAPER_FALLBACK_COLOR_BGR

    tone_mask = np.abs(gray.astype(np.int16) - paper_tone) <= PAPER_TONE_WINDOW
    paper_mask = flat_mask & tone_mask
    if int(paper_mask.sum()) < PAPER_MIN_SAMPLE_PIXELS:
        paper_mask = tone_mask
    if int(paper_mask.sum()) == 0:
        return (int(paper_tone), int(paper_tone), int(paper_tone))

    pixels = small[paper_mask]
    color = np.median(pixels.astype(np.float32), axis=0)
    return tuple(int(round(float(c))) for c in color)  # type: ignore[return-value]


def _dominant_tone(tones: np.ndarray) -> int | None:
    """Найти тон, вокруг которого сосредоточено больше всего пикселей.

    Ищется не самый высокий столбик гистограммы, а окно шириной 2*PAPER_TONE_WINDOW,
    в которое попадает больше всего пикселей. Разница принципиальная: у зернистой бумаги
    один физический тон размазан по десятку соседних столбиков, а у идеально ровной
    заливки (тёмная плашка, вклейка) весь её объём стоит в одном. По высоте столбика
    выигрывала бы плашка, даже занимая втрое меньше площади; по массе окна — бумага.

    Окно берётся то же самое, по которому потом считается цвет, так что выбранный тон
    и есть центр самой населённой выборки.

    Args:
        tones: Одномерный массив значений яркости (uint8)

    Returns:
        Яркость самого массивного тона или None, если данных нет
    """
    if tones.size == 0:
        return None

    hist = np.bincount(tones.ravel(), minlength=256).astype(np.float64)
    window = np.ones(2 * PAPER_TONE_WINDOW + 1, dtype=np.float64)
    return int(np.argmax(np.convolve(hist, window, mode="same")))


def brighten_color(color_bgr: tuple[int, int, int], amount: int | None) -> tuple[int, int, int]:
    """Осветлить цвет на amount тонов из 256 (с ограничением сверху).

    Args:
        color_bgr: Исходный цвет (B, G, R)
        amount: На сколько тонов осветлить; None или 0 — не менять

    Returns:
        Осветлённый цвет (B, G, R)
    """
    if not amount:
        return color_bgr
    return tuple(int(min(255, max(0, c + amount))) for c in color_bgr)  # type: ignore[return-value]


# --- Разбор структуры JPEG ----------------------------------------------------------


def _iter_segments(data: bytes):
    """Пройти по маркерам JPEG до начала сжатых данных.

    Yields:
        Кортежи (маркер, смещение_полезной_нагрузки, длина_полезной_нагрузки)
    """
    offset = 2  # пропускаем SOI
    while offset < len(data) - 1:
        if data[offset] != 0xFF:
            return
        marker = data[offset + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker == 0xD9:
            return
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        yield marker, offset + 4, length - 2
        if marker == 0xDA:  # SOS — дальше идут сжатые данные
            return
        offset += 2 + length


def _parse_quant_tables(payload: bytes) -> dict[int, bytes]:
    """Разобрать сегмент DQT в отдельные таблицы квантования.

    В одном сегменте DQT может лежать несколько таблиц подряд — Photoshop, например,
    пакует все три в один сегмент, а libjpeg пишет по одной в отдельных. Сравнивать
    сырые байты сегментов поэтому нельзя: одинаковые таблицы, разложенные по-разному,
    выглядят как разные.

    Args:
        payload: Полезная нагрузка сегмента DQT (без маркера и длины)

    Returns:
        Словарь {номер таблицы: 64 коэффициента в порядке зигзага}
    """
    tables: dict[int, bytes] = {}
    offset = 0
    while offset < len(payload):
        precision, table_id = payload[offset] >> 4, payload[offset] & 0x0F
        size = 128 if precision else 64  # 16-битные коэффициенты занимают вдвое больше
        tables[table_id] = payload[offset + 1 : offset + 1 + size]
        offset += 1 + size
    return tables


def read_jpeg_layout(data: bytes) -> dict:
    """Прочитать из JPEG параметры, влияющие на возможность беспотерьной вставки.

    Args:
        data: Байты JPEG-файла

    Returns:
        Словарь с ключами width, height, sampling (кортеж факторов), quant_tables
        (словарь {номер таблицы: её коэффициенты}), component_tables (номер таблицы
        для каждой компоненты), progressive (bool), adobe_transform (int | None)

    Raises:
        LosslessPaddingError: Если это не JPEG или в нём нет SOF
    """
    if not data.startswith(b"\xff\xd8"):
        raise LosslessPaddingError("файл не начинается с маркера SOI — это не JPEG")

    layout: dict = {"quant_tables": {}, "component_tables": (), "progressive": False, "adobe_transform": None}

    for marker, start, length in _iter_segments(data):
        payload = data[start : start + length]
        if marker == 0xDB:
            layout["quant_tables"].update(_parse_quant_tables(payload))
        elif marker in (0xC0, 0xC1, 0xC2):
            layout["progressive"] = marker == 0xC2
            layout["height"] = int.from_bytes(payload[1:3], "big")
            layout["width"] = int.from_bytes(payload[3:5], "big")
            n_components = payload[5]
            sampling = []
            tables = []
            for i in range(n_components):
                factors = payload[6 + i * 3 + 1]
                sampling.append((factors >> 4, factors & 0x0F))
                tables.append(payload[6 + i * 3 + 2])
            layout["sampling"] = tuple(sampling)
            layout["component_tables"] = tuple(tables)
        elif marker == 0xEE and payload[:5] == b"Adobe":
            layout["adobe_transform"] = payload[11] if length >= 12 else None

    if "width" not in layout:
        raise LosslessPaddingError("в JPEG не найден маркер SOF")
    return layout


def _patch_sof_dimensions(data: bytes, width: int, height: int) -> bytes:
    """Переписать размеры картинки в маркере SOF, не трогая сжатые данные.

    Нужно, чтобы «показать» или «спрятать» краевые пиксели, которые уже лежат в файле:
    кодировщик JPEG всегда дописывает картинку до целого числа MCU (повторяя крайний
    ряд пикселей), а SOF лишь говорит декодеру, где обрезать.

    Args:
        data: Байты JPEG-файла
        width: Новая ширина в пикселях
        height: Новая высота в пикселях

    Returns:
        Байты JPEG с новыми размерами в SOF

    Raises:
        LosslessPaddingError: Если маркер SOF не найден
    """
    patched = bytearray(data)
    for marker, start, _length in _iter_segments(data):
        if marker in (0xC0, 0xC1, 0xC2):
            patched[start + 1 : start + 3] = height.to_bytes(2, "big")
            patched[start + 3 : start + 5] = width.to_bytes(2, "big")
            return bytes(patched)
    raise LosslessPaddingError("в JPEG не найден маркер SOF")


def jpeg_mcu_size(sampling: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Размер MCU (минимальной кодируемой единицы) в пикселях.

    Args:
        sampling: Факторы субдискретизации каждой компоненты — кортеж пар (h, v)

    Returns:
        Пара (ширина, высота) MCU в пикселях
    """
    h_max = max(h for h, _ in sampling)
    v_max = max(v for _, v in sampling)
    return JPEG_DCT_BLOCK_SIZE * h_max, JPEG_DCT_BLOCK_SIZE * v_max


def align_padding_up(padding: int, step: int) -> int:
    """Округлить ширину поля вверх до кратной шагу.

    Args:
        padding: Запрошенная ширина поля в пикселях
        step: Шаг выравнивания (размер MCU)

    Returns:
        Ширина поля, кратная шагу
    """
    if step <= 1:
        return padding
    return ((padding + step - 1) // step) * step


def jpeg_padding_step(data: bytes) -> int:
    """Шаг, которому должна быть кратна ширина поля для данного JPEG.

    Args:
        data: Байты JPEG-файла

    Returns:
        Шаг выравнивания в пикселях (8 или 16 для распространённых схем)
    """
    mcu_width, mcu_height = jpeg_mcu_size(read_jpeg_layout(data)["sampling"])
    return math.lcm(mcu_width, mcu_height)


# --- Разрешение (DPI) ---------------------------------------------------------------


#: Теги Exif, задающие разрешение: XResolution, YResolution, ResolutionUnit.
_EXIF_X_RESOLUTION = 0x011A
_EXIF_Y_RESOLUTION = 0x011B
_EXIF_RESOLUTION_UNIT = 0x0128


def read_jfif_density(data: bytes) -> tuple[int, int, int] | None:
    """Прочитать плотность из сегмента JFIF APP0.

    Args:
        data: Байты JPEG-файла

    Returns:
        Кортеж (единицы измерения, X, Y) или None, если сегмента JFIF нет
    """
    for marker, start, length in _iter_segments(data):
        if marker == 0xE0 and data[start : start + 5] == b"JFIF\x00" and length >= 12:
            return (
                data[start + 7],
                int.from_bytes(data[start + 8 : start + 10], "big"),
                int.from_bytes(data[start + 10 : start + 12], "big"),
            )
    return None


def write_jfif_density(data: bytes, unit: int, x_density: int, y_density: int) -> bytes:
    """Записать плотность в сегмент JFIF APP0, вставив его при необходимости.

    Сжатые данные не трогаются: правятся (или дописываются) только байты заголовка.

    Args:
        data: Байты JPEG-файла
        unit: Единицы измерения JFIF (0 — без единиц, 1 — на дюйм, 2 — на сантиметр)
        x_density: Плотность по горизонтали
        y_density: Плотность по вертикали

    Returns:
        Байты JPEG с записанной плотностью
    """
    density = x_density.to_bytes(2, "big") + y_density.to_bytes(2, "big")

    for marker, start, length in _iter_segments(data):
        if marker == 0xE0 and data[start : start + 5] == b"JFIF\x00" and length >= 12:
            patched = bytearray(data)
            patched[start + 7] = unit
            patched[start + 8 : start + 12] = density
            return bytes(patched)

    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01" + bytes([unit]) + density + b"\x00\x00"
    return data[:2] + app0 + data[2:]


def set_jpeg_dpi(data: bytes, dpi: int | None) -> bytes:
    """Проставить разрешение в JPEG, не трогая сжатые данные.

    Правится (или вставляется) сегмент JFIF APP0: единицы измерения — точки на дюйм,
    плотность по обеим осям — dpi. Если в файле есть Exif (APP1), разрешение правится
    и там — иначе часть программ прочитает старое значение из Exif, а не из JFIF.
    Сами DCT-коэффициенты не декодируются.

    Args:
        data: Байты JPEG-файла
        dpi: Разрешение, точек на дюйм. None — не трогать разрешение

    Returns:
        Байты JPEG с проставленным разрешением (или без изменений, если dpi=None)
    """
    if dpi is None:
        return data

    patched = bytearray(data)
    for marker, start, length in _iter_segments(data):
        if marker == 0xE1 and data[start : start + 6] == b"Exif\x00\x00":
            _patch_exif_resolution(patched, start + 6, length - 6, dpi)

    return write_jfif_density(bytes(patched), 1, dpi, dpi)  # единицы: точки на дюйм


def _patch_exif_resolution(data: bytearray, tiff_start: int, tiff_length: int, dpi: int) -> None:
    """Переписать теги разрешения в IFD0 блока Exif (на месте, длина не меняется).

    Значения XResolution/YResolution лежат в блоке Exif как рациональные числа по
    смещению; переписываем их на dpi/1 и выставляем единицы измерения в «дюймы».
    Теги, которых в файле нет, не добавляются: это потребовало бы сдвига смещений.

    Args:
        data: Изменяемый буфер со всем JPEG
        tiff_start: Смещение начала TIFF-заголовка внутри буфера
        tiff_length: Длина блока Exif от TIFF-заголовка
        dpi: Разрешение, точек на дюйм
    """
    if tiff_length < 8:
        return
    byte_order = bytes(data[tiff_start : tiff_start + 2])
    if byte_order == b"II":
        order = "little"
    elif byte_order == b"MM":
        order = "big"
    else:
        return

    def read_int(offset: int, size: int) -> int:
        return int.from_bytes(data[tiff_start + offset : tiff_start + offset + size], order)

    ifd0 = read_int(4, 4)
    if not 8 <= ifd0 <= tiff_length - 2:
        return
    entry_count = read_int(ifd0, 2)

    for index in range(entry_count):
        entry = ifd0 + 2 + index * 12
        if entry + 12 > tiff_length:
            return
        tag = read_int(entry, 2)
        if tag in (_EXIF_X_RESOLUTION, _EXIF_Y_RESOLUTION):
            value_offset = read_int(entry + 8, 4)
            if value_offset + 8 > tiff_length:
                continue
            base = tiff_start + value_offset
            data[base : base + 4] = dpi.to_bytes(4, order)
            data[base + 4 : base + 8] = (1).to_bytes(4, order)
        elif tag == _EXIF_RESOLUTION_UNIT:
            base = tiff_start + entry + 8
            data[base : base + 2] = (2).to_bytes(2, order)  # 2 — дюймы


# --- Беспотерьная вставка полей в JPEG ----------------------------------------------


def _canvas_subsampling(sampling: tuple[tuple[int, int], ...]) -> int:
    """Код субдискретизации Pillow для заданных факторов SOF.

    Args:
        sampling: Факторы субдискретизации каждой компоненты

    Returns:
        Код для параметра subsampling Pillow

    Raises:
        LosslessPaddingError: Если схема не поддерживается Pillow
    """
    if len(sampling) == 1:
        return -1  # градации серого: субдискретизации нет
    if len(sampling) != 3:
        raise LosslessPaddingError(f"неподдерживаемое число компонент: {len(sampling)}")
    flat = tuple(value for pair in sampling for value in pair)
    if flat not in _PIL_SUBSAMPLING:
        raise LosslessPaddingError(f"неподдерживаемая схема субдискретизации: {flat}")
    return _PIL_SUBSAMPLING[flat]


def _compensate_fill_color(color_bgr: tuple[int, int, int], mode: str, save_kwargs: dict) -> tuple[int, int, int]:
    """Подобрать цвет заливки так, чтобы после сжатия получился именно заказанный.

    Подложка сжимается таблицами квантования исходника, а у сильно сжатых сканов шаг
    квантования DC-коэффициента цветности доходит до полутора десятков тонов — заливка
    «уезжает» на столько же. Поэтому цвет подбирается по обратной связи: кодируем
    пробный квадрат, смотрим, что получилось на выходе, и сдвигаем вход на разницу.

    Args:
        color_bgr: Желаемый цвет заливки (B, G, R)
        mode: Режим Pillow для подложки ("L" или "RGB")
        save_kwargs: Параметры сохранения JPEG (таблицы квантования, субдискретизация)

    Returns:
        Цвет (B, G, R), который надо подать на вход кодировщику
    """
    target = np.array(color_bgr, dtype=np.int16)
    current = target.copy()
    best = current.copy()
    best_error = None

    for _ in range(FILL_COLOR_FIT_STEPS):
        probe = Image.new(mode, (32, 32), _pil_fill_value(mode, tuple(int(c) for c in current)))
        buffer = BytesIO()
        probe.save(buffer, **save_kwargs)
        with Image.open(BytesIO(buffer.getvalue())) as decoded:
            sample = np.asarray(decoded.convert("RGB"))[16, 16]
        actual = np.array([sample[2], sample[1], sample[0]], dtype=np.int16)

        error = target - actual
        magnitude = int(np.abs(error).max())
        if best_error is None or magnitude < best_error:
            best_error, best = magnitude, current.copy()
        if magnitude == 0:
            break
        current = np.clip(current + error, 0, 255)

    return tuple(int(c) for c in best)  # type: ignore[return-value]


def _pil_fill_value(mode: str, color_bgr: tuple[int, int, int]) -> int | tuple[int, int, int]:
    """Значение заливки для Image.new в нужном режиме.

    Args:
        mode: Режим Pillow ("L" или "RGB")
        color_bgr: Цвет (B, G, R)

    Returns:
        Яркость для "L" либо кортеж (R, G, B) для "RGB"
    """
    if mode == "L":
        return int(round(0.114 * color_bgr[0] + 0.587 * color_bgr[1] + 0.299 * color_bgr[2]))
    return color_bgr[2], color_bgr[1], color_bgr[0]


def _build_canvas(
    data: bytes, layout: dict, padding: int, color_bgr: tuple[int, int, int], content_width: int, content_height: int
) -> bytes:
    """Собрать JPEG-«подложку» — сплошную заливку с полями, совместимую с исходником.

    Совместимость означает совпадение таблиц квантования и факторов субдискретизации:
    только тогда jpegtran сможет перенести блоки исходника без перекодирования.

    Args:
        data: Байты исходного JPEG
        layout: Результат :func:`read_jpeg_layout` для исходника
        padding: Ширина поля в пикселях (уже выровненная)
        color_bgr: Цвет заливки (B, G, R)
        content_width: Ширина области под исходник (дополненная до целого числа MCU)
        content_height: Высота области под исходник (дополненная до целого числа MCU)

    Returns:
        Байты JPEG-подложки

    Raises:
        LosslessPaddingError: Если подложку не удалось сделать совместимой
    """
    with Image.open(BytesIO(data)) as source:
        quantization = source.quantization

    subsampling = _canvas_subsampling(layout["sampling"])
    mode = "L" if len(layout["sampling"]) == 1 else "RGB"

    save_kwargs: dict = {"format": "JPEG", "qtables": quantization}
    if subsampling >= 0:
        save_kwargs["subsampling"] = subsampling

    fitted = _compensate_fill_color(color_bgr, mode, save_kwargs)
    size = (content_width + 2 * padding, content_height + 2 * padding)
    canvas = Image.new(mode, size, _pil_fill_value(mode, fitted))

    buffer = BytesIO()
    canvas.save(buffer, **save_kwargs)
    canvas_bytes = buffer.getvalue()

    # Проверяем, что подложка действительно совместима: иначе jpegtran либо откажется,
    # либо (с -trim) молча перекванутует исходник — то есть испортит его.
    canvas_layout = read_jpeg_layout(canvas_bytes)
    if canvas_layout["sampling"] != layout["sampling"]:
        raise LosslessPaddingError(
            f"подложка получилась с другой субдискретизацией: "
            f"{canvas_layout['sampling']} вместо {layout['sampling']}"
        )
    if canvas_layout["quant_tables"] != layout["quant_tables"]:
        raise LosslessPaddingError("подложка получилась с другими таблицами квантования")
    if canvas_layout["component_tables"] != layout["component_tables"]:
        raise LosslessPaddingError(
            f"подложка привязала компоненты к другим таблицам: "
            f"{canvas_layout['component_tables']} вместо {layout['component_tables']}"
        )
    return canvas_bytes


def pad_jpeg_lossless(
    data: bytes, padding: int, color_bgr: tuple[int, int, int], dpi: int | None = None
) -> tuple[bytes, int]:
    """Добавить к JPEG поля, не перекодируя исходное изображение.

    Работает так: собирается JPEG-подложка нужного размера, залитая цветом бумаги и
    использующая те же таблицы квантования и ту же субдискретизацию, что и исходник.
    Затем ``jpegtran -drop`` переносит в неё DCT-блоки исходника — без декодирования
    и без повторного сжатия, поэтому новых артефактов сжатия не появляется.

    Ширина поля округляется вверх до кратной размеру MCU: вставка возможна только по
    границам MCU. Фактически использованная ширина возвращается вторым элементом.

    Отдельная тонкость — краевые MCU. ``jpegtran -drop`` переносит только целые MCU,
    поэтому у картинки, размеры которой не кратны MCU, последний неполный ряд/столбец
    просто не переносился бы: правый и нижний край потерялись бы. Чтобы этого не было,
    исходнику временно проставляется размер, дополненный до целого числа MCU (эти
    пиксели физически уже лежат в файле — кодировщик дописал их повтором крайнего ряда),
    а результату затем проставляется точный размер W+2*padding на H+2*padding. Исходная
    картинка при этом сохраняется целиком и побитово; побочный эффект — первые несколько
    пикселей правого и нижнего поля показывают не заливку, а этот дописанный кодировщиком
    повтор края (не более размера MCU минус один пиксель).

    Args:
        data: Байты исходного JPEG
        padding: Запрошенная ширина поля в пикселях
        color_bgr: Цвет заливки полей (B, G, R)
        dpi: Разрешение, которое проставить результату. None — не трогать разрешение

    Returns:
        Пара (байты результата, фактическая ширина поля)

    Raises:
        LosslessPaddingError: Если беспотерьная вставка невозможна
    """
    if shutil.which("jpegtran") is None:
        raise LosslessPaddingError("не найдена утилита jpegtran (пакет libjpeg-turbo-progs)")

    layout = read_jpeg_layout(data)
    if layout["progressive"]:
        raise LosslessPaddingError("прогрессивный JPEG не поддерживается")
    if len(layout["sampling"]) == 3 and layout["adobe_transform"] not in (None, 1):
        raise LosslessPaddingError(f"нестандартное цветовое преобразование Adobe: {layout['adobe_transform']}")

    width, height = layout["width"], layout["height"]
    mcu_width, mcu_height = jpeg_mcu_size(layout["sampling"])
    aligned = align_padding_up(padding, math.lcm(mcu_width, mcu_height))

    # Размеры, дополненные до целого числа MCU: столько пикселей на самом деле лежит в файле.
    full_width = align_padding_up(width, mcu_width)
    full_height = align_padding_up(height, mcu_height)

    source_bytes = (
        data if (full_width, full_height) == (width, height) else _patch_sof_dimensions(data, full_width, full_height)
    )
    canvas_bytes = _build_canvas(data, layout, aligned, color_bgr, full_width, full_height)

    with tempfile.TemporaryDirectory(prefix="pdf_padding_") as tmp_dir:
        tmp = Path(tmp_dir)
        source_path = tmp / "source.jpg"
        canvas_path = tmp / "canvas.jpg"
        output_path = tmp / "output.jpg"
        source_path.write_bytes(source_bytes)
        canvas_path.write_bytes(canvas_bytes)

        result = subprocess.run(
            [
                "jpegtran",
                "-copy",
                "none",
                "-optimize",
                "-outfile",
                str(output_path),
                "-drop",
                f"+{aligned}+{aligned}",
                str(source_path),
                str(canvas_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0 or not output_path.exists():
            message = result.stderr.decode("utf-8", "replace").strip()
            raise LosslessPaddingError(f"jpegtran завершился с ошибкой: {message}")
        padded = output_path.read_bytes()

    padded_layout = read_jpeg_layout(padded)
    expected = (full_width + 2 * aligned, full_height + 2 * aligned)
    if (padded_layout["width"], padded_layout["height"]) != expected:
        raise LosslessPaddingError(
            f"размер результата {padded_layout['width']}x{padded_layout['height']} "
            f"не совпадает с ожидаемым {expected[0]}x{expected[1]}"
        )

    padded = _patch_sof_dimensions(padded, width + 2 * aligned, height + 2 * aligned)

    if dpi is None:
        # jpegtran -copy none выбрасывает заголовки исходника — переносим хотя бы
        # разрешение, иначе поля «обнулили» бы его на ровном месте.
        source_density = read_jfif_density(data)
        if source_density is not None:
            padded = write_jfif_density(padded, *source_density)
        return padded, aligned

    return set_jpeg_dpi(padded, dpi), aligned


# --- Поля на растре -----------------------------------------------------------------


def pad_image_array(image_bgr: np.ndarray, padding: int, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Добавить к растру поля, залитые сплошным цветом.

    Args:
        image_bgr: Картинка в BGR или в градациях серого
        padding: Ширина поля в пикселях
        color_bgr: Цвет заливки (B, G, R)

    Returns:
        Картинка с полями
    """
    if padding <= 0:
        return image_bgr
    if image_bgr.ndim == 2:
        value: float | tuple[int, ...] = float(_pil_fill_value("L", color_bgr))
    elif image_bgr.shape[2] == 4:
        value = (*color_bgr, 255)  # поля должны быть непрозрачными
    else:
        value = color_bgr
    return cv2.copyMakeBorder(image_bgr, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=value)
