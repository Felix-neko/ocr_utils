"""Синтетическая газетная полоса с объектами закраса (две печати, палец, рукопись) и её чистый эталон.

Синтетика, а не скан из сети, ради эталона: под печатью на настоящем скане неизвестно, что было, а
здесь та же полоса без объектов есть, и у каждого варианта закраса можно посчитать ошибку, а не
только смотреть. Генерация детерминированная (фиксированный seed), масштаб — как у скана 300–450 DPI.
"""

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Размер полосы, пикс., и правый край листа: правее — тёмный «стол», как на фотоскане разворота.
WIDTH, HEIGHT = 2400, 3200
PAGE_RIGHT = 2180

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Слова для «текста» колонок: смысл не нужен, нужна фактура кириллического набора.
WORDS = (
    "план завод выпуск станок деталь производство качество бригада смена цех рабочий норма "
    "поставка металл продукция совет сырьё изделие опыт область министерство задание пятилетка "
    "мастер инженер технология машина трактор участок года процентов было который также"
).split()

# Колонки текста (x0, x1), их общий верх и низ и рамка полутоновой «фотографии» во второй колонке.
COLUMNS = ((180, 1040), (1140, 2000))
TEXT_TOP, TEXT_BOTTOM = 320, 2950
LINE_STEP = 48
PHOTO_BOX = (1140, 1250, 2000, 1900)

# Центры и радиусы печатей, пикс.
STAMP_CENTER, STAMP_RADIUS = (620, 900), 170
BIG_STAMP_CENTER, BIG_STAMP_RADIUS = (1560, 2700), 330
# Запас сплошной маски крупной печати за её внешним кольцом — как у кисти разметчика в CVAT.
BIG_STAMP_BRUSH_MARGIN = 20
# Начало строки рукописных росчерков (левый край, базовая линия) — поверх текста первой колонки.
HANDWRITING_ORIGIN = (250, 1580)


class TargetObject(Enum):
    """Объект закраса на синтетической полосе; значение — имя файла маски и подпись на листах."""

    STAMP = "stamp"  # печать ~340 px поверх текста, маска точно по краске
    BIG_STAMP = "big_stamp"  # печать ~660 px поверх текста и поля, маска — сплошной диск (кисть CVAT)
    FINGER = "finger"  # палец со стола на лист поверх полутоновой фотографии
    HANDWRITING = "handwriting"  # шесть несвязных росчерков карандашом поверх строки


@dataclass(frozen=True)
class SyntheticPage:
    """Полоса с объектами, тот же лист без них и маски объектов (uint8 0/255, по объекту)."""

    page: np.ndarray  # RGB uint8 (H, W, 3) — с объектами, её и закрашиваем
    clean: np.ndarray  # RGB uint8 (H, W, 3) — эталон без объектов
    masks: "dict[TargetObject, np.ndarray]"


def paper(rng: np.random.Generator) -> np.ndarray:
    """Бумага с неравномерным освещением, низкочастотной «облачностью» и зерном, справа — стол.

    Аргументы:
        rng: генератор случайных чисел (зерно и облачность).

    Возвращает float32 RGB (H, W, 3) в шкале 0..255 без обрезки.
    """
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    # Освещение: светлее к центру листа, темнее к углам — как под лампой репродукционной установки.
    light = 1.0 - 0.10 * ((xx - WIDTH * 0.4) ** 2 + (yy - HEIGHT * 0.45) ** 2) / (WIDTH**2)
    img = np.array([214, 204, 184], np.float32)[None, None, :] * light[..., None]
    # Облачность бумаги: шум на сетке в 8 раз грубее, размытый и растянутый до кадра.
    low = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT // 8, WIDTH // 8)).astype(np.float32), (0, 0), 3)
    low = cv2.resize(low, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) * 6
    grain = rng.normal(0, 3.5, (HEIGHT, WIDTH)).astype(np.float32)
    img += (low + grain)[..., None]
    # Стол справа от листа и тень у кромки листа, спадающая на 25 px.
    table = rng.normal(0, 4, (HEIGHT, WIDTH - PAGE_RIGHT, 1)).astype(np.float32)
    img[:, PAGE_RIGHT:] = np.array([38, 33, 30], np.float32) + table
    for k in range(25):
        img[:, PAGE_RIGHT + k] *= 0.4 + 0.024 * k
    return img


def text_ink(rng: np.random.Generator) -> np.ndarray:
    """Краска набора: заголовок, линейки и две колонки текста с окном под фотографию.

    Аргументы:
        rng: генератор случайных чисел (выбор слов).

    Возвращает float32 (H, W) — покрытие краской 0..1, слегка размытое, как у печати на бумаге.
    """
    picture = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype(SERIF, 34)
    draw.text((180, 170), "ЗА КАЧЕСТВО ПРОДУКЦИИ", font=ImageFont.truetype(SERIF_BOLD, 64), fill=255)
    draw.line((180, 270, 2000, 270), fill=255, width=4)
    draw.line((1090, TEXT_TOP, 1090, TEXT_BOTTOM), fill=255, width=2)
    for x0, x1 in COLUMNS:
        for y in range(TEXT_TOP, TEXT_BOTTOM, LINE_STEP):
            # Во второй колонке строки обходят фотографию.
            if x0 == PHOTO_BOX[0] and PHOTO_BOX[1] - 50 < y < PHOTO_BOX[3] + 10:
                continue
            # Набираем слова, пока строка влезает в колонку.
            words, x = [], x0
            while True:
                word = str(rng.choice(WORDS))
                x += draw.textlength(word + " ", font=font)
                if x > x1:
                    break
                words.append(word)
            draw.text((x0, y), " ".join(words), font=font, fill=255)
    return cv2.GaussianBlur(np.asarray(picture, np.float32) / 255.0, (0, 0), 0.7)


def halftone_dots(width: int, height: int) -> np.ndarray:
    """Растровая «фотография»: точки на сетке 8 px, размер точки задаёт плавный тон.

    Аргументы:
        width, height: размер фотографии, пикс.

    Возвращает float32 (height, width) — покрытие краской 0..1.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    blob = np.exp(-((xx - width * 0.6) ** 2 + (yy - height * 0.4) ** 2) / 20000)
    tone = np.clip(0.5 + 0.3 * np.sin(xx / 90) * np.cos(yy / 70) + 0.2 * blob, 0, 1)
    cell = 8
    radius2 = (xx % cell - cell / 2) ** 2 + (yy % cell - cell / 2) ** 2
    dots = (radius2 < (tone * cell * 0.75) ** 2).astype(np.float32)
    return cv2.GaussianBlur(dots, (0, 0), 0.8)


def stamp_ink(center: "tuple[int, int]", radius: int, text: str, rng: np.random.Generator) -> np.ndarray:
    """Круглая печать: два кольца, надпись по кругу, номер в центре, неровный оттиск.

    Аргументы:
        center: центр печати (x, y), пикс.;
        radius: радиус внешнего кольца, пикс.;
        text: надпись по кругу (буквы раскладываются равномерно на полный оборот);
        rng: генератор случайных чисел (пятнистость оттиска).

    Возвращает float32 (H, W) — покрытие краской печати 0..1.
    """
    cx, cy = center
    picture = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(picture)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=255, width=max(4, radius // 25))
    inner = int(radius * 0.72)
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), outline=255, width=max(3, radius // 40))
    font = ImageFont.truetype(SANS_BOLD, max(14, int(radius * 0.17)))
    ring = (radius + inner) / 2
    for i, char in enumerate(text):
        # Каждая буква рисуется на своём маленьком холсте, поворачивается по касательной и вклеивается.
        angle = -math.pi / 2 + 2 * math.pi * i / len(text)
        glyph = Image.new("L", (font.size * 2, font.size * 2), 0)
        ImageDraw.Draw(glyph).text((font.size // 2, font.size // 4), char, font=font, fill=255)
        glyph = glyph.rotate(-math.degrees(angle) - 90, resample=Image.BICUBIC)
        x, y = cx + ring * math.cos(angle), cy + ring * math.sin(angle)
        picture.paste(255, (int(x - font.size), int(y - font.size)), glyph)
    draw.text((cx, cy), "№ 17", font=ImageFont.truetype(SANS_BOLD, int(radius * 0.28)), fill=255, anchor="mm")
    ink = np.asarray(picture, np.float32) / 255.0
    # Неровный оттиск: плотность краски гуляет пятнами от 0.3 до 1.
    patchy = cv2.GaussianBlur(rng.uniform(0, 1, (HEIGHT // 4, WIDTH // 4)).astype(np.float32), (0, 0), 2)
    patchy = cv2.resize(patchy, (WIDTH, HEIGHT))
    ink = ink * np.clip(0.55 + 2.4 * (patchy - 0.5), 0.3, 1.0)
    return cv2.GaussianBlur(ink, (0, 0), 1.0)


def handwriting_ink(origin: "tuple[int, int]", rng: np.random.Generator) -> np.ndarray:
    """Шесть росчерков-«слов» карандашом в строку с зазорами 12–18 px — шесть связных областей маски.

    Аргументы:
        origin: левый край и базовая линия строки росчерков, пикс.;
        rng: генератор случайных чисел (ширина и волна росчерков, зазоры).

    Возвращает float32 (H, W) — покрытие графитом 0..1.
    """
    picture = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(picture)
    x, y0 = origin
    for k in range(6):
        width = rng.uniform(50, 90)
        waves = rng.uniform(2, 4)
        points = [(x + width * t, y0 + 28 * math.sin(t * math.pi * waves + k) + 10 * t) for t in np.linspace(0, 1, 40)]
        draw.line(points, fill=255, width=7, joint="curve")
        x += width + rng.uniform(12, 18)
    return np.asarray(picture, np.float32) / 255.0


def finger_layer() -> "tuple[np.ndarray, np.ndarray]":
    """Палец: вытянутый эллипс телесного цвета со стола на лист, затенённый к краям, с ногтем на кончике.

    Возвращает ``(цвет, альфа)``: float32 RGB (H, W, 3) и float32 (H, W) 0..1.
    """
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    cx, cy = 2120.0, 1790.0
    # Ось пальца смотрит влево-вверх: с правого края кадра на полутоновую фотографию.
    angle = math.atan2(-70, -520)
    u = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
    v = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
    half_length, half_width = 300.0, 95.0
    alpha = np.clip((1.0 - (u / half_length) ** 2 - (v / half_width) ** 2) * 12, 0, 1)
    shade = np.clip(1.0 - 0.35 * (v / half_width) ** 2, 0.5, 1.0)
    skin = np.array([206, 150, 125], np.float32)[None, None, :] * shade[..., None]
    nail = ((u - (half_length - 70)) / 55) ** 2 + (v / 55) ** 2 < 1
    skin[nail] = np.array([230, 190, 180], np.float32)
    return skin, alpha


def over(img: np.ndarray, ink: np.ndarray, color: "tuple[int, int, int]", opacity: float) -> np.ndarray:
    """Кладёт краску цвета ``color`` с покрытием ``ink`` (0..1) и плотностью ``opacity`` поверх ``img``.

    Возвращает новый float32 RGB.
    """
    a = (opacity * ink)[..., None]
    return img * (1 - a) + np.array(color, np.float32) * a


def make_page(seed: int = 7) -> SyntheticPage:
    """Собирает полосу, её эталон и маски объектов.

    Аргументы:
        seed: зерно генератора; одно и то же зерно даёт бит-в-бит одну и ту же полосу.

    Возвращает :class:`SyntheticPage`.
    """
    rng = np.random.default_rng(seed)
    base = paper(rng)
    img = over(base, text_ink(rng), (25, 22, 20), 0.88)
    x0, y0, x1, y1 = PHOTO_BOX
    photo = np.zeros((HEIGHT, WIDTH), np.float32)
    photo[y0:y1, x0:x1] = halftone_dots(x1 - x0, y1 - y0)
    img = over(img, photo, (0, 0, 0), 0.85)
    img[:, PAGE_RIGHT:] = base[:, PAGE_RIGHT:]  # стол без набора
    clean = img.copy()

    masks: "dict[TargetObject, np.ndarray]" = {}
    # Печать поверх текста: маска — точно по краске (порог по покрытию).
    ink = stamp_ink(STAMP_CENTER, STAMP_RADIUS, "ОТДЕЛ ТЕХНИЧЕСКОГО КОНТРОЛЯ ★ ", rng)
    img = over(img, ink, (45, 70, 165), 0.92)
    masks[TargetObject.STAMP] = ink > 0.05
    # Крупная печать: маска — сплошной диск с запасом, как обводит кистью разметчик.
    ink = stamp_ink(BIG_STAMP_CENTER, BIG_STAMP_RADIUS, "БИБЛИОТЕКА ЗАВОДА ИМ. ЛИХАЧЁВА ★ ", rng)
    img = over(img, ink, (110, 60, 150), 0.75)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    bx, by = BIG_STAMP_CENTER
    masks[TargetObject.BIG_STAMP] = (xx - bx) ** 2 + (yy - by) ** 2 < (BIG_STAMP_RADIUS + BIG_STAMP_BRUSH_MARGIN) ** 2
    # Палец: альфа-смешение, маска — где пальца хоть сколько-то.
    skin, alpha = finger_layer()
    img = img * (1 - alpha[..., None]) + skin * alpha[..., None]
    masks[TargetObject.FINGER] = alpha > 0.02
    # Рукопись поверх строки.
    ink = handwriting_ink(HANDWRITING_ORIGIN, rng)
    img = over(img, ink, (70, 70, 80), 0.7)
    masks[TargetObject.HANDWRITING] = ink > 0.05

    return SyntheticPage(
        np.clip(img, 0, 255).astype(np.uint8),
        np.clip(clean, 0, 255).astype(np.uint8),
        {k: m.astype(np.uint8) * 255 for k, m in masks.items()},
    )
