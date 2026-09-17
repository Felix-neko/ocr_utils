"""Синтетические «страницы» для тестов: строки текста и линейки на белом, известные искажения."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WORDS = (
    "материально техническое снабжение народного хозяйства и планирование поставок металла "
    "запасы предприятий склады базы нормы расход оборачиваемость тонн рублей процентов год"
).split()


def text_page(width: int = 2000, height: int = 3000, lines: int = 40, font_px: int = 40, x0: int = 200) -> np.ndarray:
    """Серая страница 300 dpi: колонка строк настоящими буквами (DejaVu), строки через 1.6 кегля.

    Слова в строках перемешаны: одинаковые строки через равный шаг — периодика, и тайлы поля
    браковались бы как неотчётливые (на настоящей странице такого не бывает).
    """
    rng = np.random.default_rng(42)
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, font_px)
    except OSError:
        font = ImageFont.load_default()
    y = 300
    for i in range(lines):
        words = rng.choice(WORDS, size=int(rng.integers(6, 10)))
        draw.text((x0, y), " ".join(words)[:72], fill=0, font=font)
        y += int(font_px * 1.6)
    return np.asarray(image)


def add_rules(page: np.ndarray, rules: list[tuple[int, int, int, int]], thickness: int = 4) -> np.ndarray:
    out = page.copy()
    for x0, y0, x1, y1 in rules:
        cv2.line(out, (x0, y0), (x1, y1), 0, thickness, cv2.LINE_AA)
    return out


def rotate(page: np.ndarray, angle_deg: float, shift=(0.0, 0.0)) -> np.ndarray:
    """Поворот вокруг центра (ось y вниз: положительный угол — по часовой на экране) плюс сдвиг."""
    h, w = page.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
    matrix[:, 2] += shift
    return cv2.warpAffine(page, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def wave_region(page: np.ndarray, y0: int, y1: int, amplitude_px: float, period_px: float) -> np.ndarray:
    """Вертикальная волна внутри полосы строк y0..y1 — «погнутая» область."""
    h, w = page.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    inside = (ys >= y0) & (ys < y1)
    ys_src = ys + inside * amplitude_px * np.sin(2 * np.pi * xs / period_px)
    return cv2.remap(page, xs, ys_src.astype(np.float32), cv2.INTER_LINEAR, borderValue=255)


def binarize(page: np.ndarray) -> np.ndarray:
    return np.where(page < 128, 0, 255).astype(np.uint8)
