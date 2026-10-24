"""Контактные листы для отсмотра глазами: миниатюры папки сеткой с именами кадров и сайдкаров, пара кадров рядом.

``cv2.putText`` кириллицу не рисует, поэтому подписи — через PIL (DejaVu Sans).
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.frame_flags.flags import FlagsError, read_flags

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Высота полосы подписи под миниатюрой, px.
LABEL_H = 30
# Промежуток между ячейками и поле листа, px.
GAP = 8
# Цвет фона листа и подписи (BGR): тёмно-серый фон, светлый текст, флаги — оранжевым.
SHEET_BG = (40, 40, 40)
TEXT_COLOR = (235, 235, 235)
FLAG_COLOR = (0, 165, 255)


def _font(size: int) -> ImageFont.ImageFont:
    """Шрифт подписей заданного кегля; без DejaVu — встроенный шрифт PIL."""
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _put_text(canvas: np.ndarray, xy: tuple[int, int], text: str, color: tuple[int, int, int], size: int) -> None:
    """Пишет строку на BGR-холст через PIL (кириллица) на месте.

    Args:
        canvas: BGR-холст; меняется на месте.
        xy: Левый верхний угол строки, px.
        text: Текст подписи.
        color: Цвет BGR.
        size: Кегль, px.
    """
    x, y = xy
    h, w = canvas.shape[:2]
    # Рисуем только полосу строки, чтобы не гонять весь лист через PIL.
    y1 = min(h, y + size + 6)
    if y >= y1 or x >= w:
        return
    strip = Image.fromarray(cv2.cvtColor(canvas[y:y1, x:], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(strip).text((0, 0), text, font=_font(size), fill=(color[2], color[1], color[0]))
    canvas[y:y1, x:] = cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)


def cell_height(side: int) -> int:
    """Высота ячейки миниатюры при ширине ``side``: 3/4 ширины — кадры разворотов альбомные 4:3."""
    return side * 3 // 4


def thumbnail(path: Path, side: int) -> np.ndarray:
    """Миниатюра кадра, вписанная в ячейку ``side`` × ``cell_height(side)`` (по центру, поля — цвет фона).

    Args:
        path: Файл кадра.
        side: Ширина ячейки, px.

    Returns:
        BGR-миниатюра размера ячейки; нечитаемый кадр — пустая ячейка.
    """
    cell_h = cell_height(side)
    cell = np.full((cell_h, side, 3), SHEET_BG, np.uint8)
    # Чтение с уменьшением в 4 раза в декодере JPEG — в разы быстрее полного чтения.
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return cell
    h, w = img.shape[:2]
    scale = min(side / w, cell_h / h)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    y0, x0 = (cell_h - sh) // 2, (side - sw) // 2
    cell[y0 : y0 + sh, x0 : x0 + sw] = small
    return cell


def flags_label(path: Path) -> str:
    """Флаги кадра строкой для подписи; битый сайдкар — ``!сайдкар``."""
    try:
        return read_flags(path).label()
    except FlagsError:
        return "!сайдкар"


def contact_sheet(paths: list[Path], out: Path, columns: int, side: int) -> Path:
    """Один лист миниатюр сеткой: под каждой — имя кадра и флаги его сайдкара.

    Args:
        paths: Кадры листа по порядку.
        out: Путь JPEG листа; папка создаётся.
        columns: Число столбцов сетки.
        side: Ширина миниатюры, px.

    Returns:
        Путь записанного листа.
    """
    rows = (len(paths) + columns - 1) // columns
    cell_h = cell_height(side)
    width = GAP + columns * (side + GAP)
    height = GAP + rows * (cell_h + LABEL_H + GAP)
    sheet = np.full((height, width, 3), SHEET_BG, np.uint8)
    for i, path in enumerate(paths):
        r, c = divmod(i, columns)
        x, y = GAP + c * (side + GAP), GAP + r * (cell_h + LABEL_H + GAP)
        sheet[y : y + cell_h, x : x + side] = thumbnail(path, side)
        _put_text(sheet, (x, y + cell_h + 4), path.name, TEXT_COLOR, 18)
        label = flags_label(path)
        if label:
            # Флаги — справа от имени, другим цветом.
            _put_text(sheet, (x + side // 2, y + cell_h + 4), label, FLAG_COLOR, 16)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def contact_sheets(
    paths: list[Path], out_dir: Path, per_sheet: int = 20, columns: int = 5, side: int = 420
) -> list[Path]:
    """Листы миниатюр по ``per_sheet`` кадров: ``sheet_01_IMG_0001-IMG_0020.jpg`` и т. д.

    Args:
        paths: Кадры папки по порядку.
        out_dir: Папка листов (вне Я.Диска).
        per_sheet: Кадров на лист.
        columns: Число столбцов сетки.
        side: Ширина миниатюры, px.

    Returns:
        Пути листов по порядку.
    """
    sheets = []
    for n, start in enumerate(range(0, len(paths), per_sheet), 1):
        chunk = paths[start : start + per_sheet]
        name = f"sheet_{n:02d}_{chunk[0].stem}-{chunk[-1].stem}.jpg"
        sheets.append(contact_sheet(chunk, out_dir / name, columns, side))
    return sheets


def pair_sheet(path_a: Path, path_b: Path, out: Path, width: int = 1600) -> Path:
    """Два кадра рядом, каждый шириной ``width`` px, с подписью над каждым.

    Нужен, чтобы по краю листа и строкам решить, какая страница в каком кадре поднята или
    закрыта карточкой.

    Args:
        path_a: Первый кадр (слева).
        path_b: Второй кадр (справа).
        out: Путь JPEG листа; папка создаётся.
        width: Ширина каждого кадра на листе, px.

    Returns:
        Путь записанного листа.
    """
    images = []
    for path in (path_a, path_b):
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            img = np.full((width, width, 3), SHEET_BG, np.uint8)
        h, w = img.shape[:2]
        images.append(cv2.resize(img, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA))
    height = max(i.shape[0] for i in images)
    sheet = np.full((LABEL_H + 10 + height, 2 * width + GAP, 3), SHEET_BG, np.uint8)
    for k, (path, img) in enumerate(zip((path_a, path_b), images)):
        x = k * (width + GAP)
        sheet[LABEL_H + 10 : LABEL_H + 10 + img.shape[0], x : x + width] = img
        label = flags_label(path)
        _put_text(sheet, (x + 6, 4), path.name + (f"   [{label}]" if label else ""), TEXT_COLOR, 22)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out
