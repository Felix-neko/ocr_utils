"""Снятие следов шариковой ручки со скана линейной комбинацией каналов.

ЗАЧЕМ. На части полос поверх текста есть пометки синей пастой — их оставили в библиотеке
или при подготовке скана. Это дефект ИСХОДНИКА, а не разметки: в бинаризованный PDF пометка
попадёт ровно так же, чем бы её ни пометили в CVAT, и текст под ней будет потерян.

КАКУЮ КОМБИНАЦИЮ БРАТЬ. Одного канала мало. Замер по 1968/01 IMG_0045_2R, средние BGR по
четырём классам пикселей (бумага, печатный текст, клякса без текста, текст ПОД кляксой):

    бумага              B 253.2  G 247.2  R 250.3
    печатный текст      B  42.9  G  34.0  R  37.3
    клякса              B 161.5  G  86.7  R  58.9
    текст под кляксой   B  91.2  G  24.2  R  12.6

Задача двухсторонняя: клякса должна уйти к уровню бумаги, а текст под ней — остаться
тёмным. Одиночные каналы обе стороны сразу не берут (уровни после приведения «бумага 245,
печатный текст 30»):

    только синий                  клякса 151  текст под кляксой  79   клякса ещё видна
    только красный                клякса   0                          клякса чернее текста
    B - 0.47R (точное гашение)    клякса 243  текст под кляксой 148   текст под ней выцвел
    0.9B + 0.8G - R               клякса 217  текст под кляксой 118   компромисс

Последняя строка и взята умолчанием: она найдена перебором по сетке весов при двух
ограничениях — клякса не темнее 215 и шум бумаги после преобразования не выше 9 уровней.
Ограничение по шуму существенно: комбинации вида G - 0.8R разделяют классы по средним ещё
лучше, но вычитают друг из друга два похожих канала, и бумага после них идёт пятнами.

Знак у красного отрицательный не случайно: синяя паста красный свет поглощает, поэтому в
красном канале она чернее печатного текста, и вычитание красного её и высветляет.

ПОЧЕМУ ТОЛЬКО ВОКРУГ КЛЯКСЫ. Комбинация каналов — не улучшение скана, а размен: она поднимает
шум и меняет тональность. Платить этим за всю полосу ради пятна в сантиметр незачем, поэтому
пересчитанное подмешивается только внутри маски кляксы с растушёванным краем, а остальной
кадр остаётся прежним.

Результат кладётся в ОТДЕЛЬНУЮ папку, а не рядом с исходником: лишний файл в паке стал бы
лишней полосой в выпуске и сдвинул бы порядок остальных.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr_utils.scan_markup.detection.color_kind import balanced_lab, paper_color

logger = logging.getLogger(__name__)

# Веса линейной комбинации в порядке BGR (как их отдаёт OpenCV).
DEFAULT_WEIGHTS = (0.9, 0.8, -1.0)

# Уровни, к которым приводится результат. Совпадают с уровнями оригинала, иначе на границе
# подмешивания была бы видна ступенька.
TARGET_PAPER = 245.0
TARGET_TEXT = 30.0

# Разрешение, при котором заданы размеры в пикселях ниже.
REFERENCE_DPI = 600

# Хроматичность в Lab (после снятия налёта бумаги), с которой пиксель считается пастой.
# Втрое выше порога различимости оттенка: пасту ни с чем не спутать, а у растровой
# фотографии остаточная хроматичность бывает заметной, и трогать её нельзя.
PEN_CHROMA_THR = 30.0

# Размыкание маски: убирает одиночные хроматичные пиксели (ложный цвет демозаика на мелком
# тексте и на растровой сетке).
PEN_OPEN_PX = 9

# Насколько раздуть маску. 48 px при 600 dpi — 2 мм: паста по краям бледнеет и в порог не
# попадает, а текст рядом с кляксой должен пересчитываться вместе с ней.
PEN_DILATE_PX = 48

# Растушёвка края маски, чтобы подмешивание не дало видимого шва.
PEN_FEATHER_PX = 24

# Минимальная площадь пятна пасты в пикселях при 600 dpi (примерно 2.7 мм^2). Мельче бывает
# только цветной шум, и раздувать его на два миллиметра точно не надо.
PEN_MIN_AREA_PX = 4000

# Сжатие результата. Оригиналы пака — TIFF LZW; без указания PIL записал бы несжатый файл
# втрое тяжелее.
COMPRESSION = "tiff_lzw"


@dataclass
class FixResult:
    """Итог по одному файлу."""

    rel_path: str
    dst: Path | None
    pen_area_px: int = 0
    error: str = ""


def _scaled(value: int, dpi: int | None) -> int:
    """Длина в пикселях, пересчитанная с ``REFERENCE_DPI`` на разрешение полосы."""
    scale = 1.0 if not dpi else float(dpi) / REFERENCE_DPI
    return max(1, round(value * scale))


def _scaled_area(value: int, dpi: int | None) -> int:
    """То же для площади — она растёт квадратом отношения разрешений."""
    scale = 1.0 if not dpi else float(dpi) / REFERENCE_DPI
    return max(1, round(value * scale * scale))


def _odd(value: int) -> int:
    """Ближайшее нечётное не меньше 1 — размер ядра размытия должен быть нечётным."""
    return value if value % 2 else value + 1


def pen_mask(bgr: np.ndarray, dpi: int | None = REFERENCE_DPI, chroma_thr: float = PEN_CHROMA_THR) -> np.ndarray:
    """Маска пятен пасты (0/255), уже раздутая с запасом.

    Хроматичность считается после снятия налёта бумаги тем же кодом, что и в классификации
    областей (``color_kind.balanced_lab``): сканы жёлто-бежевые, и по сырой насыщенности
    пастой оказалась бы вся полоса.
    """
    a, b = balanced_lab(bgr, paper_color(bgr))
    mask = (np.hypot(a, b) > chroma_thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((_scaled(PEN_OPEN_PX, dpi),) * 2, np.uint8))

    # Мелочь отсеивается ДО раздувания: иначе крапина в двести точек раздулась бы в пятно
    # диаметром четыре миллиметра.
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    min_area = _scaled_area(PEN_MIN_AREA_PX, dpi)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    mask = keep[labels].astype(np.uint8)

    dilate = _scaled(PEN_DILATE_PX, dpi)
    if dilate > 1:
        mask = cv2.dilate(mask, np.ones((dilate,) * 2, np.uint8))
    return mask * 255


def recombine(bgr: np.ndarray, weights=DEFAULT_WEIGHTS) -> np.ndarray:
    """Линейная комбинация каналов, приведённая к уровням бумаги и текста ЭТОЙ полосы.

    Приведение по перцентилям самой полосы, а не по константам из замера: у соседних сканов
    уровень бумаги гуляет на десятки единиц, и жёсткая шкала дала бы на шве ступеньку.
    """
    weights = np.asarray(weights, np.float32)
    projected = bgr.astype(np.float32) @ weights

    paper_level = float(np.percentile(projected, 97))
    text_level = float(np.percentile(projected, 2))
    span = paper_level - text_level
    if span <= 1e-3:  # вырожденный кадр — приводить не к чему
        return cv2.cvtColor(np.clip(projected, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    scale = (TARGET_PAPER - TARGET_TEXT) / span
    leveled = np.clip((projected - text_level) * scale + TARGET_TEXT, 0, 255).astype(np.uint8)
    return cv2.cvtColor(leveled, cv2.COLOR_GRAY2BGR)


def remove_pen_marks(
    bgr: np.ndarray, dpi: int | None = REFERENCE_DPI, weights=DEFAULT_WEIGHTS, chroma_thr: float = PEN_CHROMA_THR
) -> tuple[np.ndarray, int]:
    """Полоса с погашенной пастой и площадь затронутой области в пикселях.

    Пасты на полосе нет — возвращается исходный кадр и ноль; такой файл можно не писать.
    """
    mask = pen_mask(bgr, dpi, chroma_thr)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return bgr, 0

    feather = _odd(_scaled(PEN_FEATHER_PX, dpi))
    alpha = (cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0)[..., None]
    mixed = bgr.astype(np.float32) * (1.0 - alpha) + recombine(bgr, weights).astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8), area


def fix_page(
    pack_dir: Path, rel_path: str, out_dir: Path, weights=DEFAULT_WEIGHTS, chroma_thr: float = PEN_CHROMA_THR
) -> FixResult:
    """Пишет починенную полосу в ``out_dir/<тот же относительный путь>``."""
    src = pack_dir / rel_path
    dst = out_dir / rel_path
    try:
        with Image.open(src) as image:
            dpi_tag = image.info.get("dpi")
            rgb = np.asarray(image.convert("RGB"))
        dpi = int(dpi_tag[0]) if dpi_tag else None
        if dpi is None:
            logger.warning("У %s нет тега разрешения — считаю размеры при %d dpi", rel_path, REFERENCE_DPI)

        bgr = np.ascontiguousarray(rgb[..., ::-1])
        fixed, area = remove_pen_marks(bgr, dpi, weights, chroma_thr)
        if area == 0:
            return FixResult(rel_path, None, 0, "пасты не нашлось, файл не записан")

        dst.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"compression": COMPRESSION}
        if dpi_tag:
            save_kwargs["dpi"] = dpi_tag
        Image.fromarray(np.ascontiguousarray(fixed[..., ::-1])).save(dst, **save_kwargs)
        return FixResult(rel_path, dst, area)
    except Exception as exc:  # noqa: BLE001 — один битый файл не должен валить список
        return FixResult(rel_path, None, 0, str(exc))


def fix_pages(
    pack_dir: Path, rel_paths: list[str], out_dir: Path, weights=DEFAULT_WEIGHTS, chroma_thr: float = PEN_CHROMA_THR
) -> list[FixResult]:
    """То же по списку полос. Пул не нужен: таких файлов единицы."""
    return [fix_page(pack_dir, rel_path, out_dir, weights, chroma_thr) for rel_path in rel_paths]
