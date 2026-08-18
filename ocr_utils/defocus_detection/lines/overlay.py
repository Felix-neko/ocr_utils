"""Отладочные наложения: что именно детектор нашёл и что по этому намерено.

ЗАЧЕМ. Оценка фокуса по строкам зависит от чужой модели (surya) и от полудюжины порогов.
Ни то, ни другое нельзя проверить по числу в таблице: колонка «σ 0.83» одинаково выглядит
и когда замер идёт по корпусному набору, и когда детектор нашёл одни заголовки, и когда
куски строк уехали на бумагу. Единственный способ убедиться — посмотреть глазами, поэтому
``--debug-dir`` выгружает по кадру JPEG, на котором видно всё сразу:

    * границы тайлов зональной сетки;
    * рамку каждой измеренной области строки, покрашенную по её резкости
      (зелёная — резче медианы кадра, красная — мягче, серая — не измерена);
    * число внутри рамки — резкость этой строки в единицах метрики;
    * в углу каждого тайла — его балл и число строк, по которым он получен;
    * в углу кадра — общий балл файла, нормированный балл и зональный перепад.

Наложение делается на уменьшенной копии: разглядывать надо раскладку и цвета, а не зерно.

ПОДПИСИ РИСУЮТСЯ ЧЕРЕЗ PIL, А НЕ ЧЕРЕЗ ``cv2.putText``. У OpenCV только векторные шрифты
Hershey, в которых нет кириллицы: «правый нижний угол» превращается в ряд знаков вопроса,
и самая полезная часть наложения пропадает.

ЧИСЛА ПО СТРОКАМ ПРОРЕЖИВАЮТСЯ. На газетной полосе строк за тысячу, и подписать их все —
значит залить кадр текстом так, что не видно ни самой полосы, ни цветов рамок. Поэтому
числом подписывается ВЫБОРКА, разложенная по тайлам поровну (см. ``LABELS_PER_TILE``):
цифры остаются читаемыми и при этом покрывают весь кадр, а сплошную картину даёт цвет.
"""

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.defocus_detection.lines.measure import LineMeasurements
from ocr_utils.defocus_detection.lines.zonal_tiles import TileZonalResult

# Длинная сторона наложения. 2200 px хватает, чтобы различить отдельные строки корпуса на
# газетной полосе, и при этом JPEG весит меньше мегабайта.
OVERLAY_MAX_SIDE = 2200
# Сколько строк подписывать числом в каждом тайле зональной сетки.
LABELS_PER_TILE = 12

# Пути к шрифту с кириллицей, в порядке предпочтения. Ставится вместе с любым десктопом;
# если не нашёлся ни один, подписи рисуются встроенным растровым шрифтом PIL (латиница).
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)

# Цвета RGB (наложение рисуется в RGB и переводится в BGR только на записи).
COLOUR_SHARP = (60, 170, 60)
COLOUR_SOFT = (225, 55, 55)
COLOUR_UNMEASURED = (170, 170, 170)
COLOUR_TILE = (30, 140, 210)
COLOUR_TEXT = (20, 20, 20)
COLOUR_PANEL = (255, 255, 255)


def _load_font(size: int):
    """Загружает шрифт с кириллицей.

    Args:
        size: Кегль в пикселях.

    Returns:
        Шрифт PIL; встроенный растровый, если ни один TTF не нашёлся.
    """
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _display(value: float, algorithm) -> float:
    """Переводит балл резкости в человекочитаемую величину метрики.

    Args:
        value: Балл резкости (больше = резче).
        algorithm: Алгоритм — у него спрашивается способ показа.

    Returns:
        Число для подписи (для ширины края это σ в пикселях).
    """
    if algorithm.display is None or not np.isfinite(value):
        return value
    return algorithm.display(value)


def _put_label(draw, font, text: str, origin: tuple[int, int]) -> None:
    """Пишет подпись с подложкой, чтобы её было видно и на тексте, и на бумаге.

    Args:
        draw: ``ImageDraw`` холста.
        font: Шрифт PIL.
        text: Текст подписи.
        origin: Левый верхний угол текста.
    """
    x, y = origin
    box = draw.textbbox((x, y), text, font=font)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=COLOUR_PANEL)
    draw.text((x, y), text, font=font, fill=COLOUR_TEXT)


def _label_sample(measurements: LineMeasurements, n_tiles: int, per_tile: int) -> set[int]:
    """Выбирает, какие строки подписать числом: поровну от каждого тайла.

    Равномерность по тайлам важнее случайности: подпись нужна, чтобы сверить число с тем,
    что видно глазом, и делать это надо в разных частях кадра, а не там, где гуще текст.

    Args:
        measurements: Замеры по строкам.
        n_tiles: Сторона зональной сетки.
        per_tile: Сколько строк подписывать в каждом тайле.

    Returns:
        Множество индексов строк.
    """
    chosen: set[int] = set()
    valid = np.flatnonzero(measurements.valid)
    for tile in range(n_tiles * n_tiles):
        here = valid[measurements.tile_index[valid] == tile]
        if here.size == 0:
            continue
        step = max(1, here.size // per_tile)
        chosen.update(int(i) for i in here[::step][:per_tile])
    return chosen


def draw_overlay(
    gray: np.ndarray,
    measurements: LineMeasurements,
    zonal: TileZonalResult | None,
    algorithm,
    score: float,
    score_norm: float,
    n_tiles: int,
    max_side: int = OVERLAY_MAX_SIDE,
) -> np.ndarray:
    """Рисует отладочное наложение по одному кадру.

    Args:
        gray: Полутоновый кадр полного разрешения.
        measurements: Замеры по строкам.
        zonal: Зональная карта или None.
        algorithm: Алгоритм оценки резкости (нужен для единиц подписи).
        score: Общий балл файла.
        score_norm: Нормированный балл (читаемость).
        n_tiles: Сторона зональной сетки.
        max_side: Длинная сторона наложения.

    Returns:
        Изображение RGB с наложением.
    """
    from PIL import Image as PILImage
    from PIL import ImageDraw

    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    image = PILImage.fromarray(small).convert("RGB")
    draw = ImageDraw.Draw(image)
    cw, ch = image.size

    line_font = _load_font(11)
    tile_font = _load_font(20)
    summary_font = _load_font(22)

    values = measurements.sharpness[measurements.valid]
    median = float(np.median(values)) if values.size else float("nan")
    labelled = _label_sample(measurements, n_tiles, LABELS_PER_TILE)

    for index, (region, value) in enumerate(zip(measurements.regions, measurements.sharpness)):
        polygon = [(float(x) * scale, float(y) * scale) for x, y in np.asarray(region.polygon).reshape(4, 2)]
        if not np.isfinite(value):
            # Неизмеренная строка — тонкой серой рамкой: видно, что детектор её нашёл,
            # но в баллы она не пошла.
            draw.polygon(polygon, outline=COLOUR_UNMEASURED)
            continue
        draw.polygon(polygon, outline=COLOUR_SHARP if value >= median else COLOUR_SOFT)
        if index in labelled:
            x = min(p[0] for p in polygon)
            y = min(p[1] for p in polygon)
            _put_label(draw, line_font, f"{_display(value, algorithm):.2f}", (int(x), max(0, int(y) - 13)))

    for index in range(1, n_tiles):
        x = cw * index // n_tiles
        y = ch * index // n_tiles
        draw.line((x, 0, x, ch), fill=COLOUR_TILE, width=2)
        draw.line((0, y, cw, y), fill=COLOUR_TILE, width=2)

    if zonal is not None:
        for iy in range(zonal.n):
            for ix in range(zonal.n):
                tile_value = zonal.sharpness[iy, ix]
                text = "нет" if not np.isfinite(tile_value) else f"{_display(tile_value, algorithm):.3f}"
                # Подпись в НИЖНЕМ левом углу тайла: верхний занят сводкой в первом ряду,
                # а нижний свободен во всех.
                _put_label(
                    draw,
                    tile_font,
                    f"[{text}] строк {zonal.counts[iy, ix]}",
                    (cw * ix // zonal.n + 8, ch * (iy + 1) // zonal.n - 30),
                )

    summary = [f"{algorithm.name}: {_display(score, algorithm):.3f} {algorithm.display_unit or algorithm.unit}"]
    if np.isfinite(score_norm) and score_norm > 0:
        summary.append(f"читаемость: {1.0 / score_norm:.4f} σ/высота строки")
    if zonal is not None:
        summary.append(f"зона: +{zonal.drop * 100:.0f}% — {zonal.where()}")
    summary.append(f"строк измерено {int(measurements.valid.sum())} из {measurements.n_lines_detected} найденных")
    for row, text in enumerate(summary):
        _put_label(draw, summary_font, text, (8, 8 + row * 28))
    return np.asarray(image)


def write_overlay(path: Path, canvas: np.ndarray) -> None:
    """Сохраняет наложение в JPEG.

    Args:
        path: Куда писать.
        canvas: Изображение RGB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
