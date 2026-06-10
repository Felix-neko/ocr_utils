#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rawpy",
#     "numpy",
#     "Pillow",
#     "opencv-python-headless",
#     "click",
#     "tqdm",
# ]
# ///
"""Детектор расфокуса для сканов газет (RAF с JPEG-превью или обычные JPEG).

Идея метода `moire` (по умолчанию): газетный полутоновый растр при нормальном
фокусе порождает сильный муар, если уменьшить изображение без сглаживания
(именно так пользователь выявляет расфокус глазами в XNView). Мы измеряем этот
муар как std разницы между уменьшением методом NEAREST (даёт алиасинг/муар) и
методом AREA (усредняет, муара нет). Структура текста и фотографий присутствует
в обоих уменьшениях и сокращается — остаётся только энергия растрового муара.
Растр есть везде, где лежит краска (включая фотографии), поэтому в фокусе даже
фото-тайлы дают муар; низкий муар на ПЕЧАТНОМ тайле означает расфокус. Чисто
белые поля отсеиваются гейтингом по локальному контрасту.

Метрика считается по сетке тайлов, что позволяет ловить и зональный расфокус
(когда размыта лишь часть страницы — например, наклон плоскости фокуса).

ВАЖНО: это инструмент скрининга/ранжирования, а не точный классификатор.
Сильный и средний расфокус (`_defocus`) выявляется надёжно; слабый
(`_light`/`_ultralight`) — на грани и может путаться с текстовыми страницами,
где растра мало. Используйте ранжированный вывод и порог как подсказку.
"""

import io
import sys
from pathlib import Path

import click
import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps
from tqdm import tqdm

# Поддерживаемые расширения входных файлов
RAW_SUFFIXES = {".raf"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}


def read_image_gray(path: Path) -> np.ndarray | None:
    """Загружает изображение в оттенках серого.

    Для RAF извлекает встроенное JPEG-превью (быстро), для JPEG грузит напрямую.
    Изображение поворачивается согласно EXIF.

    Args:
        path: Путь к входному файлу (RAF или JPEG).

    Returns:
        Полутоновый numpy.ndarray (uint8) либо None, если прочитать не удалось.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in RAW_SUFFIXES:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format != rawpy.ThumbFormat.JPEG:
                    return None
                img = Image.open(io.BytesIO(bytes(thumb.data)))
        else:
            img = Image.open(str(path))
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("L"))
    except Exception:
        return None


def _tile_bounds(size: int, n: int, idx: int) -> tuple[int, int]:
    """Возвращает границы тайла [start, end) по оси для равномерной сетки."""
    return idx * size // n, (idx + 1) * size // n


def moire_tile_maps(gray: np.ndarray, factor: float, grid_x: int, grid_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Считает по сетке карту энергии муара и карту контраста (наличия краски).

    Args:
        gray: Полутоновое изображение.
        factor: Во сколько раз уменьшать кадр перед измерением муара.
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.

    Returns:
        Кортеж (moire, structure) — два массива shape (grid_y, grid_x):
        moire — std разницы NEAREST−AREA в тайле (энергия муара),
        structure — std AREA-уменьшения в тайле (мера наличия печатного контента).
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    nw, nh = max(1, int(w / factor)), max(1, int(h / factor))
    nn = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ar = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    diff = nn - ar

    moire = np.zeros((grid_y, grid_x), dtype=np.float64)
    structure = np.zeros((grid_y, grid_x), dtype=np.float64)
    for ry in range(grid_y):
        y1, y2 = _tile_bounds(nh, grid_y, ry)
        for rx in range(grid_x):
            x1, x2 = _tile_bounds(nw, grid_x, rx)
            moire[ry, rx] = diff[y1:y2, x1:x2].std()
            structure[ry, rx] = ar[y1:y2, x1:x2].std()
    return moire, structure


def laplacian_tile_maps(gray: np.ndarray, grid_x: int, grid_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Считает по сетке карту дисперсии Лапласиана и карту контраста.

    Классическая метрика резкости (variance of Laplacian) как альтернативный метод.

    Args:
        gray: Полутоновое изображение.
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.

    Returns:
        Кортеж (sharp, structure) — два массива shape (grid_y, grid_x):
        sharp — дисперсия Лапласиана в тайле,
        structure — std тайла (мера наличия печатного контента).
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    sharp = np.zeros((grid_y, grid_x), dtype=np.float64)
    structure = np.zeros((grid_y, grid_x), dtype=np.float64)
    for ry in range(grid_y):
        y1, y2 = _tile_bounds(h, grid_y, ry)
        for rx in range(grid_x):
            x1, x2 = _tile_bounds(w, grid_x, rx)
            tile = g[y1:y2, x1:x2]
            sharp[ry, rx] = cv2.Laplacian(tile, cv2.CV_32F).var()
            structure[ry, rx] = tile.std()
    return sharp, structure


def analyze(gray: np.ndarray, method: str, factor: float, grid_x: int, grid_y: int, min_structure: float) -> dict:
    """Анализирует изображение и возвращает метрики резкости/расфокуса.

    Args:
        gray: Полутоновое изображение.
        method: "moire" или "laplacian".
        factor: Коэффициент уменьшения для метода moire.
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.
        min_structure: Порог локального контраста (std), ниже которого тайл
            считается пустым полем и исключается из статистики.

    Returns:
        Словарь с ключами:
        sharpness — медиана метрики по печатным тайлам (выше = резче),
        worst_zone — среднее по 10% самых «мягких» печатных тайлов (для зон),
        n_printed — число печатных тайлов,
        sharp_map, structure_map, printed_mask — карты для визуализации.
    """
    if method == "moire":
        sharp_map, structure = moire_tile_maps(gray, factor, grid_x, grid_y)
    else:
        sharp_map, structure = laplacian_tile_maps(gray, grid_x, grid_y)

    printed = structure > min_structure
    # Если печатных тайлов почти нет (титул, пустой лист) — берём верхние по контрасту,
    # чтобы метрика оставалась осмысленной, а не считалась по шуму.
    if printed.sum() < max(6, grid_x * grid_y // 20):
        printed = structure > np.percentile(structure, 60)

    vals = sharp_map[printed]
    if vals.size == 0:
        return dict(
            sharpness=float("nan"),
            worst_zone=float("nan"),
            n_printed=0,
            sharp_map=sharp_map,
            structure_map=structure,
            printed_mask=printed,
        )

    n_worst = max(1, vals.size // 10)
    worst_zone = float(np.sort(vals)[:n_worst].mean())
    return dict(
        sharpness=float(np.median(vals)),
        worst_zone=worst_zone,
        n_printed=int(printed.sum()),
        sharp_map=sharp_map,
        structure_map=structure,
        printed_mask=printed,
    )


def save_heatmap(path: Path, result: dict, vmax: float) -> None:
    """Сохраняет тепловую карту резкости тайлов (для отладки и подбора порога).

    Холодный цвет — мало резкости/муара (подозрение на расфокус), тёплый — резко.
    Серым показаны непечатные тайлы (поля), исключённые из статистики.

    Args:
        path: Куда сохранить PNG.
        result: Результат analyze() с картами.
        vmax: Значение метрики, отображаемое как максимум шкалы.
    """
    sharp_map = result["sharp_map"]
    printed = result["printed_mask"]
    norm = np.clip(sharp_map / max(vmax, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vis[~printed] = (60, 60, 60)
    scale = 24
    vis = cv2.resize(vis, (sharp_map.shape[1] * scale, sharp_map.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), vis)


def collect_inputs(inputs: tuple[str, ...]) -> list[Path]:
    """Разворачивает аргументы в список файлов.

    Директории обходятся нерекурсивно; берутся только поддерживаемые расширения.

    Args:
        inputs: Пути к файлам и/или директориям.

    Returns:
        Отсортированный список путей к файлам.
    """
    supported = RAW_SUFFIXES | JPEG_SUFFIXES
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(c for c in sorted(p.iterdir()) if c.suffix.lower() in supported)
        elif p.is_file():
            files.append(p)
        else:
            click.echo(f"Предупреждение: путь не найден — {p}", err=True)
    return files


@click.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path())
@click.option(
    "--method",
    type=click.Choice(["moire", "laplacian"]),
    default="moire",
    show_default=True,
    help="Метод оценки резкости.",
)
@click.option("--factor", type=float, default=3.0, show_default=True, help="Коэффициент уменьшения для метода moire.")
@click.option("--grid-x", type=int, default=16, show_default=True, help="Число тайлов по горизонтали.")
@click.option("--grid-y", type=int, default=11, show_default=True, help="Число тайлов по вертикали.")
@click.option(
    "--min-structure",
    type=float,
    default=8.0,
    show_default=True,
    help="Порог контраста тайла (std), ниже — считается пустым полем.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Абсолютный порог резкости: файлы ниже помечаются как подозрительные. "
    "По умолчанию для moire=15.0, для laplacian=900.0. Игнорируется при --relative.",
)
@click.option(
    "--relative/--absolute",
    default=True,
    show_default=True,
    help="relative: адаптивный порог относительно выборки (выпуска) — устойчив к "
    "разной бумаге/растру между выпусками. absolute: фиксированный --threshold.",
)
@click.option(
    "--z",
    type=float,
    default=2.0,
    show_default=True,
    help="Для --relative: на сколько робастных сигм (MAD) ниже медианы выборки "
    "считать файл подозрительным. Меньше — больше срабатываний.",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Каталог для сохранения тепловых карт тайлов (отладка).",
)
@click.option("--quiet", is_flag=True, help="Печатать только подозрительные файлы.")
def main(
    inputs: tuple[str, ...],
    method: str,
    factor: float,
    grid_x: int,
    grid_y: int,
    min_structure: float,
    threshold: float | None,
    relative: bool,
    z: float,
    debug_dir: str | None,
    quiet: bool,
) -> None:
    """Ищет расфокусные сканы среди INPUTS (файлы и/или одна директория, нерекурсивно).

    Печатает таблицу, отсортированную по возрастанию резкости (самые подозрительные
    сверху): резкость (выше = резче), «худшая зона» (для зонального расфокуса) и
    силу подозрения. Чем ниже резкость относительно порога — тем сильнее подозрение.

    Базовый уровень муара сильно различается между выпусками (бумага, растр), поэтому
    по умолчанию используется адаптивный порог (--relative): подозрительными считаются
    низкие выбросы резкости внутри самой выборки. Запускайте по одному выпуску.
    """
    if threshold is None:
        threshold = 15.0 if method == "moire" else 900.0

    files = collect_inputs(inputs)
    if not files:
        click.echo("Нет входных файлов.", err=True)
        sys.exit(1)

    debug_path = None
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    # vmax для шкалы тепловых карт: «нормальная» резкость ≈ порог * 1.5
    vmax = threshold * 1.5

    rows = []
    for f in tqdm(files, desc="Анализ", unit="файл"):
        gray = read_image_gray(f)
        if gray is None:
            click.echo(f"Предупреждение: не удалось прочитать — {f.name}", err=True)
            continue
        res = analyze(gray, method, factor, grid_x, grid_y, min_structure)
        rows.append((f, res))
        if debug_path is not None:
            save_heatmap(debug_path / f"{f.stem}_heat.png", res, vmax)

    # Сортируем по возрастанию резкости — самые подозрительные сверху
    rows.sort(key=lambda r: (np.nan_to_num(r[1]["sharpness"], nan=1e18)))

    sharp_vals = np.array([r[1]["sharpness"] for r in rows if not np.isnan(r[1]["sharpness"])])

    # Определяем порог и функцию силы подозрения
    if relative and sharp_vals.size >= 4:
        med = float(np.median(sharp_vals))
        mad = float(np.median(np.abs(sharp_vals - med))) * 1.4826
        mad = max(mad, 1e-6)
        eff_threshold = med - z * mad
        header = f"Порог (relative): медиана {med:.2f} − {z:g}·MAD = {eff_threshold:.2f}"

        def strength_of(s: float) -> float:
            return float(max(0.0, (med - s) / mad))  # в робастных сигмах

        strength_label = "сигм"
    else:
        eff_threshold = threshold

        def strength_of(s: float) -> float:
            return float(np.clip((eff_threshold - s) / eff_threshold, 0.0, 1.0))

        strength_label = "0..1"
        header = f"Порог (absolute): {eff_threshold:g}"

    click.echo("")
    click.echo(f"Метод: {method}   {header}   (резкость выше = резче)")
    click.echo(f"{'резкость':>10} {'худш.зона':>10} {'подозр,' + strength_label:>10}  файл")
    click.echo("-" * 62)

    suspects = []
    for f, res in rows:
        sharp = res["sharpness"]
        worst = res["worst_zone"]
        if np.isnan(sharp):
            strength = 0.0
            is_suspect = False
        else:
            strength = strength_of(sharp)
            is_suspect = sharp < eff_threshold
        if is_suspect:
            suspects.append((f, sharp, strength))
        if quiet and not is_suspect:
            continue
        mark = "  <-- РАСФОКУС?" if is_suspect else ""
        click.echo(f"{sharp:10.2f} {worst:10.2f} {strength:10.2f}  {f.name}{mark}")

    click.echo("-" * 62)
    click.echo(f"Подозрительных: {len(suspects)} из {len(rows)}")


if __name__ == "__main__":
    main()
