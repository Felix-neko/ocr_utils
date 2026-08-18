"""Замер: находит ли surya-ocr КОРПУСНЫЕ строки на газетной полосе.

ЗАЧЕМ. Прежде чем строить оценку фокуса по областям строк, надо убедиться, что эти
области вообще находятся. Риск конкретный: ``DetectionPredictor.prepare_image`` ужимает
вход до размера своего процессора (порядка 1024 px по стороне), а ``split_image`` режет
кадр ТОЛЬКО по высоте — ширину не режет вовсе. Значит ширина превью (2944 px у
портретного скана) давится примерно втрое, и корпусная строка газеты высотой ~17 px
приходит в сеть высотой в пять пикселей. Если детектор в таких условиях видит одни
заголовки, вся затея меряет не тот текст, а в таблице отчёта это будет выглядеть
совершенно нормальным числом.

Скрипт прогоняет одни и те же полосы в двух режимах и даёт сравнить их по числу
найденных строк и по РАСПРЕДЕЛЕНИЮ ИХ ВЫСОТ (главная цифра здесь — медиана: если она
близка к высоте заголовка, а не корпуса, режим негоден):

    page   — полоса подаётся целиком (при желании уменьшенная до --page-max-side);
    tiles  — полоса режется на перекрывающиеся тайлы в НАТИВНОМ разрешении, каждый
             детектируется отдельно, координаты склеиваются обратно.

Результат — печатная сводка и наложения с рамками строк в --out.

Запуск:
    uv run python scripts/check_surya_line_detection.py <папка или файлы> --out /tmp/surya_lines
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr_utils.defocus_detection.image_io import collect_images, read_gray

# Сторона тайла в режиме tiles. Полтора килопикселя — компромисс: процессор детектора
# ужимает вход примерно до 1024 px, поэтому при таком тайле ужатие всего в полтора раза
# (против четырёх у целой полосы), а число проходов по кадру 4416x2944 остаётся 3x2.
DEFAULT_TILE_SIDE = 1500
# Перекрытие тайлов. Строка корпуса длиной в колонку — сотни пикселей, и та, что попала
# на стык, должна целиком уместиться хотя бы в одном тайле, иначе её обрубит пополам.
DEFAULT_TILE_OVERLAP = 300


@dataclass
class Detection:
    """Результат детекции по одному кадру.

    Attributes:
        polygons: Полигоны строк (N, 4, 2) в координатах ПОЛНОГО кадра.
        seconds: Сколько заняла детекция вместе с подготовкой входа.
        passes: Сколько раз кадр (или его часть) прошёл через сеть.
    """

    polygons: list[np.ndarray]
    seconds: float
    passes: int

    def heights(self) -> np.ndarray:
        """Высоты найденных строк в пикселях полного кадра.

        Returns:
            Массив высот; пустой, если строк не нашлось.
        """
        if not self.polygons:
            return np.zeros(0, dtype=np.float64)
        return np.array([p[:, 1].max() - p[:, 1].min() for p in self.polygons], dtype=np.float64)


def load_predictor():
    """Ленивая загрузка surya-детектора строк.

    Импорт вынесен в функцию, потому что он не быстрый, а скрипту с ``--help`` он не нужен.

    Returns:
        Готовый ``DetectionPredictor`` с отключённым собственным прогресс-баром.
    """
    from surya.detection import DetectionPredictor

    predictor = DetectionPredictor()
    predictor.disable_tqdm = True
    return predictor


def _to_pil(gray: np.ndarray):
    """Полутоновый массив -> RGB-картинка Pillow для surya.

    Args:
        gray: Полутоновый кадр uint8.

    Returns:
        Изображение Pillow в режиме RGB.
    """
    from PIL import Image as PILImage

    return PILImage.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def detect_page(predictor, gray: np.ndarray, max_side: int | None, min_conf: float) -> Detection:
    """Детекция по кадру целиком.

    Args:
        predictor: Загруженный ``DetectionPredictor``.
        gray: Полутоновый кадр полного разрешения.
        max_side: До какой длинной стороны уменьшить кадр перед подачей; None — не уменьшать.
        min_conf: Порог уверенности блока.

    Returns:
        Detection с полигонами в координатах полного кадра.
    """
    h, w = gray.shape
    scale = 1.0
    small = gray
    if max_side:
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    started = time.perf_counter()
    result = predictor([_to_pil(small)])[0]
    elapsed = time.perf_counter() - started

    polygons = []
    for box in result.bboxes:
        if box.confidence is not None and box.confidence < min_conf:
            continue
        polygons.append(np.asarray(box.polygon, dtype=np.float64) / scale)
    return Detection(polygons=polygons, seconds=elapsed, passes=1)


def _tile_origins(size: int, side: int, overlap: int) -> list[int]:
    """Начала тайлов вдоль одной оси с заданным перекрытием.

    Последний тайл прижимается к дальнему краю, поэтому его перекрытие с предыдущим
    может оказаться больше запрошенного — это лучше, чем узкая полоска в конце.

    Args:
        size: Длина оси в пикселях.
        side: Сторона тайла.
        overlap: Желаемое перекрытие соседей.

    Returns:
        Список координат начала тайлов.
    """
    if size <= side:
        return [0]
    step = max(1, side - overlap)
    origins = list(range(0, size - side, step))
    origins.append(size - side)
    return origins


def detect_tiles(
    predictor, gray: np.ndarray, side: int, overlap: int, min_conf: float, batch_size: int | None
) -> Detection:
    """Детекция по перекрывающимся тайлам в нативном разрешении.

    Строки, попавшие в перекрытие, нашлись бы дважды. Дубли снимаются не по IoU, а
    геометрически: тайл принимает только те строки, ЦЕНТР которых лежит в его «ядре» —
    тайле, урезанном на половину перекрытия с тех сторон, где есть сосед. Ядра соседей
    не пересекаются и покрывают кадр целиком, поэтому каждая строка достаётся ровно
    одному тайлу, и порогов подбирать не приходится.

    Args:
        predictor: Загруженный ``DetectionPredictor``.
        gray: Полутоновый кадр полного разрешения.
        side: Сторона тайла в пикселях.
        overlap: Перекрытие соседних тайлов.
        min_conf: Порог уверенности блока.
        batch_size: Размер батча для surya; None — её собственный выбор.

    Returns:
        Detection с полигонами в координатах полного кадра.
    """
    h, w = gray.shape
    side = min(side, h, w)
    xs = _tile_origins(w, side, overlap)
    ys = _tile_origins(h, side, overlap)

    crops, cores = [], []
    half = overlap // 2
    for y0 in ys:
        for x0 in xs:
            crops.append(_to_pil(gray[y0 : y0 + side, x0 : x0 + side]))
            # Ядро урезается только со стороны реального соседа: у крайних тайлов
            # внешняя граница остаётся на месте, иначе край полосы выпал бы из покрытия.
            cores.append(
                (
                    x0 + (half if x0 != xs[0] else 0),
                    y0 + (half if y0 != ys[0] else 0),
                    x0 + side - (half if x0 != xs[-1] else 0),
                    y0 + side - (half if y0 != ys[-1] else 0),
                )
            )

    started = time.perf_counter()
    results = predictor(crops, batch_size=batch_size)
    elapsed = time.perf_counter() - started

    polygons = []
    for result, (y0, x0), core in zip(results, [(y, x) for y in ys for x in xs], cores):
        cx1, cy1, cx2, cy2 = core
        for box in result.bboxes:
            if box.confidence is not None and box.confidence < min_conf:
                continue
            poly = np.asarray(box.polygon, dtype=np.float64) + np.array([x0, y0], dtype=np.float64)
            centre = poly.mean(axis=0)
            if cx1 <= centre[0] < cx2 and cy1 <= centre[1] < cy2:
                polygons.append(poly)
    return Detection(polygons=polygons, seconds=elapsed, passes=len(crops))


def save_overlay(path: Path, gray: np.ndarray, detection: Detection, max_side: int = 2200) -> None:
    """Сохраняет уменьшенную копию кадра с обведёнными строками.

    Рамки красятся по высоте строки относительно МЕДИАНЫ этого же кадра: мелкий текст
    зелёный, крупный красный. Так по одному взгляду видно, нашёл ли детектор корпусный
    набор или отработал только по заголовкам.

    Args:
        path: Куда писать JPEG.
        gray: Полутоновый кадр полного разрешения.
        detection: Результат детекции.
        max_side: Длинная сторона наложения в пикселях.
    """
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    canvas = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)

    heights = detection.heights()
    median = float(np.median(heights)) if heights.size else 1.0
    for poly, height in zip(detection.polygons, heights):
        colour = (80, 220, 80) if height <= 1.4 * median else (60, 60, 240)
        cv2.polylines(canvas, [np.round(poly * scale).astype(np.int32)], True, colour, 1, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def tile_counts(detection: Detection, shape: tuple[int, int], n: int = 3) -> np.ndarray:
    """Сколько строк пришлось на каждый тайл сетки n x n (по центру тяжести строки).

    Это и есть решающая цифра при выборе режима детекции. Общее число найденных строк
    обманчиво: режим может набрать их много, но собрать в одной половине кадра, а
    зональная карта строится ровно на противопоставлении частей кадра друг другу. Тайл,
    в который не попало ни одной строки, — это дыра в карте, и чем таких больше, тем
    меньше смысла в «худшем тайле».

    Args:
        detection: Результат детекции.
        shape: Размер кадра (height, width).
        n: Сторона сетки.

    Returns:
        Массив (n, n) с числом строк на тайл.
    """
    counts = np.zeros((n, n), dtype=np.int64)
    height, width = shape
    for poly in detection.polygons:
        cx, cy = poly.mean(axis=0)
        ix = min(n - 1, max(0, int(cx * n / width)))
        iy = min(n - 1, max(0, int(cy * n / height)))
        counts[iy, ix] += 1
    return counts


def summarise(name: str, detection: Detection, shape: tuple[int, int]) -> str:
    """Однострочная сводка по режиму детекции.

    Args:
        name: Имя режима.
        detection: Результат детекции.
        shape: Размер кадра (height, width) — для раскладки по тайлам.

    Returns:
        Строка для печати.
    """
    heights = detection.heights()
    if heights.size == 0:
        return f"    {name:6s}  строк 0  ({detection.seconds:.1f} с, проходов {detection.passes})"
    p10, p50, p90 = np.percentile(heights, [10, 50, 90])
    counts = tile_counts(detection, shape)
    return (
        f"    {name:6s}  строк {heights.size:5d}  "
        f"высота p10/p50/p90 = {p10:5.1f}/{p50:5.1f}/{p90:5.1f} px  "
        f"тайлы 3x3 min/med = {counts.min():4d}/{int(np.median(counts)):4d}  "
        f"({detection.seconds:.1f} с, проходов {detection.passes})"
    )


def main() -> int:
    """Точка входа скрипта.

    Returns:
        Код возврата процесса.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Папка со сканами или отдельные файлы.")
    parser.add_argument("--out", type=Path, default=Path("surya_lines_check"), help="Куда класть наложения.")
    parser.add_argument("--count", type=int, default=5, help="Сколько кадров взять (по порядку имён).")
    parser.add_argument(
        "--mode", choices=("page", "tiles", "both"), default="both", help="Какие режимы детекции сравнивать."
    )
    parser.add_argument(
        "--page-max-side",
        type=int,
        default=0,
        help="В режиме page уменьшить кадр до этой стороны; 0 — подавать как есть.",
    )
    parser.add_argument("--tile-side", type=int, default=DEFAULT_TILE_SIDE, help="Сторона тайла в режиме tiles.")
    parser.add_argument("--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP, help="Перекрытие тайлов.")
    parser.add_argument("--batch-size", type=int, default=0, help="Батч для surya; 0 — её собственный выбор.")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Порог уверенности блока.")
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.inputs:
        files.extend(collect_images(item, recursive=False))
    files = sorted(set(files))[: args.count]
    if not files:
        print("Не нашлось ни одного поддерживаемого файла.", file=sys.stderr)
        return 1

    predictor = load_predictor()
    batch_size = args.batch_size or None
    modes = ("page", "tiles") if args.mode == "both" else (args.mode,)

    for path in files:
        gray = read_gray(path)
        if gray is None:
            print(f"{path.name}: не прочитан", file=sys.stderr)
            continue
        print(f"{path.name}  {gray.shape[1]}x{gray.shape[0]}")
        for mode in modes:
            if mode == "page":
                detection = detect_page(predictor, gray, args.page_max_side or None, args.min_conf)
            else:
                detection = detect_tiles(predictor, gray, args.tile_side, args.tile_overlap, args.min_conf, batch_size)
            print(summarise(mode, detection, gray.shape))
            save_overlay(args.out / f"{path.stem}_{mode}.jpg", gray, detection)

    print(f"\nНаложения: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
