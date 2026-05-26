#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rawpy",
#     "imagededup",
#     "numpy",
#     "Pillow",
#     "opencv-python-headless",
#     "click",
#     "tqdm",
#     "torch",
# ]
# ///
"""Выбирает самый резкий RAF-файл из каждой группы дубликатов."""

import io
import shutil
import tempfile
from enum import Enum
from pathlib import Path

import click
import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps
from tqdm import tqdm


class Method(str, Enum):
    CNN = "cnn"
    LOCAL = "local"


def read_raf_image(path: Path) -> np.ndarray:
    """Читает RAF-файл и возвращает RGB-массив.

    Сначала пробует извлечь встроенный JPEG-превью (быстро),
    при неудаче — обрабатывает RAW через rawpy (медленно).

    Args:
        path: Путь к RAF-файлу для чтения.

    Returns:
        RGB-массив изображения в формате numpy.ndarray с shape (height, width, 3)
        и dtype uint8. Изображение автоматически поворачивается согласно EXIF.

    Raises:
        rawpy.LibRawError: Если файл не является корректным RAW-файлом.
    """
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(bytes(thumb.data)))
                img = ImageOps.exif_transpose(img)
                return np.array(img.convert("RGB"))
        except Exception:
            pass
        return raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)


def compute_sharpness(rgb: np.ndarray) -> float:
    """Вычисляет метрику резкости изображения через дисперсию лапласиана.

    Метод использует центральный кроп размером 2/3 × 2/3 от исходного изображения,
    чтобы исключить влияние краёв. Высокая дисперсия лапласиана указывает на
    большое количество резких границ и деталей.

    Args:
        rgb: RGB-изображение в формате numpy.ndarray с shape (height, width, 3).

    Returns:
        Значение резкости (дисперсия лапласиана). Чем выше значение, тем резче
        изображение. Типичные значения для резких фото: 500-2000+, для размытых: <200.
    """
    h, w = rgb.shape[:2]
    ch, cw = h * 2 // 3, w * 2 // 3
    cy, cx = h // 2, w // 2
    crop = rgb[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_local_features(rgb: np.ndarray, max_width: int = 1024) -> tuple[list, np.ndarray | None]:
    """Вычисляет AKAZE-ключевые точки и дескрипторы центрального кропа.

    Извлекает центральный кроп размером 2/3 × 2/3 от исходного изображения,
    ресайзит его до max_width пикселей по ширине для ускорения детекции,
    затем вычисляет AKAZE-признаки. AKAZE выбран за скорость и устойчивость
    к изменениям освещения.

    Args:
        rgb: RGB-изображение в формате numpy.ndarray с shape (height, width, 3).
        max_width: Максимальная ширина изображения для детекции признаков.
            При 2944→1024 пикселей газетный текст остаётся читаемым,
            а скорость вырастает примерно в 8 раз.

    Returns:
        Кортеж из двух элементов:
            - list: Список ключевых точек cv2.KeyPoint.
            - np.ndarray | None: Массив дескрипторов shape (n_keypoints, 61) dtype uint8,
              или None, если ключевые точки не найдены.
    """
    h, w = rgb.shape[:2]
    ch, cw = h * 2 // 3, w * 2 // 3
    cy, cx = h // 2, w // 2
    crop = rgb[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
    crop_h, crop_w = crop.shape[:2]
    if crop_w > max_width:
        crop = cv2.resize(crop, (max_width, int(crop_h * max_width / crop_w)))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    kp, desc = cv2.AKAZE_create().detectAndCompute(gray, None)
    return kp, desc


def match_local_features(
    kp1: list,
    desc1: np.ndarray | None,
    kp2: list,
    desc2: np.ndarray | None,
    ratio: float = 0.75,
    max_scale: float = 1.3,
    min_match_ratio: float = 0.2,
) -> int:
    """Сопоставляет локальные признаки двух изображений и считает RANSAC-инлайеры.

    Использует Lowe's ratio test для фильтрации совпадений, затем вычисляет
    гомографию через RANSAC. Проверяет масштаб гомографии: если он выходит
    за пределы [1/max_scale, max_scale], возвращает 0 — это признак разного
    кадрирования (например, одна полоса vs. целый разворот, scale ≈ 0.7).
    Для истинных дубликатов scale ≈ 1.0.

    Args:
        kp1: Список ключевых точек первого изображения (cv2.KeyPoint).
        desc1: Дескрипторы первого изображения shape (n1, 61) dtype uint8, или None.
        kp2: Список ключевых точек второго изображения (cv2.KeyPoint).
        desc2: Дескрипторы второго изображения shape (n2, 61) dtype uint8, или None.
        ratio: Порог для Lowe's ratio test. Совпадение принимается, если
            distance(m) < ratio * distance(n), где m — лучшее совпадение,
            n — второе лучшее. Типичное значение: 0.7-0.8.
        max_scale: Максимально допустимое изменение масштаба гомографии.
            Пары с scale вне диапазона [1/max_scale, max_scale] отбрасываются.
        min_match_ratio: Минимальный процент RANSAC-инлайеров от максимума
            количества ключевых точек в паре изображений. Типичное значение: 0.3-0.5.
            Например, при 0.4 и max(len(desc1), len(desc2))=1200 требуется ≥480 инлайеров.

    Returns:
        Количество RANSAC-инлайеров (совпадений, согласующихся с гомографией).
        Возвращает 0, если дескрипторы отсутствуют, совпадений мало (<10),
        гомография не найдена или масштаб выходит за допустимые пределы.
    """
    # Проверка наличия дескрипторов: если дескрипторы отсутствуют или их слишком мало (<2),
    # то сопоставление невозможно (knnMatch требует минимум 2 дескриптора для k=2)
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return 0

    # Lowe's ratio test: сопоставляем дескрипторы через Brute-Force Matcher
    # - cv2.NORM_HAMMING: расстояние Хэмминга для бинарных дескрипторов (AKAZE)
    # - knnMatch(k=2): для каждого дескриптора из desc1 находим 2 ближайших в desc2
    #   * m — лучшее совпадение (минимальное расстояние Хэмминга)
    #   * n — второе лучшее совпадение
    # - if m.distance < ratio * n.distance: принимаем только однозначные совпадения,
    #   где лучшее совпадение значительно лучше второго (по умолчанию ratio=0.75)
    # Это отсекает неоднозначные совпадения (повторяющиеся текстуры, симметричные объекты)
    good = [
        m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2) if m.distance < ratio * n.distance
    ]

    # Если совпадений мало (<10), то RANSAC не сможет надёжно вычислить гомографию
    # (минимум 4 точки для гомографии, но нужен запас для устойчивости)
    # Возвращаем количество совпадений как есть (обычно это будет < min_matches)
    if len(good) < 10:
        return len(good)

    # Извлекаем координаты совпавших ключевых точек из обоих изображений
    # - m.queryIdx: индекс ключевой точки в первом изображении (kp1)
    # - m.trainIdx: индекс ключевой точки во втором изображении (kp2)
    # - .pt: координаты точки (x, y) в пикселях
    # Результат: два массива координат shape (n_good, 2) dtype float32
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # RANSAC-гомография: вычисляем матрицу проективного преобразования 3×3
    # - pts1, pts2: соответствующие точки из двух изображений
    # - cv2.RANSAC: метод устойчивой оценки (отсекает выбросы/аутлайеры)
    #   * Случайно выбирает 4 точки, вычисляет гомографию, проверяет остальные
    #   * Повторяет N раз, выбирает гомографию с максимальным числом инлайеров
    # - 5.0: порог репроекционной ошибки в пикселях (точка — инлайер, если ошибка < 5px)
    # Возвращает:
    # - H: матрица гомографии 3×3 (или None, если не найдена)
    # - mask: бинарный массив shape (n_good, 1), где mask[i]=1 → инлайер, mask[i]=0 → аутлайер
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Если гомография не найдена (слишком мало точек, все аутлайеры, вырожденная конфигурация),
    # то изображения не являются дубликатами
    if H is None:
        return 0

    # Вычисляем масштаб гомографии для проверки кадрирования
    # - H[:2, :2]: левый верхний блок 2×2 (аффинная часть: поворот + масштаб + сдвиг)
    # - det(...): определитель = изменение площади при преобразовании
    # - abs(...): модуль (отрицательный det означает зеркальное отражение)
    # - sqrt(...): переход от площади (2D) к линейному масштабу (1D)
    # Результат: scale ≈ 1.0 для дубликатов, scale ≈ 0.7 для разного кадрирования
    scale = np.sqrt(abs(np.linalg.det(H[:2, :2])))

    # Проверка масштаба: отсекаем пары с сильно отличающимся кадрированием
    # Например, при max_scale=1.15 допускаем scale в диапазоне [1/1.15, 1.15] ≈ [0.87, 1.15]
    # Это предотвращает объединение в группу кадров типа "полоса газеты" vs "целый разворот"
    if not (1 / max_scale <= scale <= max_scale):
        return 0

    # Возвращаем количество RANSAC-инлайеров (правильных совпадений)
    # Это основная метрика качества сопоставления: чем больше инлайеров, тем выше сходство
    # Типичные значения для дубликатов: 500-2000+
    return int(mask.sum())


def find_duplicates_local(
    all_names: list[str],
    all_kps: list[list],
    all_descs: list[np.ndarray | None],
    n_search: int,
    min_match_ratio: float,
    max_scale: float,
) -> dict[str, list[str]]:
    """Находит дубликаты через попарное сравнение AKAZE-дескрипторов.

    Сравнивает каждое изображение только с n_search следующими в отсортированном
    списке (скользящее окно). Это ускоряет поиск с O(n²) до O(n·n_search) и
    отсекает ложные совпадения между удалёнными кадрами.

    Args:
        all_names: Список имён файлов (например, ['IMG_0001.jpg', 'IMG_0002.jpg']).
        all_kps: Список ключевых точек для каждого файла (соответствует all_names).
        all_descs: Список дескрипторов для каждого файла (соответствует all_names).
        n_search: Размер скользящего окна — максимальное расстояние в позициях
            между сравниваемыми кадрами.
        min_match_ratio: Минимальный процент RANSAC-инлайеров от максимума
            количества ключевых точек в паре изображений для признания пары дубликатами.
            Типичное значение: 0.3-0.5.
        max_scale: Максимально допустимое изменение масштаба гомографии
            (передаётся в match_local_features).

    Returns:
        Словарь adjacency list: {имя_файла: [список_имён_дубликатов]}.
        Если у файла нет дубликатов, его значение — пустой список.
    """
    duplicates: dict[str, list[str]] = {name: [] for name in all_names}
    for i in tqdm(range(len(all_names)), desc="Сравниваем признаки"):
        for j in range(i + 1, min(i + n_search + 1, len(all_names))):
            # Вычисляем динамический порог на основе количества ключевых точек в паре
            n_kp_i = len(all_descs[i]) if all_descs[i] is not None else 0
            n_kp_j = len(all_descs[j]) if all_descs[j] is not None else 0
            max_keypoints = max(n_kp_i, n_kp_j)
            min_matches = int(max_keypoints * min_match_ratio)

            # Сопоставляем признаки
            score = match_local_features(
                all_kps[i], all_descs[i], all_kps[j], all_descs[j], max_scale=max_scale, min_match_ratio=min_match_ratio
            )

            # Проверяем, достаточно ли совпадений для признания дубликатами
            if score >= min_matches:
                duplicates[all_names[i]].append(all_names[j])
                duplicates[all_names[j]].append(all_names[i])
    return duplicates


def filter_by_window(duplicates: dict[str, list[str]], all_names: list[str], n_search: int) -> dict[str, list[str]]:
    """Фильтрует дубликаты по позиции в отсортированном списке файлов.

    Оставляет только пары дубликатов, находящиеся в пределах n_search позиций
    друг от друга. Это критично для CNN-метода: разные полосы одного газетного
    номера имеют высокое CNN-сходство (≥0.96), но находятся далеко друг от друга
    в очереди съёмки. Скользящее окно отсекает такие ложные совпадения, сохраняя
    только соседние кадры одной и той же полосы.

    Args:
        duplicates: Словарь adjacency list дубликатов до фильтрации.
            Формат: {имя_файла: [список_имён_дубликатов]}.
        all_names: Отсортированный список всех имён файлов.
        n_search: Максимальное расстояние в позициях между файлами,
            чтобы они считались дубликатами.

    Returns:
        Отфильтрованный словарь adjacency list. Для каждого файла оставлены
        только те соседи, которые находятся в пределах n_search позиций.
    """
    pos = {name: i for i, name in enumerate(all_names)}
    return {
        name: [nb for nb in neighbors if abs(pos[nb] - pos[name]) <= n_search]
        for name, neighbors in duplicates.items()
        if name in pos
    }


def build_groups(all_names: list[str], duplicates: dict[str, list[str]]) -> list[list[str]]:
    """Преобразует adjacency list дубликатов в список связных компонент.

    Использует поиск в ширину (BFS) для обхода графа дубликатов и выделения
    связных компонент. Каждая компонента — это группа взаимосвязанных дубликатов,
    из которой будет выбран один лучший файл.

    Args:
        all_names: Список всех имён файлов.
        duplicates: Словарь adjacency list: {имя_файла: [список_имён_дубликатов]}.

    Returns:
        Список групп, где каждая группа — это список имён файлов-дубликатов.
        Одиночные файлы (без дубликатов) представлены группами из одного элемента.
        Порядок файлов внутри группы не определён.
    """
    visited: set[str] = set()
    groups: list[list[str]] = []

    for node in tqdm(all_names, desc="Строим группы"):
        if node in visited:
            continue
        group: list[str] = []
        queue = [node]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            for neighbor in duplicates.get(cur, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        groups.append(group)

    return groups


@click.command()
@click.argument(
    "input_dir", default="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/неразобранное/2026-05-25 ЭГ 1965 IV-VI"
)
@click.argument("output_dir", default="/mnt/system/raw/1965_4_6_1_out")
@click.option("--n-search", default=5, show_default=True, help="Макс. расстояние в позициях между кадрами одной группы")
@click.option(
    "--method",
    default="local",
    show_default=True,
    type=click.Choice([m.value for m in Method]),
    help="Метод детектирования дубликатов: cnn — MobileNet-эмбеддинги, local — AKAZE локальные признаки",
)
@click.option(
    "--min-similarity",
    default=0.97,
    show_default=True,
    help="[cnn] Мин. косинусное сходство CNN-эмбеддингов для объединения в группу (0..1)",
)
@click.option(
    "--min-match-ratio",
    default=0.2,
    show_default=True,
    help="[local] Мин. процент RANSAC-инлайеров от max(кол-во ключ. точек в паре) для объединения в группу (0..1)",
)
@click.option(
    "--max-scale-change",
    default=1.15,
    show_default=True,
    help="[local] Макс. изменение масштаба гомографии; пары с большим scale отброcываются (полоса vs. разворот)",
)
def main(
    input_dir: str,
    output_dir: str,
    n_search: int,
    method: str,
    min_similarity: float,
    min_match_ratio: float,
    max_scale_change: float,
) -> None:
    """Выбирает самый резкий RAF-файл из каждой группы дубликатов.

    Основная функция скрипта. Сканирует папку с RAF-файлами, находит дубликаты
    одним из двух методов (CNN или локальные признаки), группирует их, выбирает
    самый резкий файл из каждой группы и копирует в выходную папку.

    Args:
        input_dir: Путь к папке с исходными RAF-файлами.
        output_dir: Путь к папке для сохранения лучших файлов.
            Будет создана автоматически, если не существует.
        n_search: Размер скользящего окна — максимальное расстояние в позициях
            между кадрами, которые могут быть признаны дубликатами.
            Типичное значение: 3-10.
        method: Метод детектирования дубликатов:
            - 'cnn': MobileNet-эмбеддинги через imagededup (быстро, требует GPU).
            - 'local': AKAZE локальные признаки (медленнее, но точнее для разного кадрирования).
        min_similarity: [только для CNN] Минимальное косинусное сходство
            CNN-эмбеддингов для объединения в группу. Диапазон: 0.0-1.0.
            Типичное значение: 0.95-0.99.
        min_match_ratio: [только для local] Минимальный процент RANSAC-инлайеров
            от максимума количества ключевых точек в паре изображений.
            Диапазон: 0.0-1.0. Типичное значение: 0.3-0.5.
        max_scale_change: [только для local] Максимально допустимое изменение
            масштаба гомографии. Пары с большим изменением масштаба отбрасываются
            (например, полоса vs. разворот). Типичное значение: 1.1-1.3.

    Returns:
        None. Результаты выводятся в консоль, файлы копируются в output_dir.
    """
    m = Method(method)

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raf_files = sorted(in_path.glob("*.[Rr][Aa][Ff]"))
    if not raf_files:
        click.echo(f"RAF-файлы не найдены в {in_path}")
        return

    click.echo(f"Найдено {len(raf_files)} RAF-файлов в {in_path}, метод: {m.value}")

    all_names = [f.stem + ".jpg" for f in raf_files]
    sharpness: dict[str, float] = {}
    raf_by_jpeg: dict[str, Path] = {}
    all_kps: list[list] = []
    all_descs: list[np.ndarray | None] = []

    if m is Method.CNN:
        # Импорт тяжёлый — здесь, чтобы не грузить torch при выборе local
        from imagededup.methods import CNN

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Шаг 1: извлекаем превью, считаем резкость, сохраняем кроп для CNN
            for f in tqdm(raf_files, desc="Читаем и анализируем"):
                rgb = read_raf_image(f)
                jpeg_name = f.stem + ".jpg"
                sharpness[jpeg_name] = compute_sharpness(rgb)
                raf_by_jpeg[jpeg_name] = f
                img = Image.fromarray(rgb)
                iw, ih = img.size
                cw, ch = iw * 2 // 3, ih * 2 // 3
                img.crop(((iw - cw) // 2, (ih - ch) // 2, (iw + cw) // 2, (ih + ch) // 2)).save(
                    tmp_path / jpeg_name, quality=90
                )
                tqdm.write(f"  {f.name}: sharpness={sharpness[jpeg_name]:.1f}")

            # Шаг 2: CNN-эмбеддинги через MobileNet; imagededup использует CUDA автоматически.
            # Внутренний батч-сайз 32 — для RTX 5060Ti 16 GB MobileNet укладывается в один батч
            # при любом разумном объёме папки; num_enc_workers=0 убирает IPC-оверхед.
            cnn = CNN()
            encodings = cnn.encode_images(image_dir=str(tmp_path), num_enc_workers=0)

            # Шаг 3: поиск дубликатов по косинусному сходству + фильтр по позиции
            duplicates = cnn.find_duplicates(
                encoding_map=encodings, min_similarity_threshold=min_similarity, scores=False
            )

        duplicates = filter_by_window(duplicates, all_names, n_search)

    elif m is Method.LOCAL:
        # Шаг 1: извлекаем превью, считаем резкость и AKAZE-дескрипторы
        for f in tqdm(raf_files, desc="Читаем и анализируем"):
            rgb = read_raf_image(f)
            jpeg_name = f.stem + ".jpg"
            sharpness[jpeg_name] = compute_sharpness(rgb)
            raf_by_jpeg[jpeg_name] = f
            kp, desc = compute_local_features(rgb)
            all_kps.append(kp)
            all_descs.append(desc)
            tqdm.write(f"  {f.name}: sharpness={sharpness[jpeg_name]:.1f}")

        # Шаг 2: попарное сравнение дескрипторов в скользящем окне
        duplicates = find_duplicates_local(all_names, all_kps, all_descs, n_search, min_match_ratio, max_scale_change)

    else:
        raise ValueError(f"Неизвестный метод: {m!r}")

    # Шаг 4: связные компоненты → группы
    groups = build_groups(all_names, duplicates)

    dup_groups = [g for g in groups if len(g) > 1]
    click.echo(f"\nГрупп дубликатов: {len(dup_groups)}, одиночных файлов: {len(groups) - len(dup_groups)}")

    # Шаг 5: копируем лучший файл из каждой группы
    for group in groups:
        best_jpeg = max(group, key=lambda j: sharpness[j])
        src = raf_by_jpeg[best_jpeg]

        if len(group) == 1:
            click.echo(f"  {src.name}: резкость={sharpness[best_jpeg]:.1f}")
        else:
            names = [raf_by_jpeg[j].name for j in group]
            sharp_map = {raf_by_jpeg[j].name: f"{sharpness[j]:.1f}" for j in group}
            click.echo(f"  Дубликаты {names}")
            click.echo(f"    резкость: {sharp_map}")
            click.echo(f"    -> выбран: {src.name}")

        shutil.copy2(src, out_path / src.name)

    click.echo(f"\nСкопировано {len(groups)} файлов в {out_path}")


if __name__ == "__main__":
    main()
