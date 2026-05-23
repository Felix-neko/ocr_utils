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
    """Дисперсия лапласиана центрального кропа 2/3 × 2/3 — мера резкости."""
    h, w = rgb.shape[:2]
    ch, cw = h * 2 // 3, w * 2 // 3
    cy, cx = h // 2, w // 2
    crop = rgb[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_local_features(rgb: np.ndarray, max_width: int = 1024) -> tuple[list, np.ndarray | None]:
    """Вычисляет AKAZE-ключевые точки и дескрипторы центрального кропа 2/3 × 2/3.

    Изображение ресайзится до max_width по ширине перед детекцией —
    при 2944→1024 пикселей газетный текст ещё читаем, скорость вырастает ~8×.
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
) -> int:
    """Считает RANSAC-инлайеры AKAZE-совпадений с проверкой масштаба гомографии.

    Если гомография между парой изображений показывает scale за пределами
    [1/max_scale, max_scale], возвращает 0 — это признак разного кадрирования
    (например, одна полоса vs. целый разворот, scale ≈ 0.7).
    Для истинных дублей scale ≈ 1.0.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return 0
    good = [
        m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2) if m.distance < ratio * n.distance
    ]
    if len(good) < 10:
        return len(good)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        return 0
    scale = np.sqrt(abs(np.linalg.det(H[:2, :2])))
    if not (1 / max_scale <= scale <= max_scale):
        return 0
    return int(mask.sum())


def find_duplicates_local(
    all_names: list[str],
    all_kps: list[list],
    all_descs: list[np.ndarray | None],
    n_search: int,
    min_matches: int,
    max_scale: float,
) -> dict[str, list[str]]:
    """Находит дубликаты через попарное сравнение AKAZE-дескрипторов в скользящем окне."""
    duplicates: dict[str, list[str]] = {name: [] for name in all_names}
    for i in tqdm(range(len(all_names)), desc="Сравниваем признаки"):
        for j in range(i + 1, min(i + n_search + 1, len(all_names))):
            score = match_local_features(all_kps[i], all_descs[i], all_kps[j], all_descs[j], max_scale=max_scale)
            if score >= min_matches:
                duplicates[all_names[i]].append(all_names[j])
                duplicates[all_names[j]].append(all_names[i])
    return duplicates


def filter_by_window(duplicates: dict[str, list[str]], all_names: list[str], n_search: int) -> dict[str, list[str]]:
    """Оставляет только пары дубликатов в пределах n_search позиций в отсортированном списке.

    Разные полосы одного газетного номера имеют высокое CNN-сходство (≥0.96),
    но находятся далеко друг от друга в очереди съёмки. Скользящее окно отсекает
    такие ложные совпадения, сохраняя только соседние кадры одной и той же полосы.
    """
    pos = {name: i for i, name in enumerate(all_names)}
    return {
        name: [nb for nb in neighbors if abs(pos[nb] - pos[name]) <= n_search]
        for name, neighbors in duplicates.items()
        if name in pos
    }


def build_groups(all_names: list[str], duplicates: dict[str, list[str]]) -> list[list[str]]:
    """Преобразует adjacency-list дубликатов в список связных компонент (BFS)."""
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
@click.argument("input_dir", default="/mnt/system/raw/1962")
@click.argument("output_dir", default="/mnt/system/raw/1962_out")
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
    "--min-matches", default=500, show_default=True, help="[local] Мин. число RANSAC-инлайеров для объединения в группу"
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
    min_matches: int,
    max_scale_change: float,
) -> None:  # method приходит как str из click.Choice, конвертируем в enum внутри
    """Выбирает самый резкий RAF-файл из каждой группы дубликатов.

    INPUT_DIR  — папка с исходными RAF-файлами
    OUTPUT_DIR — папка для лучших файлов (будет создана при необходимости)
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
        duplicates = find_duplicates_local(all_names, all_kps, all_descs, n_search, min_matches, max_scale_change)

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
