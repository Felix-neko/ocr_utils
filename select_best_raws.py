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
from pathlib import Path

import click
import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps
from tqdm import tqdm


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

    for node in all_names:
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
    "--min-similarity",
    default=0.97,
    show_default=True,
    help="Мин. косинусное сходство CNN-эмбеддингов для объединения в группу (0..1)",
)
def main(input_dir: str, output_dir: str, n_search: int, min_similarity: float) -> None:
    """Выбирает самый резкий RAF-файл из каждой группы дубликатов.

    INPUT_DIR  — папка с исходными RAF-файлами
    OUTPUT_DIR — папка для лучших файлов (будет создана при необходимости)
    """
    # Импорт тяжёлый — здесь, чтобы click отработал аргументы до загрузки torch
    from imagededup.methods import CNN

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raf_files = sorted(in_path.glob("*.[Rr][Aa][Ff]"))
    if not raf_files:
        click.echo(f"RAF-файлы не найдены в {in_path}")
        return

    click.echo(f"Найдено {len(raf_files)} RAF-файлов в {in_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Шаг 1: извлекаем JPEG-превью, считаем резкость
        sharpness: dict[str, float] = {}
        raf_by_jpeg: dict[str, Path] = {}

        for f in tqdm(raf_files, desc="Читаем и анализируем"):
            rgb = read_raf_image(f)
            s = compute_sharpness(rgb)
            jpeg_name = f.stem + ".jpg"
            sharpness[jpeg_name] = s
            raf_by_jpeg[jpeg_name] = f
            img = Image.fromarray(rgb)
            iw, ih = img.size
            cw, ch = iw * 2 // 3, ih * 2 // 3
            img_crop = img.crop(((iw - cw) // 2, (ih - ch) // 2, (iw + cw) // 2, (ih + ch) // 2))
            img_crop.save(tmp_path / jpeg_name, quality=90)
            tqdm.write(f"  {f.name}: sharpness={s:.1f}")

        # Шаг 2: CNN-эмбеддинги через MobileNet; imagededup использует CUDA автоматически.
        # Внутренний батч-сайз 32 — для RTX 5060Ti 16 GB MobileNet укладывается в один батч
        # при любом разумном объёме папки; num_enc_workers=0 убирает IPC-оверхед.
        cnn = CNN()
        encodings = cnn.encode_images(image_dir=str(tmp_path), num_enc_workers=0)

        # Шаг 3: глобальный поиск дубликатов по косинусному сходству
        duplicates = cnn.find_duplicates(encoding_map=encodings, min_similarity_threshold=min_similarity, scores=False)

    # Шаг 4: отсекаем пары, слишком далеко стоящие в очереди съёмки
    all_names = [f.stem + ".jpg" for f in raf_files]
    duplicates = filter_by_window(duplicates, all_names, n_search)

    # Шаг 5: связные компоненты → группы
    groups = build_groups(all_names, duplicates)

    dup_groups = [g for g in groups if len(g) > 1]
    click.echo(f"\nГрупп дубликатов: {len(dup_groups)}, одиночных файлов: {len(groups) - len(dup_groups)}")

    # Шаг 6: копируем лучший файл из каждой группы
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
