#!/usr/bin/env python3
"""
Прогон сканов через Surya: детекция текстовых строк и анализ layout.

Рекурсивно обходит директорию с картинками, для каждой картинки строит
две визуализации (оверлеи поверх исходника) и раскладывает их по двум
выходным директориям с сохранением относительных путей.

Модели Surya загружаются один раз на весь прогон.
"""

import time
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# Расширения картинок, которые ищем при обходе (регистр не важен)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# Языки документов. В Surya 0.17 детекция и layout работают без указания языка
# (модели языконезависимые), язык нужен только на этапе распознавания текста.
LANGUAGES = ["ru", "en"]

# Цвета оверлеев для разных типов блоков layout
LAYOUT_COLORS = {
    "Text": "red",
    "Title": "blue",
    "SectionHeader": "blue",
    "Picture": "green",
    "Figure": "green",
    "Table": "orange",
    "Caption": "purple",
    "PageHeader": "brown",
    "PageFooter": "brown",
}


def find_images(in_path: Path) -> List[Path]:
    """Рекурсивно собирает все картинки в директории, отсортированные по пути."""
    images = [p for p in in_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def load_predictors():
    """Загружает предикторы Surya (один раз на весь прогон)."""
    import torch
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings

    print("Загрузка модели детекции Surya...")
    detection_predictor = DetectionPredictor()

    print("Загрузка модели layout Surya...")
    foundation_predictor = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    layout_predictor = LayoutPredictor(foundation_predictor)

    # Гасим внутренние прогресс-бары Surya, чтобы не ломать общий прогресс-бар
    detection_predictor.disable_tqdm = True
    layout_predictor.disable_tqdm = True

    # Явно сообщаем, на чём считаем: молчаливый откат на CPU замедляет прогон в разы
    for name, model in (
        ("детекция", detection_predictor.model),
        ("layout", layout_predictor.foundation_predictor.model),
    ):
        param = next(model.parameters())
        print(f"  {name}: device={param.device}, dtype={param.dtype}")
    if not torch.cuda.is_available():
        print("ВНИМАНИЕ: CUDA недоступна, Surya считает на CPU — это в разы медленнее")

    return detection_predictor, layout_predictor


def _draw_polygons(image: Image.Image, polygons, colors, labels=None) -> Image.Image:
    """Рисует полигоны (и опционально подписи) поверх копии изображения.

    Толщина линий и размер шрифта масштабируются от размера картинки, иначе на
    сканах в 4000+ px оверлеи не видно.
    """
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    scale = max(1, round(max(overlay.size) / 1000))
    line_width = 2 * scale
    font = _get_font(10 * scale)

    for idx, polygon in enumerate(polygons):
        color = colors[idx]
        points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(points, outline=color, width=line_width)

        if labels:
            x = min(p[0] for p in points)
            y = min(p[1] for p in points)
            draw.text((x, max(0, y - 12 * scale)), labels[idx], fill=color, font=font)

    return overlay


def _get_font(size: int) -> ImageFont.ImageFont:
    """Подбирает шрифт нужного размера, с откатом на встроенный в PIL."""
    for font_path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def draw_detection(image: Image.Image, result) -> Image.Image:
    """Рисует найденные текстовые строки поверх копии изображения."""
    polygons = [box.polygon for box in result.bboxes]
    return _draw_polygons(image, polygons, ["red"] * len(polygons))


def draw_layout(image: Image.Image, result) -> Image.Image:
    """Рисует блоки layout поверх копии изображения, подписывая тип блока."""
    polygons = [box.polygon for box in result.bboxes]
    labels = [f"{box.label}-{box.position}" for box in result.bboxes]
    colors = [LAYOUT_COLORS.get(box.label, "red") for box in result.bboxes]
    return _draw_polygons(image, polygons, colors, labels=labels)


def save_overlay(overlay: Image.Image, in_path: Path, image_path: Path, out_root: Path) -> None:
    """Сохраняет визуализацию в out_root, повторяя относительный путь исходника.

    Визуализации кладём в JPEG: это на порядок быстрее и компактнее PNG, а для
    просмотра оверлеев потерь качества достаточно.
    """
    rel_path = image_path.relative_to(in_path)
    out_file = out_root / rel_path.with_suffix(".jpg")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_file, quality=92, subsampling=0)


def process_directory(in_path: Path, out_detection_path: Path, out_layout_path: Path) -> None:
    """Прогоняет все картинки из in_path через детекцию и layout Surya."""
    images = find_images(in_path)
    if not images:
        print(f"В {in_path} не найдено ни одной картинки")
        return

    print(f"Найдено картинок: {len(images)}")

    load_start = time.time()
    detection_predictor, layout_predictor = load_predictors()
    print(f"Модели загружены за {time.time() - load_start:.1f} с")

    total_detection = 0.0
    total_layout = 0.0
    run_start = time.time()

    progress = tqdm(images, desc="Surya", unit="img")
    for image_path in progress:
        try:
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
        except Exception as exc:  # битые файлы просто пропускаем
            progress.write(f"Не удалось открыть {image_path}: {exc}")
            continue

        t0 = time.time()
        detection_result = detection_predictor([image])[0]
        t1 = time.time()
        layout_result = layout_predictor([image])[0]
        t2 = time.time()

        total_detection += t1 - t0
        total_layout += t2 - t1

        save_overlay(draw_detection(image, detection_result), in_path, image_path, out_detection_path)
        save_overlay(draw_layout(image, layout_result), in_path, image_path, out_layout_path)

        progress.set_postfix(det=f"{t1 - t0:.1f}s", lay=f"{t2 - t1:.1f}s")

    elapsed = time.time() - run_start
    print(f"\nОбработано картинок: {len(images)} за {elapsed:.1f} с")
    print(f"  детекция: {total_detection:.1f} с (в среднем {total_detection / len(images):.2f} с/картинка)")
    print(f"  layout:   {total_layout:.1f} с (в среднем {total_layout / len(images):.2f} с/картинка)")


if __name__ == "__main__":

    in_path = "/mnt/system/raw/ve_80s/in/1989/06 проверить зональный пересвет"
    out_detection_path = "/mnt/system/raw/ve_80s/test_896/surya_detection"
    out_layout_path = "/mnt/system/raw/ve_80s/test_896/surya_layout"

    process_directory(Path(in_path), Path(out_detection_path), Path(out_layout_path))
