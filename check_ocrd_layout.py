#!/usr/bin/env python3
"""
Тестирование ocrd_anybaseocr и ocrd_segment для определения границ полезной области на страницах.
Обрабатывает все TIF-файлы из входной директории, запускает цепочку OCR-D процессоров
(anybaseocr-crop → segment-repair) и сохраняет изображения с нарисованными границами.

Примечание: ocrd-anybaseocr-block-segmentation требует скачивания весов модели
(mask_rcnn_block_0099.h5), которые не зарегистрированы в реестре ресурсов OCR-D 1.8.2.
Поэтому пайплайн использует crop (находит PrintSpace/Border) + repair (валидация).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import lxml.etree as ET
from PIL import Image, ImageDraw


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

COLOR_MAP = {
    "TextRegion": "blue",
    "ImageRegion": "purple",
    "TableRegion": "orange",
    "GraphicRegion": "green",
    "SeparatorRegion": "red",
    "MathsRegion": "cyan",
    "ChartRegion": "magenta",
    "MapRegion": "yellow",
    "AdvertRegion": "pink",
    "ChemRegion": "lime",
    "MusicRegion": "teal",
    "NoiseRegion": "gray",
    "UnknownRegion": "white",
}


def create_workspace(workspace_dir: Path, image_path: Path) -> None:
    """Создаёт OCR-D workspace и добавляет изображение с привязкой к странице."""
    img_dest = workspace_dir / "OCR-D-IMG" / image_path.name
    img_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, img_dest)

    subprocess.run(
        ["ocrd", "workspace", "-d", str(workspace_dir), "init"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ocrd", "workspace", "-d", str(workspace_dir), "add",
            "-G", "OCR-D-IMG",
            "-i", "FILE_0001",
            "-g", "PAGE_0001",
            "-m", "image/tiff",
            str(img_dest),
        ],
        check=True,
        capture_output=True,
    )


def run_processor(args: list[str], workspace_dir: Path, step_name: str) -> bool:
    """Запускает OCR-D процессор, возвращает True при успехе."""
    result = subprocess.run(args + ["-w", str(workspace_dir)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ {step_name} завершился с ошибкой (код {result.returncode}):")
        stderr_lines = [l for l in result.stderr.splitlines() if not any(
            skip in l for skip in ["pkg_resources", "tensorflow", "DeprecationWarning",
                                   "non-resource", "I0000", "To enable", "WARNING: All"]
        )]
        for line in stderr_lines[-5:]:
            print(f"    {line}")
        return False
    return True


def find_page_xml(workspace_dir: Path, file_grp: str) -> Path | None:
    """Ищет первый PAGE XML файл в директории файловой группы."""
    grp_dir = workspace_dir / file_grp
    if grp_dir.exists():
        xml_files = sorted(grp_dir.glob("*.xml"))
        if xml_files:
            return xml_files[0]
    return None


def parse_page_content(page_xml_path: Path) -> tuple[tuple | None, list[dict]]:
    """
    Парсит PAGE XML и возвращает (border_bbox, список регионов).
    border_bbox — область из элемента Border (PrintSpace от crop).
    regions — список TextRegion/ImageRegion/... с типом и bbox.
    """
    tree = ET.parse(str(page_xml_path))
    root = tree.getroot()
    ns = {"p": PAGE_NS}

    border_bbox = None
    border_el = root.find(".//p:Border/p:Coords", ns)
    if border_el is not None:
        pts = _parse_points(border_el.get("points", ""))
        if pts:
            border_bbox = _bbox_from_points(pts)

    regions = []
    for rtype in COLOR_MAP:
        for region in root.findall(f".//p:{rtype}", ns):
            coords_el = region.find("p:Coords", ns)
            if coords_el is None:
                continue
            pts = _parse_points(coords_el.get("points", ""))
            if pts:
                regions.append({"type": rtype, "bbox": _bbox_from_points(pts)})

    return border_bbox, regions


def _parse_points(points_str: str) -> list[tuple[int, int]]:
    pts = []
    for pt in points_str.split():
        parts = pt.split(",")
        if len(parts) == 2:
            pts.append((int(float(parts[0])), int(float(parts[1]))))
    return pts


def _bbox_from_points(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def find_useful_area_bbox(border_bbox: tuple | None, regions: list[dict]) -> tuple | None:
    """Возвращает общий bbox: из регионов если они есть, иначе из Border."""
    if regions:
        x1 = min(r["bbox"][0] for r in regions)
        y1 = min(r["bbox"][1] for r in regions)
        x2 = max(r["bbox"][2] for r in regions)
        y2 = max(r["bbox"][3] for r in regions)
        return (x1, y1, x2, y2)
    return border_bbox


def process_images(input_dir: Path, output_dir: Path):
    """
    Обрабатывает все TIF-файлы из input_dir через цепочку OCR-D процессоров
    и сохраняет результаты с нарисованными bounding boxes в output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(input_dir.glob("*.tif"))
    if not tif_files:
        print(f"Не найдено TIF-файлов в {input_dir}")
        return

    print(f"Найдено {len(tif_files)} TIF-файлов")

    # block-segmentation требует Mask R-CNN модели; проверяем один раз перед началом
    block_seg_available: bool | None = None

    for idx, tif_path in enumerate(tif_files, 1):
        print(f"\n[{idx}/{len(tif_files)}] Обработка: {tif_path.name}")

        img = Image.open(tif_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        print(f"  Размер изображения: {img.size}")

        with tempfile.TemporaryDirectory(prefix="ocrd_") as tmpdir:
            workspace_dir = Path(tmpdir) / "ws"
            workspace_dir.mkdir()

            print(f"  Создание OCR-D workspace...")
            try:
                create_workspace(workspace_dir, tif_path)
            except subprocess.CalledProcessError as e:
                print(f"  ⚠ Ошибка создания workspace: {e}")
                continue

            # crop находит область печати (Border/PrintSpace) традиционными методами
            print(f"  Запуск ocrd-anybaseocr-crop...")
            crop_ok = run_processor(
                ["ocrd-anybaseocr-crop", "-I", "OCR-D-IMG", "-O", "OCR-D-CROP"],
                workspace_dir,
                "ocrd-anybaseocr-crop",
            )

            seg_ok = False
            if crop_ok:
                if block_seg_available is None:
                    # пробуем один раз; если модели нет — пропускаем все дальнейшие файлы
                    print(f"  Запуск ocrd-anybaseocr-block-segmentation...")
                    seg_ok = run_processor(
                        ["ocrd-anybaseocr-block-segmentation", "-I", "OCR-D-CROP", "-O", "OCR-D-SEG"],
                        workspace_dir,
                        "ocrd-anybaseocr-block-segmentation",
                    )
                    block_seg_available = seg_ok
                    if not block_seg_available:
                        print(f"  (block-segmentation будет пропущен для остальных файлов)")
                elif block_seg_available:
                    print(f"  Запуск ocrd-anybaseocr-block-segmentation...")
                    seg_ok = run_processor(
                        ["ocrd-anybaseocr-block-segmentation", "-I", "OCR-D-CROP", "-O", "OCR-D-SEG"],
                        workspace_dir,
                        "ocrd-anybaseocr-block-segmentation",
                    )

            # repair валидирует и чистит PAGE XML на лучшем доступном выводе
            best_input = "OCR-D-SEG" if seg_ok else ("OCR-D-CROP" if crop_ok else "OCR-D-IMG")
            print(f"  Запуск ocrd-segment-repair (вход: {best_input})...")
            repair_ok = run_processor(
                ["ocrd-segment-repair", "-I", best_input, "-O", "OCR-D-REPAIRED"],
                workspace_dir,
                "ocrd-segment-repair",
            )

            final_grp = "OCR-D-REPAIRED" if repair_ok else best_input
            page_xml = find_page_xml(workspace_dir, final_grp)

            if page_xml is None:
                print(f"  ⚠ PAGE XML не найден в группе {final_grp}")
                output_path = output_dir / tif_path.name
                img.save(output_path, compression="tiff_deflate")
                print(f"  ✓ Сохранено без разметки: {output_path}")
                continue

            border_bbox, regions = parse_page_content(page_xml)

            if border_bbox:
                bx1, by1, bx2, by2 = border_bbox
                print(f"  PrintSpace (Border): ({bx1}, {by1}) → ({bx2}, {by2})")

            if regions:
                region_types: dict[str, int] = {}
                for r in regions:
                    region_types[r["type"]] = region_types.get(r["type"], 0) + 1
                print(f"  Регионы: {region_types}")
            else:
                print(f"  Регионов не найдено (только PrintSpace от crop)")

            bbox = find_useful_area_bbox(border_bbox, regions)
            draw = ImageDraw.Draw(img)

            if bbox:
                x1, y1, x2, y2 = bbox
                print(f"  Границы полезной области: ({x1}, {y1}) → ({x2}, {y2})")
                print(f"  Размер области: {x2 - x1} x {y2 - y1}")
                draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

                for region in regions:
                    rx1, ry1, rx2, ry2 = region["bbox"]
                    color = COLOR_MAP.get(region["type"], "blue")
                    draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=2)

                if border_bbox and not regions:
                    bx1, by1, bx2, by2 = border_bbox
                    draw.rectangle([bx1, by1, bx2, by2], outline="orange", width=3)
            else:
                print(f"  ⚠ Полезная область не найдена")

        output_path = output_dir / tif_path.name
        img.save(output_path, compression="tiff_deflate")
        print(f"  ✓ Сохранено: {output_path}")


if __name__ == "__main__":
    input_directory = Path("/mnt/dump3/DOWN/1975-12/out")
    output_directory = Path("/mnt/dump3/DOWN/1975-12/out_ocrd")

    print("=" * 80)
    print("Тестирование ocrd_anybaseocr + ocrd_segment для определения границ полезной области")
    print("=" * 80)
    print(f"Входная директория: {input_directory}")
    print(f"Выходная директория: {output_directory}")
    print()

    process_images(input_directory, output_directory)

    print("\n" + "=" * 80)
    print("✓ Обработка завершена")
    print("=" * 80)
