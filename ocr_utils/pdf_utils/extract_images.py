"""Извлечение изображений из PDF без перекодирования.

Этот модуль позволяет извлекать встроенные изображения из PDF-файлов в их исходном формате
(обычно JPEG) без перекодирования, что сохраняет качество и размер файлов.

Примеры использования:

    # Извлечь изображения из одного PDF
    from pathlib import Path
    from ocr_utils.pdf_utils import extract_images_from_pdf
    
    pdf_path = Path("document.pdf")
    output_dir = Path("output/document")
    extract_images_from_pdf(pdf_path, output_dir)

    # Рекурсивно обработать все PDF в директории
    from ocr_utils.pdf_utils import extract_images_recursive
    
    input_dir = Path("/path/to/pdfs")
    output_dir = Path("/path/to/output")
    extract_images_recursive(input_dir, output_dir)

    # Запуск из командной строки
    python -m ocr_utils.pdf_utils.extract_images /path/to/input /path/to/output

Формат выходных файлов:
    Для каждого PDF создаётся отдельная папка с сохранением относительной структуры директорий.
    Например, для input_dir/1936/ПХ-1936-05.pdf изображения сохраняются в
    output_dir/1936/ПХ-1936-05/ПХ-1936-05_0001.jpg, ПХ-1936-05_0002.jpg и т.д.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import fitz
from tqdm import tqdm
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_path: Path, output_dir: Path, pdf_basename: str | None = None) -> int:
    """Извлечь все изображения из PDF-файла без перекодирования.

    Args:
        pdf_path: Путь к исходному PDF-файлу
        output_dir: Директория для сохранения извлечённых изображений
        pdf_basename: Базовое имя для выходных файлов (без расширения). Если None, используется pdf_path.stem

    Returns:
        Количество извлечённых изображений

    Raises:
        FileNotFoundError: Если PDF-файл не существует
        ValueError: Если PDF-файл повреждён или не может быть открыт
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF-файл не найден: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if pdf_basename is None:
        pdf_basename = pdf_path.stem

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Не удалось открыть PDF {pdf_path}: {e}")

    extracted_count = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        images = page.get_images(full=True)

        if not images:
            continue

        # Подсчитываем количество изображений
        all_images = []
        for img_info in images:
            xref = img_info[0]
            try:
                img_dict = doc.extract_image(xref)
                all_images.append(img_dict)
            except Exception as e:
                continue

        # Если несколько изображений (MRC) - рендерим всю страницу
        if len(all_images) > 1:
            try:
                # Находим максимальное разрешение среди изображений
                max_width = max(img["width"] for img in all_images)
                max_height = max(img["height"] for img in all_images)

                # Рендерим страницу с разрешением, соответствующим наибольшему изображению
                # Вычисляем DPI на основе размера страницы и желаемого разрешения
                page_rect = page.rect
                dpi_x = (max_width / page_rect.width) * 72
                dpi_y = (max_height / page_rect.height) * 72
                dpi = int(max(dpi_x, dpi_y))

                # Рендерим страницу
                pix = page.get_pixmap(dpi=dpi)
                img_bytes = pix.tobytes("png")

                output_path = output_dir / f"{pdf_basename}_{page_idx + 1:04d}.png"
                output_path.write_bytes(img_bytes)
                extracted_count += 1

            except Exception as e:
                logger.warning("Не удалось отрендерить страницу %d из %s: %s", page_idx, pdf_path, e)
                continue

        # Если одно изображение - извлекаем его напрямую
        elif len(all_images) == 1:
            try:
                img_dict = all_images[0]
                img_bytes = img_dict["image"]
                img_ext = img_dict["ext"]

                if img_ext in ("jpeg", "jpg"):
                    output_path = output_dir / f"{pdf_basename}_{page_idx + 1:04d}.jpg"
                    output_path.write_bytes(img_bytes)
                elif img_ext == "png":
                    output_path = output_dir / f"{pdf_basename}_{page_idx + 1:04d}.png"
                    output_path.write_bytes(img_bytes)
                elif img_ext in ("jpx", "jp2"):
                    output_path = output_dir / f"{pdf_basename}_{page_idx + 1:04d}.png"
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imwrite(str(output_path), img)
                    else:
                        logger.warning("Не удалось декодировать JPEG 2000 на странице %d из %s", page_idx, pdf_path)
                        continue
                else:
                    output_path = output_dir / f"{pdf_basename}_{page_idx + 1:04d}.{img_ext}"
                    output_path.write_bytes(img_bytes)

                extracted_count += 1

            except Exception as e:
                logger.warning("Не удалось извлечь изображение со страницы %d из %s: %s", page_idx, pdf_path, e)

    doc.close()
    return extracted_count


def extract_images_recursive(input_dir: Path, output_dir: Path, show_progress: bool = True) -> dict[str, int]:
    """Рекурсивно извлечь изображения из всех PDF в директории.

    Для каждого найденного PDF создаётся поддиректория в output_dir с тем же относительным путём,
    что и у исходного PDF. Например, если PDF находится в input_dir/1936/1/3/5/mega.pdf,
    то изображения будут сохранены в output_dir/1936/1/3/5/mega/mega_0001.jpg и т.д.

    Args:
        input_dir: Входная директория для поиска PDF-файлов
        output_dir: Выходная директория для сохранения изображений
        show_progress: Показывать ли прогресс-бар

    Returns:
        Словарь {относительный_путь_к_pdf: количество_извлечённых_изображений}

    Raises:
        FileNotFoundError: Если входная директория не существует
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Путь не является директорией: {input_dir}")

    pdf_files = sorted(input_dir.rglob("*.pdf"))

    if not pdf_files:
        logger.warning("PDF-файлы не найдены в %s", input_dir)
        return {}

    results = {}
    total_pdfs = len(pdf_files)

    if show_progress:
        iterator = tqdm(
            pdf_files,
            desc="Обработка PDF",
            unit="файл",
            total=total_pdfs,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    else:
        iterator = pdf_files

    for pdf_idx, pdf_path in enumerate(iterator, start=1):
        rel_path = pdf_path.relative_to(input_dir)
        pdf_name_without_ext = pdf_path.stem

        parent_rel_path = rel_path.parent
        output_subdir = output_dir / parent_rel_path / pdf_name_without_ext

        try:
            count = extract_images_from_pdf(pdf_path, output_subdir, pdf_basename=pdf_name_without_ext)
            results[str(rel_path)] = count

            if show_progress:
                iterator.set_postfix_str(f"{rel_path.name}: {count} изобр.")
            else:
                logger.info("✓ [%d/%d] %s: извлечено %d изображений", pdf_idx, total_pdfs, rel_path, count)

        except Exception as e:
            logger.error("✗ [%d/%d] Ошибка при обработке %s: %s", pdf_idx, total_pdfs, rel_path, e)
            results[str(rel_path)] = 0

    return results


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--no-progress", is_flag=True, help="Не показывать прогресс-бар")
@click.option("-v", "--verbose", is_flag=True, help="Подробный вывод")
def main(input_dir: Path, output_dir: Path, no_progress: bool, verbose: bool):
    """Извлечение изображений из PDF без перекодирования.

    INPUT_DIR: Входная директория с PDF-файлами

    OUTPUT_DIR: Выходная директория для изображений

    \b
    Примеры:
      uv run python -m ocr_utils.pdf_utils /path/to/pdfs /path/to/output
      uv run python -m ocr_utils.pdf_utils ~/Documents/scans ~/Pictures/extracted --no-progress
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")

    try:
        results = extract_images_recursive(input_dir, output_dir, show_progress=not no_progress)

        total_images = sum(results.values())
        total_pdfs = len(results)
        successful_pdfs = sum(1 for count in results.values() if count > 0)

        print(f"\n{'=' * 60}")
        print(f"Готово!")
        print(f"Обработано PDF: {successful_pdfs}/{total_pdfs}")
        print(f"Извлечено изображений: {total_images}")

    except Exception as e:
        logger.error("Ошибка: %s", e)
        raise click.Abort()


if __name__ == "__main__":
    main()
