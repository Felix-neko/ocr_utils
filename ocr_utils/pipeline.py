"""Основной пайплайн: обработка одного PDF и рекурсивная обработка директории."""

from __future__ import annotations

import logging
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from ocr_utils.config import OCR_LANGUAGE, OCR_OVERSAMPLE_DPI
from ocr_utils.ocr import run_ocr
from ocr_utils.splitting import split_pdf_pages

logger = logging.getLogger(__name__)


def process_single_pdf(
    src_pdf: Path | str,
    dst_pdf: Path | str,
    pages: Sequence[int] | slice | None = None,
    tmp_dir: Path | str | None = None,
    language: str = OCR_LANGUAGE,
    oversample_dpi: int = OCR_OVERSAMPLE_DPI,
    deskew: bool = False,
    clean: bool = True,
    rotate_pages: bool = True,
    intermediate_pdf_path: Path | str | None = None,
) -> Path:
    """Обработать один PDF: разбить развороты на страницы, добавить OCR-слой.

    Алгоритм:
    1. Разбить развороты на отдельные страницы → промежуточный PDF #1 (исходные изображения).
    2. OCR с deskew/clean/rotate для качественного распознавания → промежуточный PDF #2.
    3. Перенос текстового слоя из PDF #2 на изображения из PDF #1 → финальный PDF.

    Аргументы:
        src_pdf: путь к исходному PDF.
        dst_pdf: путь к выходному PDF.
        pages: какие страницы обрабатывать.
            None → все страницы.
            list[int] → список 0-based индексов страниц.
            slice → Python-слайс страниц.
        tmp_dir: директория для временных файлов.
            None → создаётся временная директория, которая удаляется после работы.
        language: язык(и) Tesseract для OCR.
        oversample_dpi: DPI для оверсемплинга перед OCR.
        deskew: выравнивать ли страницы при OCR (только для промежуточного PDF #2).
        clean: очищать ли шум при OCR (только для промежуточного PDF #2).
        rotate_pages: автоповорот страниц при OCR (только для промежуточного PDF #2).
        intermediate_pdf_path: путь для сохранения промежуточного PDF #2 (после OCR с обработкой).
            None → промежуточный PDF не сохраняется.

    Возвращает:
        Path к выходному PDF.
    """
    src_pdf = Path(src_pdf)
    dst_pdf = Path(dst_pdf)

    if not src_pdf.exists():
        raise FileNotFoundError(f"Исходный PDF не найден: {src_pdf}")

    dst_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Если tmp_dir не задан — создаём временную директорию, которая будет удалена после работы
    managed_tmp = tmp_dir is None
    if managed_tmp:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="ocr_utils_")
        tmp_path = Path(tmp_dir_obj.name)
    else:
        tmp_path = Path(tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_dir_obj = None

    try:
        logger.info("Обработка: %s → %s", src_pdf, dst_pdf)

        # Шаг 1: разбить развороты на отдельные страницы → промежуточный PDF #1
        split_pdf = split_pdf_pages(src_pdf, tmp_path, pages=pages)
        logger.info("Шаг 1 (разбивка) завершён: %s", split_pdf)

        # Шаг 2: OCR через ocrmypdf → финальный PDF
        run_ocr(
            input_pdf=split_pdf,
            output_pdf=dst_pdf,
            language=language,
            oversample_dpi=oversample_dpi,
            deskew=deskew,
            clean=clean,
            rotate_pages=rotate_pages,
            intermediate_pdf_path=intermediate_pdf_path,
        )
        logger.info("Шаг 2 (OCR) завершён: %s", dst_pdf)

        return dst_pdf
    finally:
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


def _process_one(args: tuple) -> tuple[str, str | None]:
    """Обёртка для process_single_pdf, пригодная для ProcessPoolExecutor.

    Возвращает (относительный путь, None) при успехе или (относительный путь, текст ошибки) при ошибке.
    """
    src_pdf_str, dst_pdf_str, language, oversample_dpi, deskew, clean, rotate_pages = args
    src_pdf = Path(src_pdf_str)
    dst_pdf = Path(dst_pdf_str)
    rel = src_pdf.name
    try:
        process_single_pdf(
            src_pdf=src_pdf,
            dst_pdf=dst_pdf,
            language=language,
            oversample_dpi=oversample_dpi,
            deskew=deskew,
            clean=clean,
            rotate_pages=rotate_pages,
        )
        return (rel, None)
    except Exception as e:
        logger.error("Ошибка при обработке %s: %s", rel, e)
        return (rel, str(e))


def process_directory(
    src_dir: Path | str,
    dst_dir: Path | str,
    workers: int | None = None,
    language: str = OCR_LANGUAGE,
    oversample_dpi: int = OCR_OVERSAMPLE_DPI,
    deskew: bool = True,
    clean: bool = True,
    rotate_pages: bool = True,
) -> dict[str, str | None]:
    """Рекурсивно обработать все PDF в директории.

    Аргументы:
        src_dir: путь к исходной директории.
        dst_dir: путь к выходной директории (будет создана, если не существует).
        workers: количество параллельных процессов.
            None → 3/4 от доступных ядер, но не менее 1.
        language: язык(и) Tesseract для OCR.
        oversample_dpi: DPI для оверсемплинга перед OCR.
        deskew: выравнивать ли страницы при OCR.
        clean: очищать ли шум при OCR.
        rotate_pages: автоповорот страниц при OCR.

    Возвращает:
        dict: {относительный_путь: None (успех) или строка ошибки}.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.is_dir():
        raise NotADirectoryError(f"Исходная директория не найдена: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Собираем все PDF-файлы рекурсивно
    pdf_files = sorted(src_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("PDF-файлы не найдены в %s", src_dir)
        return {}

    logger.info("Найдено %d PDF-файлов в %s", len(pdf_files), src_dir)

    # Определяем количество воркеров
    if workers is None:
        cpu_count = multiprocessing.cpu_count()
        workers = max(1, int(cpu_count * 3 / 4))
    workers = max(1, workers)
    logger.info("Используем %d воркеров", workers)

    # Формируем задачи: (src, dst, параметры)
    tasks = []
    for pdf_path in pdf_files:
        rel_path = pdf_path.relative_to(src_dir)
        out_path = dst_dir / rel_path
        tasks.append((str(pdf_path), str(out_path), language, oversample_dpi, deskew, clean, rotate_pages))

    results: dict[str, str | None] = {}

    if workers == 1:
        # Без пула, чтобы ошибки были нагляднее
        for task in tasks:
            rel, err = _process_one(task)
            results[rel] = err
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one, task): task for task in tasks}
            for future in as_completed(futures):
                rel, err = future.result()
                results[rel] = err

    ok = sum(1 for v in results.values() if v is None)
    fail = sum(1 for v in results.values() if v is not None)
    logger.info("Готово: %d успешно, %d с ошибками (из %d)", ok, fail, len(results))
    return results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    src = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989)/1939/Плановое хозяйство 1-1939.pdf")
    dst = Path("/tmp/test_ocr_result6.pdf")
    intermediate = Path("/tmp/test_ocr_intermediate5.pdf")

    print(f"Обработка: {src}")
    print(f"Результат: {dst}")
    print(f"Промежуточный PDF (после OCR с обработкой): {intermediate}")
    print("Обрабатываем первые 5 страниц с oversample_dpi=600...")

    result = process_single_pdf(
        src_pdf=src,
        dst_pdf=dst,
        pages=slice(0, 5),
        oversample_dpi=600,
        intermediate_pdf_path=intermediate,
        deskew=False,
    )

    print(f"\nГотово! Результат сохранён в: {result}")
    print("Проверяем количество страниц...")

    import fitz

    doc = fitz.open(str(result))
    print(f"Количество страниц в результате: {len(doc)}")

    for i in range(min(len(doc), 10)):
        text = doc[i].get_text().strip()
        has_text = len(text) > 50
        print(f"  Страница {i}: текст {'найден' if has_text else 'НЕ найден'} ({len(text)} символов)")

    doc.close()

    print(f"\nСравнение OCR-слоёв на последней странице:")
    print("=" * 60)

    old_pdf = Path("/tmp/test_ocr_result.pdf")
    new_pdf = Path("/tmp/test_ocr_result6.pdf")

    if old_pdf.exists():
        old_doc = fitz.open(str(old_pdf))
        old_last_page = len(old_doc) - 1
        old_text = old_doc[old_last_page].get_text().strip()
        old_doc.close()
        print(f"\nСтарый PDF ({old_pdf}):")
        print(f"  Последняя страница ({old_last_page}): {len(old_text)} символов")
        print(f"  Первые 200 символов: {old_text[:200]!r}")
    else:
        print(f"\nСтарый PDF не найден: {old_pdf}")

    new_doc = fitz.open(str(new_pdf))
    new_last_page = len(new_doc) - 1
    new_text = new_doc[new_last_page].get_text().strip()
    new_doc.close()
    print(f"\nНовый PDF ({new_pdf}):")
    print(f"  Последняя страница ({new_last_page}): {len(new_text)} символов")
    print(f"  Первые 200 символов: {new_text[:200]!r}")
