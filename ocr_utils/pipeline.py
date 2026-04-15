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
    deskew: bool = True,
    clean: bool = True,
    rotate_pages: bool = True,
) -> Path:
    """Обработать один PDF: разбить развороты на страницы, добавить OCR-слой.

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
        deskew: выравнивать ли страницы перед OCR.
        clean: очищать ли шум перед OCR.
        rotate_pages: автоповорот страниц по ориентации текста.

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
        deskew: выравнивать ли страницы.
        clean: очищать ли шум.
        rotate_pages: автоповорот страниц.

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
        workers = max(1, (cpu_count * 3) // 4)
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
