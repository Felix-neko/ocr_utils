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

    # С полями в 64 px, залитыми осветлённым на 10 тонов цветом бумаги, и тегом 300 DPI
    python -m ocr_utils.pdf_utils /path/to/input /path/to/output --padding 64 --brighten 10 --dpi 300

Поля:
    Опция --padding добавляет вокруг страницы поля заданной ширины, залитые оценённым
    цветом бумаги этой самой страницы (см. ocr_utils.pdf_utils.padding). Для JPEG поля
    добавляются на уровне DCT-блоков, без перекодирования; ширина при этом округляется
    вверх до кратной размеру MCU (8 или 16 px в зависимости от субдискретизации).

Разрешение:
    Опция --dpi проставляет выходным файлам заданное разрешение (по умолчанию выключена
    — разрешение остаётся таким, каким было). Для JPEG это правка сегмента JFIF APP0
    прямо в байтах, без перекодирования. Это только тег: количество пикселей от него
    не меняется, картинка не масштабируется.

Битые выпуски:
    Если у PDF потерялась хоть одна страница — не открылась, не содержит картинок,
    картинка не извлеклась, — выгрузка этого выпуска выбрасывается целиком, а не
    остаётся половиной. Половинный выпуск ничем не отличается на вид от целого, и
    обнаружить подмену потом уже нечем. Отключается ключом --keep-broken.

Параллелизм:
    PDF обрабатываются пулом процессов (--jobs, по умолчанию по числу ядер, но не больше
    DEFAULT_MAX_JOBS): работа CPU-интенсивная, а файлы друг от друга не зависят.

Формат выходных файлов:
    Для каждого PDF создаётся отдельная папка с сохранением относительной структуры директорий.
    Например, для input_dir/1936/ПХ-1936-05.pdf изображения сохраняются в
    output_dir/1936/ПХ-1936-05/ПХ-1936-05_0001.jpg, ПХ-1936-05_0002.jpg и т.д.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import click
import fitz
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

from ocr_utils.pdf_utils.padding import (
    JPEG_DCT_BLOCK_SIZE,
    PAPER_ANALYSIS_MAX_SIDE,
    JPEG_MAX_MCU_SIZE,
    TARGET_DPI,
    LosslessPaddingError,
    brighten_color,
    estimate_paper_color,
    pad_image_array,
    pad_jpeg_lossless,
    set_jpeg_dpi,
)

logger = logging.getLogger(__name__)

#: Больше воркеров, чем физических ядер, смысла не имеет: задача упирается в счёт,
#: а исходники лежат на медленном диске.
DEFAULT_MAX_JOBS = 16


class BrokenPdfError(RuntimeError):
    """В PDF потеряны страницы, и его выгрузка выброшена целиком.

    Attributes:
        pdf_path: Путь к исходному PDF
        page_count: Сколько в нём всего страниц
        problems: Список (номер страницы, что с ней не так)
    """

    def __init__(self, pdf_path: Path, page_count: int, problems: list[tuple[int, str]]) -> None:
        self.pdf_path = pdf_path
        self.page_count = page_count
        self.problems = problems
        super().__init__(f"{pdf_path}: {describe_problems(problems, page_count)}")


def describe_problems(problems: list[tuple[int, str]], page_count: int) -> str:
    """Коротко описать, что не так с выпуском.

    Args:
        problems: Список (номер страницы, причина)
        page_count: Всего страниц в PDF

    Returns:
        Строка вида «27 страниц из 116: нет картинок (стр. 90-116)»
    """
    if not problems:
        return "без проблем"

    reasons: dict[str, list[int]] = {}
    for page, reason in problems:
        reasons.setdefault(reason, []).append(page)

    parts = []
    for reason, pages in reasons.items():
        if len(pages) > 2 and pages == list(range(pages[0], pages[-1] + 1)):
            where = f"стр. {pages[0]}-{pages[-1]}"
        elif len(pages) > 4:
            where = f"стр. {', '.join(str(p) for p in pages[:4])} и ещё {len(pages) - 4}"
        else:
            where = f"стр. {', '.join(str(p) for p in pages)}"
        parts.append(f"{reason} ({where})")
    return f"{len(problems)} страниц из {page_count}: " + "; ".join(parts)


def _discard_output(output_dir: Path, pdf_basename: str) -> None:
    """Убрать выгрузку выпуска, чтобы от него не осталось половины.

    Удаляются файлы, названные по схеме этого экспорта — включая оставшиеся от прежних
    прогонов, — после чего папка сносится, если опустела. Посторонние файлы не трогаем:
    output_dir приходит снаружи, и снести её целиком было бы слишком лихо.

    Args:
        output_dir: Папка выгрузки выпуска
        pdf_basename: Базовое имя, с которого начинаются имена файлов
    """
    if not output_dir.exists():
        return

    for path in output_dir.glob(f"{pdf_basename}_[0-9]*.*"):
        if path.is_file():
            path.unlink(missing_ok=True)

    leftovers = sorted(p.name for p in output_dir.iterdir())
    if leftovers:
        logger.warning("В %s остались посторонние файлы, папка не удалена: %s", output_dir, ", ".join(leftovers[:5]))
        return
    output_dir.rmdir()


def extract_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    pdf_basename: str | None = None,
    padding: int | None = None,
    brighten: int | None = None,
    dpi: int | None = None,
    discard_on_error: bool = True,
) -> int:
    """Извлечь все изображения из PDF-файла без перекодирования.

    Args:
        pdf_path: Путь к исходному PDF-файлу
        output_dir: Директория для сохранения извлечённых изображений
        pdf_basename: Базовое имя для выходных файлов (без расширения). Если None, используется pdf_path.stem
        padding: Ширина добавляемых полей в пикселях. None или 0 — без полей
        brighten: На сколько тонов из 256 осветлить цвет заливки полей. None — не осветлять
        dpi: Разрешение, которое проставить выходным файлам. None — оставить как есть.
            Это только тег: количество пикселей от него не меняется
        discard_on_error: Если хоть одна страница потерялась — выбросить выгрузку выпуска
            целиком и поднять BrokenPdfError. Иначе просто пожаловаться в лог и оставить,
            что вышло

    Returns:
        Количество извлечённых изображений

    Raises:
        FileNotFoundError: Если PDF-файл не существует
        ValueError: Если PDF-файл повреждён или не может быть открыт
        BrokenPdfError: Если страницы потерялись, а discard_on_error включён
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

    if getattr(doc, "is_repaired", False):
        # Само по себе не приговор, но у битых выпусков пака это первый признак.
        logger.info("PDF %s пришлось чинить при открытии", pdf_path)

    extracted_count = 0
    problems: list[tuple[int, str]] = []

    for page_idx in range(len(doc)):
        page_no = page_idx + 1
        try:
            page = doc[page_idx]
            images = page.get_images(full=True)
        except Exception as e:
            problems.append((page_no, "страница не открылась"))
            logger.warning("Страница %d из %s не открылась: %s", page_no, pdf_path, e)
            continue

        if not images:
            # Каждая страница пака — это скан; пустая страница значит потерянный кусок PDF.
            problems.append((page_no, "нет картинок"))
            continue

        # Подсчитываем количество изображений
        all_images = []
        for img_info in images:
            xref = img_info[0]
            try:
                img_dict = doc.extract_image(xref)
                all_images.append(img_dict)
            except Exception as e:
                logger.warning("Не удалось прочитать картинку %d на стр. %d из %s: %s", xref, page_no, pdf_path, e)

        if not all_images:
            problems.append((page_no, "картинки не читаются"))
            continue

        # Если несколько изображений (MRC) - рендерим всю страницу
        if len(all_images) > 1:
            try:
                # Находим максимальное разрешение среди изображений
                max_width = max(img["width"] for img in all_images)
                max_height = max(img["height"] for img in all_images)

                # Рендерим страницу ровно в размер наибольшего из вложенных изображений:
                # масштаб задаём матрицей по каждой оси отдельно. Через параметр dpi
                # получалось округление вниз до целого DPI — и страница выходила на
                # несколько пикселей меньше оригинала.
                page_rect = page.rect
                zoom_x = max_width / page_rect.width
                zoom_y = max_height / page_rect.height
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y))

                if pix.width < max_width or pix.height < max_height:
                    logger.warning(
                        "Страница %d из %s отрендерена в %dx%d вместо %dx%d",
                        page_no,
                        pdf_path,
                        pix.width,
                        pix.height,
                        max_width,
                        max_height,
                    )

                image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 1:
                    image = image[:, :, 0]
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)

                output_path = output_dir / f"{pdf_basename}_{page_no:04d}.png"
                _write_raster(image, output_path, padding, brighten, dpi)
                extracted_count += 1

            except Exception as e:
                problems.append((page_no, "страница не отрендерилась"))
                logger.warning("Не удалось отрендерить страницу %d из %s: %s", page_no, pdf_path, e)
                continue

        # Если одно изображение - извлекаем его напрямую
        else:
            try:
                img_dict = all_images[0]
                img_bytes = img_dict["image"]
                img_ext = img_dict["ext"]

                if img_ext in ("jpeg", "jpg"):
                    output_path = output_dir / f"{pdf_basename}_{page_no:04d}.jpg"
                    output_path.write_bytes(
                        _pad_jpeg(img_bytes, padding, brighten, dpi, f"{pdf_path}, страница {page_no}")
                    )
                elif img_ext == "png":
                    output_path = output_dir / f"{pdf_basename}_{page_no:04d}.png"
                    if padding:
                        image = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        if image is None:
                            raise ValueError("PNG не декодируется")
                        _write_raster(image, output_path, padding, brighten, dpi)
                    else:
                        output_path.write_bytes(_png_with_dpi(img_bytes, dpi))
                elif img_ext in ("jpx", "jp2"):
                    output_path = output_dir / f"{pdf_basename}_{page_no:04d}.png"
                    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("JPEG 2000 не декодируется")
                    _write_raster(img, output_path, padding, brighten, dpi)
                else:
                    output_path = output_dir / f"{pdf_basename}_{page_no:04d}.{img_ext}"
                    if padding:
                        logger.warning(
                            "Формат %s на странице %d из %s не поддерживает добавление полей — сохраняем как есть",
                            img_ext,
                            page_no,
                            pdf_path,
                        )
                    output_path.write_bytes(img_bytes)

                extracted_count += 1

            except Exception as e:
                problems.append((page_no, "картинка не извлеклась"))
                logger.warning("Не удалось извлечь изображение со страницы %d из %s: %s", page_no, pdf_path, e)

    page_count = len(doc)
    doc.close()

    if problems:
        if discard_on_error:
            _discard_output(output_dir, pdf_basename)
            raise BrokenPdfError(pdf_path, page_count, problems)
        logger.warning("Выпуск %s выгружен не полностью — %s", pdf_path, describe_problems(problems, page_count))

    return extracted_count


def _fill_color(image_bgr: np.ndarray, brighten: int | None) -> tuple[int, int, int]:
    """Подобрать цвет заливки полей для картинки.

    Args:
        image_bgr: Картинка, по которой оценивается цвет бумаги
        brighten: На сколько тонов из 256 осветлить полученный цвет

    Returns:
        Цвет заливки (B, G, R)
    """
    return brighten_color(estimate_paper_color(image_bgr), brighten)


def _pad_jpeg(data: bytes, padding: int | None, brighten: int | None, dpi: int | None, source_label: str = "") -> bytes:
    """Добавить к JPEG поля и проставить разрешение, по возможности без перекодирования.

    Если беспотерьная вставка невозможна (нестандартный JPEG, нет jpegtran), картинка
    пересжимается с исходными таблицами квантования — это худший вариант, поэтому о нём
    сообщается предупреждением.

    Args:
        data: Байты исходного JPEG
        padding: Ширина полей в пикселях. None или 0 — только проставить разрешение
        brighten: На сколько тонов из 256 осветлить цвет заливки
        dpi: Разрешение, которое проставить. None — не трогать
        source_label: Откуда картинка (PDF и страница) — попадает в предупреждение об откате

    Returns:
        Байты результата
    """
    if not padding:
        return set_jpeg_dpi(data, dpi)

    # Цвет бумаги оцениваем по уменьшенной копии: draft() распаковывает JPEG прямо из
    # DCT-коэффициентов в уменьшенном виде, это заметно быстрее полного декодирования.
    # Просим размер не меньше того, на котором всё равно идёт анализ: на более мелкой
    # копии строки текста сливаются в серую массу и утягивают оценку в тень.
    with Image.open(BytesIO(data)) as source:
        source.draft("RGB", (PAPER_ANALYSIS_MAX_SIDE, PAPER_ANALYSIS_MAX_SIDE))
        preview = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    color = _fill_color(preview, brighten)

    try:
        padded, actual = pad_jpeg_lossless(data, padding, color, dpi)
        if actual != padding:
            logger.debug("Поле %d px округлено вверх до %d px (кратно размеру MCU)", padding, actual)
        return padded
    except LosslessPaddingError as e:
        logger.warning(
            "Перекодирование всей картинки: %s — беспотерьная вставка полей невозможна (%s)",
            source_label or "источник неизвестен",
            e,
        )

    with Image.open(BytesIO(data)) as source:
        quantization = source.quantization
        image = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    padded_image = pad_image_array(image, padding, color)
    buffer = BytesIO()
    Image.fromarray(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)).save(buffer, format="JPEG", qtables=quantization)
    return set_jpeg_dpi(buffer.getvalue(), dpi)


def _write_raster(
    image_bgr: np.ndarray, output_path: Path, padding: int | None, brighten: int | None, dpi: int | None
) -> None:
    """Записать растр в PNG с полями и, если попросили, с разрешением.

    Args:
        image_bgr: Картинка в BGR (или BGRA / градациях серого)
        output_path: Путь к выходному PNG
        padding: Ширина полей в пикселях. None или 0 — без полей
        brighten: На сколько тонов из 256 осветлить цвет заливки
        dpi: Разрешение, которое проставить. None — не трогать
    """
    if padding:
        image_bgr = pad_image_array(image_bgr, padding, _fill_color(image_bgr, brighten))

    if image_bgr.ndim == 2:
        pil_image = Image.fromarray(image_bgr, mode="L")
    elif image_bgr.shape[2] == 4:
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA))
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    save_kwargs = {"dpi": (dpi, dpi)} if dpi is not None else {}
    pil_image.save(output_path, format="PNG", **save_kwargs)


def _png_with_dpi(data: bytes, dpi: int | None) -> bytes:
    """Проставить PNG разрешение, перезаписав его (сжатие PNG беспотерьное).

    Args:
        data: Байты исходного PNG
        dpi: Разрешение, которое проставить. None — вернуть исходные байты как есть

    Returns:
        Байты PNG
    """
    if dpi is None:
        return data
    buffer = BytesIO()
    with Image.open(BytesIO(data)) as source:
        source.save(buffer, format="PNG", dpi=(dpi, dpi))
    return buffer.getvalue()


@dataclass(frozen=True)
class ExtractionReport:
    """Итог рекурсивного обхода.

    Attributes:
        counts: {относительный путь к PDF: сколько картинок извлечено}
        discarded: {относительный путь к PDF: чего в нём не хватило} — выпуски, выгрузка
            которых выброшена целиком
    """

    counts: dict[str, int]
    discarded: dict[str, str]


@dataclass(frozen=True)
class _PdfJob:
    """Одна задача для воркера: какой PDF куда разобрать и с какими параметрами."""

    pdf_path: Path
    output_subdir: Path
    pdf_basename: str
    padding: int | None
    brighten: int | None
    dpi: int | None
    discard_on_error: bool


def _run_job(job: _PdfJob) -> tuple[int, str | None, str | None]:
    """Выполнить одну задачу в воркере.

    Исключения не выпускаем наружу: один битый PDF не должен ронять весь пул.

    Args:
        job: Что и куда разбирать

    Returns:
        Кортеж (сколько картинок извлечено, текст ошибки или None, описание потерь или None)
    """
    try:
        count = extract_images_from_pdf(
            job.pdf_path,
            job.output_subdir,
            pdf_basename=job.pdf_basename,
            padding=job.padding,
            brighten=job.brighten,
            dpi=job.dpi,
            discard_on_error=job.discard_on_error,
        )
        return count, None, None
    except BrokenPdfError as e:
        return 0, None, describe_problems(e.problems, e.page_count)
    except Exception as e:
        return 0, str(e), None


def _remove_empty_dirs(root: Path) -> None:
    """Убрать опустевшие подпапки — например, год, все выпуски которого выброшены.

    Сам root не трогаем: его создал вызывающий.

    Args:
        root: Корень выгрузки
    """
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def extract_images_recursive(
    input_dir: Path,
    output_dir: Path,
    show_progress: bool = True,
    padding: int | None = None,
    brighten: int | None = None,
    dpi: int | None = None,
    jobs: int | None = None,
    discard_on_error: bool = True,
) -> ExtractionReport:
    """Рекурсивно извлечь изображения из всех PDF в директории.

    Для каждого найденного PDF создаётся поддиректория в output_dir с тем же относительным путём,
    что и у исходного PDF. Например, если PDF находится в input_dir/1936/1/3/5/mega.pdf,
    то изображения будут сохранены в output_dir/1936/1/3/5/mega/mega_0001.jpg и т.д.

    PDF разбираются параллельно, по одному на воркер: декодирование картинок и работа
    jpegtran упираются в процессор, а файлы друг от друга не зависят.

    Args:
        input_dir: Входная директория для поиска PDF-файлов
        output_dir: Выходная директория для сохранения изображений
        show_progress: Показывать ли прогресс-бар
        padding: Ширина добавляемых полей в пикселях. None или 0 — без полей
        brighten: На сколько тонов из 256 осветлить цвет заливки полей
        dpi: Разрешение, которое проставить выходным файлам. None — оставить как есть
        jobs: Сколько процессов запускать. None — по числу ядер, но не больше DEFAULT_MAX_JOBS
        discard_on_error: Выпуск, у которого потерялась хоть одна страница, выбрасывать
            целиком, а не оставлять половину

    Returns:
        Отчёт: сколько картинок вышло по каждому PDF и какие выпуски выброшены

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
        return ExtractionReport({}, {})

    queue = {}
    for pdf_path in pdf_files:
        rel_path = pdf_path.relative_to(input_dir)
        queue[str(rel_path)] = _PdfJob(
            pdf_path=pdf_path,
            output_subdir=output_dir / rel_path.parent / pdf_path.stem,
            pdf_basename=pdf_path.stem,
            padding=padding,
            brighten=brighten,
            dpi=dpi,
            discard_on_error=discard_on_error,
        )

    worker_count = max(1, min(jobs or (os.cpu_count() or 1), DEFAULT_MAX_JOBS, len(queue)))
    logger.info("Обработка %d PDF в %d процессов", len(queue), worker_count)

    results: dict[str, int] = {}
    discarded: dict[str, str] = {}
    if show_progress:
        progress = tqdm(
            total=len(queue),
            desc="Обработка PDF",
            unit="файл",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    else:
        progress = None

    def record(rel_path: str, count: int, error: str | None, losses: str | None) -> None:
        """Учесть результат одного PDF и обновить прогресс."""
        results[rel_path] = count
        if losses is not None:
            discarded[rel_path] = losses
            logger.error("✗ Выпуск %s выброшен целиком — %s", rel_path, losses)
        elif error is not None:
            logger.error("✗ Ошибка при обработке %s: %s", rel_path, error)
        elif not show_progress:
            logger.info("✓ [%d/%d] %s: извлечено %d изображений", len(results), len(queue), rel_path, count)
        if progress is not None:
            progress.update(1)
            progress.set_postfix_str(f"{Path(rel_path).name}: {count} изобр.")

    try:
        if worker_count == 1:
            for rel_path, job in queue.items():
                record(rel_path, *_run_job(job))
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = {pool.submit(_run_job, job): rel_path for rel_path, job in queue.items()}
                for future in as_completed(futures):
                    record(futures[future], *future.result())
    finally:
        if progress is not None:
            progress.close()

    _remove_empty_dirs(output_dir)
    return ExtractionReport(results, discarded)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--padding",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Ширина добавляемых полей в пикселях (по умолчанию — без полей). Поля заливаются "
        "оценённым цветом бумаги страницы. Для JPEG ширина округляется вверх до кратной "
        f"размеру MCU ({JPEG_DCT_BLOCK_SIZE} или {JPEG_MAX_MCU_SIZE} px) — тогда поля "
        "добавляются без перекодирования картинки."
    ),
)
@click.option(
    "--brighten",
    type=click.IntRange(min=0, max=255),
    default=None,
    help="Осветлить цвет заливки полей на N тонов из 256 (по умолчанию — не осветлять)",
)
@click.option(
    "--dpi",
    type=click.IntRange(min=1, max=65535),
    default=None,
    help=(
        "Проставить выходным файлам это разрешение, точек на дюйм (по умолчанию — "
        f"оставить как есть; обычное значение для сканов — {TARGET_DPI}). Это только тег: "
        "количество пикселей не меняется, картинка не масштабируется."
    ),
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=None,
    help=f"Сколько PDF обрабатывать параллельно (по умолчанию — по числу ядер, но не больше {DEFAULT_MAX_JOBS})",
)
@click.option(
    "--discard-broken/--keep-broken",
    default=True,
    help=(
        "Выпуск, у которого потерялась хоть одна страница (страница не открылась, в ней "
        "нет картинок, картинка не извлеклась), выбрасывать целиком, а не оставлять "
        "половину. По умолчанию включено: половинный выпуск легко принять за целый."
    ),
)
@click.option("--no-progress", is_flag=True, help="Не показывать прогресс-бар")
@click.option("-v", "--verbose", is_flag=True, help="Подробный вывод")
def main(
    input_dir: Path,
    output_dir: Path,
    padding: int | None,
    brighten: int | None,
    dpi: int | None,
    jobs: int | None,
    discard_broken: bool,
    no_progress: bool,
    verbose: bool,
):
    """Извлечение изображений из PDF без перекодирования.

    INPUT_DIR: Входная директория с PDF-файлами

    OUTPUT_DIR: Выходная директория для изображений

    \b
    Примеры:
      uv run python -m ocr_utils.pdf_utils /path/to/pdfs /path/to/output
      uv run python -m ocr_utils.pdf_utils ~/Documents/scans ~/Pictures/extracted --no-progress
      uv run python -m ocr_utils.pdf_utils /path/to/pdfs /path/to/output --padding 64 --brighten 10 --dpi 300
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")

    try:
        report = extract_images_recursive(
            input_dir,
            output_dir,
            show_progress=not no_progress,
            padding=padding,
            brighten=brighten,
            dpi=dpi,
            jobs=jobs,
            discard_on_error=discard_broken,
        )

        total_images = sum(report.counts.values())
        total_pdfs = len(report.counts)
        successful_pdfs = sum(1 for count in report.counts.values() if count > 0)

        print(f"\n{'=' * 60}")
        print(f"Готово!")
        print(f"Обработано PDF: {successful_pdfs}/{total_pdfs}")
        print(f"Извлечено изображений: {total_images}")

        if report.discarded:
            print()
            print(f"ВЫБРОШЕНО ЦЕЛИКОМ выпусков с потерянными страницами: {len(report.discarded)}")
            for rel_path, losses in sorted(report.discarded.items()):
                print(f"  {rel_path} — {losses}")
            print()
            print("  Их папки удалены: половинный выпуск легко принять за целый.")
            print("  Чтобы вместо этого оставлять, что вышло, — ключ --keep-broken.")

    except Exception as e:
        logger.error("Ошибка: %s", e)
        raise click.Abort()


if __name__ == "__main__":
    main()
