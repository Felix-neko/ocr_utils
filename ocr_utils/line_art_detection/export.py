"""Выгрузка найденных страниц картинками — по папке на выпуск.

ЧТО НА КАРТИНКЕ. Вся полоса целиком плюс рамка вокруг находки. Целиком — потому что
проверяется не сама картинка, а не поехала ли она ОТНОСИТЕЛЬНО ТЕКСТА: по вырезу одной
только находки покорёженность видно хуже, сравнивать не с чем. Рамка — потому что иначе
находку на полосе ещё надо глазами искать.

ПОЧЕМУ ЭТО ОТДЕЛЬНАЯ КОМАНДА. Прямоугольники уже лежат в CSV, поэтому выгрузка ничего не
считает заново: рендер плюс рисование рамки. Порог можно менять сколько угодно раз, не
трогая тяжёлый проход (см. ``cli``).
"""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import fitz
from tqdm import tqdm

from ocr_utils.line_art_detection.analysis import PageResult, _init_worker
from ocr_utils.line_art_detection.render import render_page

logger = logging.getLogger(__name__)

# Цвет и толщина рамки. Толщина считается от длинной стороны выгружаемой картинки, а не
# от кадра: на уменьшенной копии рамка в один пиксель оригинала попросту исчезла бы.
BOX_COLOR = (0, 0, 255)
BOX_WIDTH_FRACTION = 1 / 400


def _render_dpi(result: PageResult, long_side: int) -> float:
    """Разрешение, при котором длинная сторона страницы даёт ровно ``long_side`` px."""
    longest = max(result.width, result.height) or 1
    return result.dpi * long_side / longest


def export_page(result: PageResult, out_dir: Path, long_side: int, quality: int) -> "Path | None":
    """Одна страница: рендер в нужный размер, рамки вокруг находок, JPEG на диск."""
    scale = long_side / max(1, max(result.width, result.height))
    try:
        with fitz.open(result.pdf) as doc:
            gray = render_page(doc[result.page_no - 1], _render_dpi(result, long_side))
    except Exception as error:  # noqa: BLE001 - одна битая страница не должна валить выгрузку
        logger.error("Не отрендерилась стр. %d из %s: %s", result.page_no, result.pdf, error)
        return None

    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thickness = max(2, int(round(long_side * BOX_WIDTH_FRACTION)))
    for x1, y1, x2, y2 in result.boxes:
        top_left = (int(x1 * scale), int(y1 * scale))
        bottom_right = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(image, top_left, bottom_right, BOX_COLOR, thickness)

    folder = out_dir / result.issue
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result.issue}_p{result.page_no:04d}_cov{result.coverage:.3f}.jpg"
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def _worker(args) -> int:
    return 1 if export_page(*args) else 0


def export_pages(results: list[PageResult], out_dir: Path, long_side: int, quality: int, jobs: int) -> int:
    """Выгружает страницы пулом процессов; возвращает число записанных файлов."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(r, out_dir, long_side, quality) for r in results]

    if jobs <= 1 or len(tasks) <= 1:
        return sum(_worker(t) for t in tqdm(tasks, desc="Страницы"))

    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), initializer=_init_worker, mp_context=context) as pool:
        return sum(tqdm(pool.map(_worker, tasks), total=len(tasks), desc="Страницы"))
