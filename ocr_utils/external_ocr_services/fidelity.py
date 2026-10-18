"""Сверка ответа модели с tesseract той же полосы: ловит полосы, где модель пересказала текст, а не прочла.

Зачем. Модель может вернуть валидный JSON со связным, правдоподобным текстом, которого на полосе нет:
2026-10-18 пак-2 ушёл к провайдеру OpenInference, и тот на ~16 % полос писал «по теме» — выдуманные
заводы, даты, номера писем. Конвейер этого не видел: схема ответа соблюдена, сбоев нет. tesseract
читает хуже модели, но не сочиняет, поэтому совпадение с ним — грубая, но честная мера «прочитано ли».

Мера — доля троек подряд идущих слов ответа (слова от 5 букв, без регистра, ё = е), найденных среди
троек tesseract той же полосы. Отдельные слова пересказ сохраняет («поставщик», «договора»), тройки —
нет. Короткие слова отброшены: их tesseract и модель путают чаще всего, а смысла для меры в них мало.

Калибровка (выборки глазами; подробнее — ``README.md`` пакета, раздел «Сверка с tesseract»):

* пак-1 (провайдер DeepSeek), 783 случайные полосы: медиана 0,96, ниже 0,5 — 1 %, и все такие —
  промахи самого tesseract (фото, слитые колонки, таблицы), а не модели;
* пак-2 через OpenInference: медиана 0,80, ниже 0,5 — 16 %, это пересказы (0,03–0,45);
* пак-2 через DeepSeek: медиана 0,97, ниже 0,5 — одна полоса (оглавление, tesseract слил колонки).

Поэтому порог подозрения — 0,5: выше него пересказов не встречалось, ниже — у правильного провайдера
почти только оглавления и таблицы, и их немного, чтобы посмотреть глазами.

Результат — ``fidelity.csv`` в корне pages-dir, по строке на полосу, худшие сверху. Готовые строки
берутся из прошлого ``fidelity.csv``, если ``.md`` полосы с тех пор не менялся: повторный прогон
``--skip-done`` не гоняет tesseract по всему паку заново.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import ImageOps

from ocr_utils.external_ocr_services.tiling import load
from ocr_utils.scan_markup import tesseract

logger = logging.getLogger(__name__)

# Порог подозрения: доля троек ответа, найденных у tesseract, ниже — полосу смотреть глазами (см. докстринг модуля).
SUSPECT_SHARE = 0.5
# Меньше стольких слов (от 5 букв) — мера не считается: колонтитул, подпись под фото, пустая полоса.
MIN_WORDS = 50
# Файл результата в корне pages-dir.
REPORT_NAME = "fidelity.csv"
REPORT_FIELDS = ("page", "image", "words", "share", "suspect", "md_mtime_ns")

# Слово меры: кириллица от 5 букв (после приведения к нижнему регистру и замены ё на е).
_WORD = re.compile(r"[а-я]{5,}")
# Перенос в выводе tesseract: дефис в конце строки склеивается со следующей строкой.
_TESSERACT_HYPHEN = re.compile(r"-\n")


@dataclass(frozen=True)
class PageFidelity:
    """Мера одной полосы.

    Args:
        page: Путь ``.md`` полосы относительно pages-dir (``1977/02/0430_1L.md``).
        image: Путь входной картинки относительно in-dir — её читал tesseract.
        words: Слов меры в ответе модели; меньше ``MIN_WORDS`` — мера не считалась (``share = None``).
        share: Доля троек слов ответа, найденных у tesseract, 0..1; ``None`` — слишком мало слов.
        md_mtime_ns: Время изменения ``.md`` на момент сверки — по нему строка переиспользуется.
    """

    page: str
    image: str
    words: int
    share: float | None
    md_mtime_ns: int

    @property
    def suspect(self) -> bool:
        """Полоса подозрительна: мера посчитана и ниже ``SUSPECT_SHARE``."""
        return self.share is not None and self.share < SUSPECT_SHARE


def measure_words(text: str) -> list[str]:
    """Слова меры по порядку: кириллица от 5 букв, нижний регистр, ё → е.

    Args:
        text: Любой текст — markdown модели (теги и разметка отсеиваются сами: в них нет кириллицы)
            или вывод tesseract.

    Returns:
        Список слов в порядке появления.
    """
    return _WORD.findall(text.lower().replace("ё", "е"))


def shingles(words: list[str]) -> set[tuple[str, str, str]]:
    """Тройки подряд идущих слов.

    Args:
        words: Слова меры по порядку.

    Returns:
        Множество троек; у списка короче трёх слов — пустое.
    """
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def shingle_share(model_text: str, reference_text: str) -> tuple[int, float | None]:
    """Мера совпадения ответа модели с эталонным чтением.

    Args:
        model_text: Текст модели (тело ``.md`` полосы).
        reference_text: Чтение tesseract той же полосы.

    Returns:
        ``(слов меры у модели, доля её троек, найденных у эталона)``; доля ``None``, если слов
        меньше ``MIN_WORDS``.
    """
    words = measure_words(model_text)
    if len(words) < MIN_WORDS:
        return len(words), None
    model_shingles = shingles(words)
    reference = shingles(measure_words(_TESSERACT_HYPHEN.sub("", reference_text)))
    return len(words), round(len(model_shingles & reference) / len(model_shingles), 3)


def md_body(md_path: Path) -> str:
    """Тело ``.md`` полосы без YAML-шапки: в шапке колонтитулы и заголовки, их нет смысла сверять дважды.

    Args:
        md_path: Файл полосы.

    Returns:
        Текст после закрывающего ``---`` шапки; файл без шапки — целиком.
    """
    text = md_path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :]
    return text


def tesseract_text(image_path: Path) -> str:
    """Чтение полосы tesseract (русский, разметка страницы по умолчанию).

    Args:
        image_path: Входная картинка полосы — та же, что ушла в модель.

    Returns:
        Текст tesseract; пустая строка, если он не справился (тогда мера выйдет 0 и полоса попадёт в
        подозрительные — лучше лишний взгляд, чем пропуск).
    """
    gray = np.asarray(ImageOps.grayscale(load(image_path)))
    return tesseract.run(gray, ["-l", tesseract.LANGUAGE], prefix="fidelity_")


def page_jobs(pages_dir: Path, only_year: str | None = None, only_issue: str | None = None) -> list[tuple[Path, str]]:
    """Полосы для сверки: ``.md`` и путь входной картинки из соседнего ``.meta.json``.

    Имя ``.md`` и входа могут расходиться (у очищенных полос пака-2 во входе отпечаток:
    ``0430_1L.57bcb874.tif`` → ``0430_1L.md``), поэтому путь входа берётся из поля ``page`` meta,
    а не угадывается по имени.

    Args:
        pages_dir: Папка полос внешнего OCR (``--pages-dir``).
        only_year: Только этот годовой комплект (имя папки).
        only_issue: Только этот выпуск (имя папки).

    Returns:
        Пары ``(путь .md, путь входа относительно in-dir)``, отсортированные по пути ``.md``.
    """
    jobs: list[tuple[Path, str]] = []
    for meta_path in sorted(pages_dir.glob("*/*/*.meta.json")):
        year, issue = meta_path.parent.parent.name, meta_path.parent.name
        if (only_year and year != only_year) or (only_issue and issue != only_issue):
            continue
        md_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".md")
        if not md_path.is_file():
            continue  # полоса со сбоем: ответа нет, сверять нечего
        image = json.loads(meta_path.read_text(encoding="utf-8")).get("page")
        if image:
            jobs.append((md_path, image))
    return jobs


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и с пониженным приоритетом, tesseract сам в один поток."""
    try:
        np._core.multiarray._set_madvise_hugepage(False)
    except AttributeError:
        pass
    try:
        os.nice(10)
    except OSError:
        pass


def check_page(pages_dir: Path, in_dir: Path, md_path: Path, image: str) -> PageFidelity:
    """Мера одной полосы (выполняется в воркере пула).

    Args:
        pages_dir: Папка полос — для относительного пути ``.md`` в отчёте.
        in_dir: Корень входа модели.
        md_path: Файл ответа полосы.
        image: Путь входной картинки относительно ``in_dir``.

    Returns:
        Мера полосы; нет входной картинки — эталон пустой, мера 0.
    """
    image_path = in_dir / image
    reference = tesseract_text(image_path) if image_path.is_file() else ""
    words, share = shingle_share(md_body(md_path), reference)
    return PageFidelity(md_path.relative_to(pages_dir).as_posix(), image, words, share, md_path.stat().st_mtime_ns)


def read_report(path: Path) -> dict[str, PageFidelity]:
    """Прошлый ``fidelity.csv`` по пути ``.md``; нет файла — пустой словарь.

    Args:
        path: Файл отчёта.

    Returns:
        Строки отчёта, ключ — ``page``.
    """
    if not path.is_file():
        return {}
    rows: dict[str, PageFidelity] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            share = float(row["share"]) if row["share"] else None
            rows[row["page"]] = PageFidelity(
                row["page"], row["image"], int(row["words"]), share, int(row["md_mtime_ns"])
            )
    return rows


def write_report(path: Path, rows: list[PageFidelity]) -> Path:
    """Записать ``fidelity.csv``: худшие полосы сверху, без меры — в конце.

    Args:
        path: Файл отчёта.
        rows: Меры полос.

    Returns:
        Путь к записанному файлу.
    """
    ordered = sorted(rows, key=lambda row: (row.share is None, row.share or 0.0, row.page))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REPORT_FIELDS)
        for row in ordered:
            share = "" if row.share is None else f"{row.share:.3f}"
            writer.writerow((row.page, row.image, row.words, share, int(row.suspect), row.md_mtime_ns))
    return path


def check_pages(
    pages_dir: Path, in_dir: Path, jobs: int = 16, only_year: str | None = None, only_issue: str | None = None
) -> list[PageFidelity]:
    """Сверить ответы модели с tesseract по папке полос и обновить ``fidelity.csv``.

    Строки прошлого отчёта, у которых ``.md`` не менялся, берутся как есть; остальные полосы
    считаются заново в пуле процессов (tesseract — CPU, по полосе на воркер). Строки других
    выпусков из прошлого отчёта сохраняются, чтобы прогон по одному выпуску не стирал остальные.

    Args:
        pages_dir: Папка полос внешнего OCR (``--pages-dir``), в её корень пишется ``fidelity.csv``.
        in_dir: Корень входа модели (``--in-dir``) — те же картинки, что видела модель.
        jobs: Число воркеров пула.
        only_year: Только этот годовой комплект.
        only_issue: Только этот выпуск.

    Returns:
        Меры полос отобранного подмножества (и пересчитанных, и взятых из отчёта).
    """
    report_path = pages_dir / REPORT_NAME
    previous = read_report(report_path)
    reused: list[PageFidelity] = []
    todo: list[tuple[Path, str]] = []
    for md_path, image in page_jobs(pages_dir, only_year, only_issue):
        old = previous.get(md_path.relative_to(pages_dir).as_posix())
        # Строка годится, если ответ полосы тот же (mtime .md) и вход тот же
        if old is not None and old.md_mtime_ns == md_path.stat().st_mtime_ns and old.image == image:
            reused.append(old)
        else:
            todo.append((md_path, image))
    logger.info("сверка с tesseract: полос %d, из прошлого отчёта %d", len(reused) + len(todo), len(reused))
    fresh: list[PageFidelity] = []
    if todo and not tesseract.tesseract_available():
        logger.warning("tesseract не найден: сверка ответов модели пропущена")
        return reused
    if todo:
        with ProcessPoolExecutor(
            max_workers=max(1, jobs), mp_context=get_context("forkserver"), initializer=_init_worker
        ) as pool:
            futures = [pool.submit(check_page, pages_dir, in_dir, md_path, image) for md_path, image in todo]
            fresh = [future.result() for future in futures]
    checked = reused + fresh
    # В отчёт — и строки вне отбора (другие выпуски), чтобы частичный прогон их не терял
    selected = {row.page for row in checked}
    kept = [row for page, row in previous.items() if page not in selected]
    write_report(report_path, kept + checked)
    return checked


def summarize(rows: list[PageFidelity]) -> str:
    """Строка итога для лога и терминала: сколько сверено, медиана, сколько подозрительных.

    Args:
        rows: Меры полос.

    Returns:
        Человекочитаемая сводка; подозрительные перечислены поимённо (до 20).
    """
    shares = sorted(row.share for row in rows if row.share is not None)
    if not shares:
        return f"сверка с tesseract: полос {len(rows)}, мера не посчитана ни у одной"
    suspects = sorted((row for row in rows if row.suspect), key=lambda row: row.share or 0.0)
    text = (
        f"сверка с tesseract: полос {len(rows)}, с мерой {len(shares)}, медиана {shares[len(shares) // 2]:.2f}, "
        f"ниже {SUSPECT_SHARE} — {len(suspects)}"
    )
    if suspects:
        names = ", ".join(f"{row.page} ({row.share:.2f})" for row in suspects[:20])
        text += f": {names}" + (" …" if len(suspects) > 20 else "")
    return text
