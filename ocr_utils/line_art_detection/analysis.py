"""Обход папки с PDF: рендер страниц, признаки, пул процессов.

УСТРОЙСТВО ПРОГОНА. Единица работы воркера — ЦЕЛЫЙ PDF, а не страница: документ
открывается один раз, и через pickle не ездят ни двадцативосьмимегапиксельные рендеры,
ни хэндлы PyMuPDF. Наружу из воркера едет только по строчке чисел на страницу.

GPU, ЕСЛИ ВКЛЮЧЁН, ЖИВЁТ В РОДИТЕЛЕ. Surya требует видеопамяти, а её одна на всех
(CLAUDE.md), поэтому при ``--use-surya-layout`` разметка страниц считается
последовательно в главном процессе, а в воркеры уезжают готовые прямоугольники — та же
схема, что в ``defocus_detection.analysis.detect_lines``. Побочная выгода: torch в
дочерние процессы не попадает вовсе.
"""

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz
from tqdm import tqdm

from ocr_utils.line_art_detection.features import LineArtParams, analyse_gray
from ocr_utils.line_art_detection.markup import PdfPageMarkup
from ocr_utils.line_art_detection.render import render_page

logger = logging.getLogger(__name__)

# Приоритет воркеров: прогон фоновый, машина должна оставаться отзывчивой.
WORKER_NICE = 10
# Переменные, которыми numpy/OpenCV ограничивают свои внутренние пулы потоков. Без них
# каждый воркер вправе развернуть по потоку на ядро, и полтора десятка процессов
# устраивают многократную перезапись — машина занята вся, считая при этом не быстрее.
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

DEFAULT_JOBS = 16

# Сколько файл должен пролежать неизменным, чтобы считаться дописанным. Значение и довод
# те же, что в ``pdf_utils.collect_sharpened``: из идущей выгрузки нельзя читать файл,
# который прямо сейчас пишут. Здесь это не теория — прогон по паку однажды пришёлся ровно
# на выгрузку FineReader в ту же папку, и число PDF под руками менялось 93 -> 89 -> 90,
# а число страниц в отчёте — 9170 -> 8774. Такую разницу в отчёте не отличить от правки
# порогов, поэтому свежие файлы отбрасываются, и о каждом пишется в лог.
DEFAULT_MIN_AGE_MINUTES = 10.0


def _init_worker() -> None:
    """Готовит процесс-воркер: один поток счёта и пониженный приоритет."""
    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        # Понижать приоритет можно всегда, но на всякий случай не падаем из-за политики.
        pass


@dataclass
class PageResult:
    """Итог по одной странице PDF; ровно то, что уезжает из воркера и ложится в CSV."""

    pdf: str
    page_no: int
    width: int
    height: int
    dpi: int
    coverage: float = 0.0
    n_candidates: int = 0
    n_components: int = 0
    max_cc_area: int = 0
    full_page: bool = False
    sources: str = ""
    boxes: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)
    status: str = "ok"

    @property
    def issue(self) -> str:
        """Имя выпуска — оно же имя PDF без расширения; по нему раскладывается экспорт."""
        return Path(self.pdf).stem


def analyse_pdf(
    pdf_path: Path,
    params: LineArtParams,
    markup: "dict[int, PdfPageMarkup] | None" = None,
    surya: "dict[int, tuple[list, list]] | None" = None,
) -> list[PageResult]:
    """Признаки всех страниц одного PDF.

    Args:
        pdf_path: Файл PDF.
        params: Пороги детектора.
        markup: Разметка растра по номеру страницы (0-based) — из :mod:`markup`.
        surya: Предложения и исключения Surya по номеру страницы (0-based).

    Returns:
        По записи на страницу; страница, которую не удалось прочитать, получает
        ``status`` с текстом ошибки и не теряется молча.
    """
    results: list[PageResult] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as error:  # noqa: BLE001 - битый файл не должен ронять весь прогон
        logger.error("Не открылся %s: %s", pdf_path, error)
        return [PageResult(str(pdf_path), 0, 0, 0, params.dpi, status=f"не открылся: {error}")]

    with doc:
        for index in range(doc.page_count):
            page_markup = (markup or {}).get(index)
            if page_markup is not None and page_markup.full_page:
                results.append(
                    PageResult(str(pdf_path), index + 1, 0, 0, params.dpi, status="полосная иллюстрация по разметке")
                )
                continue

            try:
                gray = render_page(doc[index], params.dpi)
            except Exception as error:  # noqa: BLE001
                logger.error("Не отрендерилась стр. %d из %s: %s", index + 1, pdf_path, error)
                results.append(
                    PageResult(str(pdf_path), index + 1, 0, 0, params.dpi, status=f"не отрендерилась: {error}")
                )
                continue

            exclude = list(page_markup.boxes_at(params.dpi)) if page_markup else []
            extra = []
            if surya and index in surya:
                proposals, surya_excludes = surya[index]
                extra = proposals
                exclude.extend(surya_excludes)

            findings = analyse_gray(gray, params, exclude_boxes=exclude, extra_boxes=extra)
            results.append(
                PageResult(
                    pdf=str(pdf_path),
                    page_no=index + 1,
                    width=gray.shape[1],
                    height=gray.shape[0],
                    dpi=params.dpi,
                    coverage=findings.coverage,
                    n_candidates=len(findings.candidates),
                    n_components=findings.n_components,
                    max_cc_area=findings.max_cc_area,
                    full_page=findings.full_page,
                    sources=",".join(sorted({c.source for c in findings.candidates})),
                    boxes=findings.boxes,
                    dropped=findings.dropped,
                )
            )
    return results


def _worker(args) -> list[PageResult]:
    return analyse_pdf(*args)


def collect_pdfs(
    input_dir: Path, recursive: bool = False, min_age_minutes: float = DEFAULT_MIN_AGE_MINUTES
) -> "tuple[list[Path], list[Path]]":
    """Файлы PDF в папке: ``(готовые, ещё пишущиеся)``.

    Свежий файл не читается вовсе — только ``stat``: у того, который прямо сейчас
    дописывают, узнаём время правки и ничего не открываем.

    Args:
        input_dir: Папка с PDF или один файл.
        recursive: Искать ли во вложенных папках.
        min_age_minutes: Моложе этого возраста файл считается недописанным; 0 — брать всё.

    Returns:
        Пара списков: годные к счёту и отложенные как слишком свежие.
    """
    if input_dir.is_file():
        return [input_dir], []

    found = sorted(p for p in input_dir.glob("**/*.pdf" if recursive else "*.pdf") if p.is_file())
    if min_age_minutes <= 0:
        return found, []

    deadline = time.time() - min_age_minutes * 60.0
    ready, fresh = [], []
    for path in found:
        (fresh if path.stat().st_mtime > deadline else ready).append(path)
    return ready, fresh


def detect_layout(pdfs: list[Path], params: LineArtParams, proposals, progress: bool = True) -> dict:
    """Прогон Surya по всем страницам — ЭТАП ДО пула процессов.

    Живёт в главном процессе намеренно: модели нужна видеопамять, а держать по копии в
    каждом из шестнадцати воркеров нельзя ни по памяти, ни по здравому смыслу.

    Returns:
        ``{путь PDF: {номер страницы: (предложения, исключения)}}``.
    """
    from ocr_utils.line_art_detection import layout as layout_module

    result: dict = {}
    for pdf_path in tqdm(pdfs, desc="Разметка страниц (GPU)", disable=not progress):
        pages: dict = {}
        try:
            with fitz.open(pdf_path) as doc:
                for index in range(doc.page_count):
                    gray = render_page(doc[index], params.dpi)
                    pages[index] = layout_module.split(proposals.boxes(pdf_path, index, gray))
        except Exception as error:  # noqa: BLE001
            logger.error("Разметка %s не удалась: %s", pdf_path, error)
        proposals.flush(pdf_path)
        result[str(pdf_path)] = pages
    return result


def analyse_folder(
    pdfs: list[Path],
    params: LineArtParams,
    markup: "dict[tuple[str, int], PdfPageMarkup] | None" = None,
    jobs: int = DEFAULT_JOBS,
    progress: bool = True,
    surya_by_pdf: "dict | None" = None,
) -> list[PageResult]:
    """Признаки всех страниц всех PDF; единица работы воркера — целый PDF.

    Args:
        pdfs: Список файлов.
        params: Пороги детектора.
        markup: Разметка пака по ``(имя PDF, номер страницы)``.
        jobs: Число процессов; 1 — считать в текущем процессе.
        progress: Показывать ли полосу прогресса.
        surya_by_pdf: Готовая разметка Surya (см. :func:`detect_layout`).

    Returns:
        Записи по всем страницам, в порядке файлов и страниц.
    """
    for name in THREAD_LIMIT_VARS:
        os.environ.setdefault(name, "1")

    def markup_for(pdf_path: Path) -> "dict[int, PdfPageMarkup] | None":
        if not markup:
            return None
        name = pdf_path.name
        return {page: value for (pdf, page), value in markup.items() if pdf == name} or None

    tasks = [(p, params, markup_for(p), (surya_by_pdf or {}).get(str(p))) for p in pdfs]

    results: list[PageResult] = []
    if jobs <= 1 or len(tasks) <= 1:
        for task in tqdm(tasks, desc="PDF", disable=not progress):
            results.extend(_worker(task))
        return results

    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), initializer=_init_worker, mp_context=context) as pool:
        for page_results in tqdm(pool.map(_worker, tasks), total=len(tasks), desc="PDF", disable=not progress):
            results.extend(page_results)
    return results
