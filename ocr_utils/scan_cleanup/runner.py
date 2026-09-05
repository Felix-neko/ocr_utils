"""Прогон по паку: чтение полосы, закрас, размытие, запись — за один проход.

ОДИН ПРОХОД ПО ФАЙЛУ. Оригиналы лежат на медленном NTFS-3G, пак весит 304 ГиБ.
Два отдельных прогона (сперва закрасить весь пак, потом размыть) означали бы
запись и повторное чтение ещё трёхсот гигабайт — дороже, чем весь счёт.

РАСКЛАДКА ПО ПРОЦЕССАМ. Закрас нужен 282 полосам из 12 135 (2.3%), размытие —
всем. Множества «нужен GPU» и «хватит CPU» не пересекаются, поэтому:

* полосы с масками идут ПОСЛЕДОВАТЕЛЬНО в родительском процессе, где живёт
  единственный ``GpuModels`` (видеопамять одна на всех, в пул её заворачивать
  нельзя);
* все остальные — в ``ProcessPoolExecutor``.

Картинка при этом вообще не пересекает границу процесса: воркер сам читает файл,
сам пишет результат, а из родителя получает только разметку — плоские датаклассы
из ``source``, где ни базы, ни ORM.

Тот же двухэтапный приём уже применён и обоснован в ``scan_markup.detection.run``,
только там этапы делят одну и ту же полосу, а здесь — разные.
"""

import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
from tqdm import tqdm

from ocr_utils.scan_cleanup.inpaint import InpaintOptions, inpaint_page
from ocr_utils.scan_cleanup.overlay import write_overlays
from ocr_utils.scan_cleanup.protect import ProtectOptions, analysis_roi, build_protect, is_full_page
from ocr_utils.scan_cleanup.smoothing import SmoothOptions, smooth_page
from ocr_utils.scan_cleanup.source import PageMarkup, load_markup
from ocr_utils.scan_cropping.image_io import imwrite_params, read_dpi, resolve_output_suffix, write_image

logger = logging.getLogger(__name__)

# Размер куска для пула. Полоса считается секундами, и мелкий кусок лучше ровняет
# хвост: при крупном последний воркер доделывал бы свою пачку в одиночку.
CHUNK_SIZE = 2


@dataclass
class CleanupParams:
    """Все настройки прогона в одном месте (значения опций CLI)."""

    db_path: Path
    pack_name: str
    pack_dir: Path
    out_dir: Path
    debug_dir: "Path | None" = None

    do_inpaint: bool = True
    do_smooth: bool = True

    only_year: "str | None" = None
    only_issue: "str | None" = None
    only_with_masks: bool = False
    pages_file: "Path | None" = None
    explicit_pages: "tuple[str, ...]" = ()
    limit: "int | None" = None

    skip_if_exists: bool = True
    output_format: "str | None" = None
    jobs: int = 8
    report_csv: "Path | None" = None

    inpaint: InpaintOptions = field(default_factory=InpaintOptions)
    smooth: SmoothOptions = field(default_factory=SmoothOptions)
    protect: ProtectOptions = field(default_factory=ProtectOptions)


@dataclass
class PageReport:
    """Что случилось с одной полосой."""

    rel_path: str
    status: str = "ok"  # ok | skipped | copied | missing | error
    reason: str = ""
    zones: int = 0
    zones_by_kind: "dict[str, int]" = field(default_factory=dict)
    dilate_px: float = 0.0
    blur_px: float = 0.0


def process_page(markup: PageMarkup, params: CleanupParams, models=None) -> PageReport:
    """Обрабатывает одну полосу: одно чтение, одна запись.

    ПОРЯДОК ВАЖЕН В ДВУХ МЕСТАХ:

    * серая версия считается ПОСЛЕ закраса. Иначе краска печати попала бы в маску
      контента и осталась бы неразмытой — защищали бы то, чего уже нет;
    * полосная иллюстрация (обложка, вкладка) пропускает размытие, но НЕ закрас:
      по паку-1 больше половины масок под удаление стоит именно на обложках.

    Запись атомарная (временный файл + ``os.replace``): прерванный прогон иначе
    оставил бы обрезанный TIFF, который ``--skip-if-exists`` потом считает готовым.
    Расширение у временного файла ТО ЖЕ, что у выходного: и cv2, и PIL выбирают
    кодек по нему, и `.part` в конце имени означал бы «формат неизвестен».
    """
    report = PageReport(markup.rel_path)
    src_path = markup.source_path(params.pack_dir)
    out_suffix = resolve_output_suffix(src_path.suffix, params.output_format)
    out_path = (params.out_dir / markup.rel_path).with_suffix(out_suffix)

    # Проверяем ДО imread: чтение 25-45 МБ TIFF заметно дороже, чем stat файла.
    if params.skip_if_exists and out_path.exists():
        report.status = "skipped"
        return report
    if not src_path.exists():
        report.status = "missing"
        report.reason = "нет файла на диске"
        return report

    bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if bgr is None:
        report.status = "error"
        report.reason = "не читается"
        return report
    dpi = read_dpi(src_path)
    before = bgr.copy() if params.debug_dir else None

    inpaint_report = None
    if params.do_inpaint and markup.masks:
        bgr, inpaint_report = inpaint_page(bgr, markup, params.inpaint, models)
        report.zones = inpaint_report.zones
        report.zones_by_kind = inpaint_report.counts()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    protect, rects = build_protect(gray.shape, markup, params.protect)

    result = bgr
    m_primary = m_dilated = None
    full_page = is_full_page(markup, params.protect)
    if full_page is not None:
        report.status = "copied"
        report.reason = f"полосная иллюстрация ({full_page.kind})"
    elif params.do_smooth:
        res = smooth_page(bgr, gray, protect if rects else None, analysis_roi(gray.shape, markup), params.smooth)
        result, m_primary, m_dilated = res.image, res.m_primary, res.m_dilated
        report.dilate_px, report.blur_px = res.dilate_px, res.blur_px
        if res.skip_reason:
            report.status = "copied"
            report.reason = res.skip_reason

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.stem}.part{out_path.suffix}")
    write_image(tmp_path, result, imwrite_params(out_suffix), dpi)
    os.replace(tmp_path, out_path)

    if params.debug_dir is not None:
        write_overlays(
            params.debug_dir,
            markup.rel_path,
            before,
            result,
            m_primary,
            m_dilated,
            rects,
            inpaint_report.masks if inpaint_report else None,
            inpaint_report.rois if inpaint_report else (),
        )
    return report


def _worker(args) -> PageReport:
    """Обёртка для пула: лямбды не пиклуются, а метод — тащил бы за собой объект."""
    markup, params = args
    try:
        return process_page(markup, params)
    except Exception as e:  # одна битая полоса не должна ронять прогон на 12 тысяч
        logger.exception("Ошибка на %s", markup.rel_path)
        return PageReport(markup.rel_path, status="error", reason=str(e))


def _read_pages_file(path: Path) -> "set[str]":
    """Список относительных путей: по строке на полосу, пустые и ``#`` игнорируются."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return {s.strip() for s in lines if s.strip() and not s.lstrip().startswith("#")}


def wanted_rel(pages_file: "Path | None", explicit: "tuple[str, ...]") -> "set[str] | None":
    """Набор относительных путей из ``--pages-file`` и ``--page`` вместе, либо ``None``."""
    wanted: "set[str]" = set(explicit)
    if pages_file is not None:
        wanted |= _read_pages_file(pages_file)
    return wanted or None


def select_pages(params: CleanupParams) -> "list[PageMarkup]":
    """Полосы к обработке с учётом всех фильтров."""
    pages = load_markup(
        params.db_path,
        params.pack_name,
        only_year=params.only_year,
        only_issue=params.only_issue,
        only_rel=wanted_rel(params.pages_file, params.explicit_pages),
    )
    if params.only_with_masks:
        pages = [p for p in pages if p.needs_inpaint]
    if params.limit is not None:
        pages = pages[: params.limit]
    return pages


def run_cleanup(params: CleanupParams) -> "list[PageReport]":
    """Полный прогон. Возвращает отчёты по всем полосам."""
    pages = select_pages(params)
    needs_gpu = [params.do_inpaint and p.needs_inpaint for p in pages]
    gpu_pages = [p for p, gpu in zip(pages, needs_gpu) if gpu]
    cpu_pages = [p for p, gpu in zip(pages, needs_gpu) if not gpu]
    logger.info(
        "Полос: %d (с закрасом %d, только размытие %d) | бэкенд %s | воркеров %d",
        len(pages),
        len(gpu_pages),
        len(cpu_pages),
        params.inpaint.backend,
        params.jobs,
    )

    reports: "list[PageReport]" = []

    if gpu_pages:
        # Модели грузятся один раз на весь этап; детекция страниц и пальцев здесь
        # не нужна — маски уже есть, их нарисовал человек.
        from ocr_utils.inpainting.backends import BACKEND_SD
        from ocr_utils.scan_cropping.gpu_models import GpuModels

        sd_model = params.inpaint.sd.model if params.inpaint.backend == BACKEND_SD else None
        with GpuModels(with_detection=False, with_lama=True, sd_model=sd_model) as models:
            for markup in tqdm(gpu_pages, desc="Закрас+размытие", unit="полоса"):
                try:
                    reports.append(process_page(markup, params, models))
                except Exception as e:
                    logger.exception("Ошибка на %s", markup.rel_path)
                    reports.append(PageReport(markup.rel_path, status="error", reason=str(e)))

    if cpu_pages:
        jobs = max(1, params.jobs)
        if jobs == 1:
            for markup in tqdm(cpu_pages, desc="Размытие", unit="полоса"):
                reports.append(_worker((markup, params)))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                tasks = ((markup, params) for markup in cpu_pages)
                for report in tqdm(
                    pool.map(_worker, tasks, chunksize=CHUNK_SIZE), total=len(cpu_pages), desc="Размытие", unit="полоса"
                ):
                    reports.append(report)

    if params.report_csv is not None:
        write_report(reports, params.report_csv)
    return reports


def write_report(reports: "list[PageReport]", path: Path) -> None:
    """Отчёт по полосам в CSV — в первую очередь чтобы просмотреть «скопировано без изменений»."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(PageReport("")).keys()))
        writer.writeheader()
        for r in reports:
            row = asdict(r)
            row["zones_by_kind"] = ", ".join(f"{k}={v}" for k, v in sorted(r.zones_by_kind.items()))
            writer.writerow(row)


def summary(reports: "list[PageReport]") -> str:
    """Сводка прогона одной строкой на пункт."""
    by_status: "dict[str, int]" = {}
    zones: "dict[str, int]" = {}
    for r in reports:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        for kind, n in r.zones_by_kind.items():
            zones[kind] = zones.get(kind, 0) + n

    lines = [f"Полос обработано: {len(reports)}"]
    for status, label in (
        ("ok", "размыто"),
        ("copied", "скопировано без изменений"),
        ("skipped", "пропущено (выход уже есть)"),
        ("missing", "нет на диске"),
        ("error", "ошибок"),
    ):
        if by_status.get(status):
            lines.append(f"  {label}: {by_status[status]}")
    if zones:
        lines.append("Зон закраса: " + ", ".join(f"{k} — {v}" for k, v in sorted(zones.items())))
    return "\n".join(lines)
