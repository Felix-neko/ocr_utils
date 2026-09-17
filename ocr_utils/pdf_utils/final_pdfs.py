"""Финальные PDF: распознанные страницы плюс возвращённые на место иллюстрации.

Вход — то, что вернул FineReader после пакетного распознания промежуточных PDF:

* полные PDF (бинаризация И распрямление строк) — из них берутся полосы БЕЗ иллюстраций:
  текст на них распознан лучше всего, а геометрия страницы больше не нужна;
* PDF типа PAGES_WITH_PICS_ONLY (бинаризация БЕЗ распрямления) — из них берутся полосы
  С иллюстрациями. Распрямления там нет намеренно: геометрия страницы осталась прежней,
  и размеченный в пикселях оригинала прямоугольник ложится на неё пересчётом масштаба.

На такую страницу поверх текстового слоя кладётся кусок оригинала — цветной или серый по
виду области, JPEG качества 75. Текстовый слой при этом не трогается вовсе: он остаётся
под картинкой и продолжает искаться, а видно на этом месте настоящую иллюстрацию, а не её
бинаризованный призрак.

Какая полоса какой странице распознанной PDF отвечает, известно из базы: номера страниц
записал туда сборщик промежуточных PDF. Поэтому первым делом сверяется количество страниц —
если FineReader полосу выбросил или добавил, все номера дальше врут, и собирать такой
выпуск нельзя.

Запуск::

    python -m ocr_utils.pdf_utils.final_pdfs --db base.sqlite --pack-name пак-1 \\
        --originals-dir .../blurred --full-pdf-dir .../full_recognized \\
        --pics-only-pdf-dir .../pics_only_recognized --final-dir .../final_pdfs
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import click
import pikepdf
from tqdm import tqdm

from ocr_utils.pdf_utils.intermediate_pdfs import DEFAULT_JOBS, IssuePlan, iter_with_progress, load_plans, safe_name
from ocr_utils.pdf_utils.jpeg_pdf import crop, encode_jpeg, load_image, overlay_on_page

logger = logging.getLogger(__name__)

# Качество JPEG у иллюстраций в ФИНАЛЬНОЙ PDF. Ниже, чем в промежуточной: этот файл читают,
# а не пересобирают, и весит он тем меньше, чем ниже качество врезок.
DEFAULT_JPEG_QUALITY = 75


def final_pdf_name(plan: IssuePlan) -> str:
    return f"{safe_name(plan.year_name)}_{safe_name(plan.issue_name)}.pdf"


@dataclass
class AssembleParams:
    """Параметры прогона, едущие в воркер вместе с планом выпуска."""

    originals_dir: Path
    full_pdf_dir: Path
    pics_only_pdf_dir: Path
    final_dir: Path
    by_year: bool = False
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    skip_if_exists: bool = True
    # Имена распознанных PDF, как их записал сборщик промежуточных: выпуск -> (полная,
    # с растром). Разрешаются в родителе, где есть база.
    names: "dict[int, tuple[str, str | None]]" = field(default_factory=dict)


@dataclass
class IssueReport:
    rel_path: str
    issue_id: int
    status: str = "ok"  # ok | skipped | error
    reason: str = ""
    pages: int = 0
    pictures: int = 0


@dataclass
class AssembleStats:
    issues: int = 0
    skipped: int = 0
    failed: int = 0
    pages: int = 0
    pictures: int = 0
    reports: "list[IssueReport]" = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"выпусков собрано {self.issues}, пропущено {self.skipped}, с ошибкой {self.failed}; "
            f"страниц {self.pages}, врезанных иллюстраций {self.pictures}"
        )


def _resolve(directory: Path, name: str) -> Path:
    """Файл в папке по имени; при промахе — поиск по основе имени без учёта регистра.

    FineReader пишет вывод пакета в отдельную папку, и как именно он обойдётся с именем —
    оставит, поменяет регистр расширения, — зависит от его настроек. Основу имени он не
    меняет, поэтому по ней и подстраховываемся.
    """
    direct = directory / name
    if direct.exists():
        return direct
    stem = Path(name).stem.lower()
    for candidate in sorted(directory.glob("*")):
        if candidate.is_file() and candidate.stem.lower() == stem:
            return candidate
    raise FileNotFoundError(f"в {directory} нет распознанной PDF {name}")


def assemble_issue(plan: IssuePlan, params: AssembleParams) -> IssueReport:
    """Собирает финальную PDF одного выпуска."""
    report = IssueReport(rel_path=plan.rel_path, issue_id=plan.issue_id)
    out_dir = params.final_dir / plan.year_name if params.by_year else params.final_dir
    out_path = out_dir / final_pdf_name(plan)
    if params.skip_if_exists and out_path.exists():
        report.status = "skipped"
        return report

    full_name, pics_name = params.names.get(plan.issue_id, (None, None))
    if full_name is None:
        return IssueReport(
            plan.rel_path,
            plan.issue_id,
            "error",
            "в базе нет имени промежуточной PDF — сначала нужен intermediate_pdfs",
        )

    try:
        full_src = pikepdf.Pdf.open(_resolve(params.full_pdf_dir, full_name))
        pics_src = pikepdf.Pdf.open(_resolve(params.pics_only_pdf_dir, pics_name)) if pics_name else None

        expected_full = len(plan.pages)
        expected_pics = sum(1 for p in plan.pages if p.pages_with_pics_only_pdf_page_idx is not None)
        if len(full_src.pages) != expected_full:
            raise ValueError(
                f"в распознанной {full_name} страниц {len(full_src.pages)}, а полос в выпуске {expected_full} — "
                "номера страниц в базе к ней не относятся"
            )
        if pics_src is not None and len(pics_src.pages) != expected_pics:
            raise ValueError(
                f"в распознанной {pics_name} страниц {len(pics_src.pages)}, а ожидалось {expected_pics} — "
                "номера страниц в базе к ней не относятся"
            )

        out = pikepdf.Pdf.new()
        for page_plan in plan.pages:
            pics_index = page_plan.pages_with_pics_only_pdf_page_idx
            if pics_index is None or pics_src is None:
                out.pages.append(full_src.pages[page_plan.full_pdf_page_idx])
                report.pages += 1
                continue

            out.pages.append(pics_src.pages[pics_index])
            page = out.pages[-1]
            original = load_image(params.originals_dir / page_plan.original_rel_path)
            for picture in page_plan.pictures:
                if picture.x2 <= picture.x1 or picture.y2 <= picture.y1:
                    continue
                jpeg = encode_jpeg(crop(original, picture.rect), picture.gray, params.jpeg_quality)
                overlay_on_page(out, page, jpeg, picture.rect, (page_plan.width, page_plan.height))
                report.pictures += 1
            del original
            report.pages += 1

        out_dir.mkdir(parents=True, exist_ok=True)
        out.save(out_path)
    except Exception as exc:  # noqa: BLE001 — один битый выпуск не должен ронять прогон
        logger.exception("Ошибка на выпуске %s", plan.rel_path)
        return IssueReport(plan.rel_path, plan.issue_id, "error", str(exc))

    return report


def _run_job(job: "tuple[IssuePlan, AssembleParams]") -> IssueReport:
    return assemble_issue(*job)


def load_pdf_names(db_path: Path, pack_name: str) -> "dict[int, tuple[str, str | None]]":
    """Имена промежуточных PDF по выпускам — их же ищем среди распознанных."""
    from ocr_utils.db.repo import require_pack
    from ocr_utils.db.session import open_db

    names: "dict[int, tuple[str, str | None]]" = {}
    with open_db(db_path, create=False)() as session:
        pack = require_pack(session, pack_name)
        for year in pack.year_packages:
            for issue in year.issues:
                if issue.full_intermediate_pdf_name:
                    names[issue.id] = (
                        issue.full_intermediate_pdf_name,
                        issue.pages_with_pics_only_intermediate_pdf_name,
                    )
    return names


def save_results(
    db_path: Path, pack_name: str, plans: "list[IssuePlan]", params: AssembleParams, built: "set[int]"
) -> None:
    """Записывает корень финальных PDF и имена собранных файлов."""
    from ocr_utils.db.repo import require_pack
    from ocr_utils.db.session import open_db

    by_issue = {plan.issue_id: plan for plan in plans}
    with open_db(db_path)() as session:
        pack = require_pack(session, pack_name)
        pack.final_pdfs_root = str(params.final_dir)
        for year in pack.year_packages:
            for issue in year.issues:
                plan = by_issue.get(issue.id)
                if plan is not None and issue.id in built:
                    issue.final_pdf_name = final_pdf_name(plan)
        session.commit()


def run_assemble(
    db_path: Path,
    pack_name: str,
    params: AssembleParams,
    *,
    only_year: "str | None" = None,
    only_issue: "str | None" = None,
    jobs: int = DEFAULT_JOBS,
    progress: bool = True,
) -> AssembleStats:
    """Полный прогон: план по базе, сборка пулом, запись имён обратно в базу."""
    plans = load_plans(db_path, pack_name, only_year=only_year, only_issue=only_issue)
    params.names = load_pdf_names(db_path, pack_name)
    stats = AssembleStats()
    built: "set[int]" = set()

    jobs_list = [(plan, params) for plan in plans]
    bar = tqdm(
        total=len(jobs_list),
        disable=not progress,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    with bar:
        if jobs > 1 and len(jobs_list) > 1:
            with ProcessPoolExecutor(max_workers=min(jobs, len(jobs_list))) as pool:
                results = list(iter_with_progress(pool.map(_run_job, jobs_list), bar))
        else:
            results = list(iter_with_progress((_run_job(job) for job in jobs_list), bar))

    for report in results:
        stats.reports.append(report)
        if report.status == "error":
            stats.failed += 1
            logger.error("Выпуск %s: %s", report.rel_path, report.reason)
            continue
        built.add(report.issue_id)
        stats.pages += report.pages
        stats.pictures += report.pictures
        if report.status == "skipped":
            stats.skipped += 1
        else:
            stats.issues += 1

    save_results(db_path, pack_name, plans, params, built)
    return stats


DIR_IN = click.Path(exists=True, file_okay=False, path_type=Path)
DIR_OUT = click.Path(file_okay=False, path_type=Path)
FILE_IN = click.Path(exists=True, dir_okay=False, path_type=Path)


@click.command()
@click.option("--db", "db_path", required=True, type=FILE_IN, help="База разметки.")
@click.option("--pack-name", required=True, help="Имя пака в базе.")
@click.option("--originals-dir", required=True, type=DIR_IN, help="Корень оригиналов: из них режутся иллюстрации.")
@click.option("--full-pdf-dir", required=True, type=DIR_IN, help="Распознанные полные PDF (после FineReader).")
@click.option("--pics-only-pdf-dir", required=True, type=DIR_IN, help="Распознанные PDF типа PAGES_WITH_PICS_ONLY.")
@click.option("--final-dir", required=True, type=DIR_OUT, help="Куда класть готовые PDF.")
@click.option(
    "--by-year/--no-by-year",
    default=False,
    show_default=True,
    help="Раскладывать финальные PDF по папкам годов или сложить все в одну папку.",
)
@click.option(
    "--jpeg-quality",
    default=DEFAULT_JPEG_QUALITY,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Качество врезок.",
)
@click.option("--only-year", default=None, help="Только этот год.")
@click.option("--only-issue", default=None, help="Только этот выпуск.")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, type=int, help="Воркеров; выпуск на воркер.")
@click.option("--skip-if-exists/--no-skip-if-exists", default=True, show_default=True)
@click.option("--no-progress", is_flag=True, help="Без полосы прогресса.")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(
    db_path: Path,
    pack_name: str,
    originals_dir: Path,
    full_pdf_dir: Path,
    pics_only_pdf_dir: Path,
    final_dir: Path,
    by_year: bool,
    jpeg_quality: int,
    only_year: "str | None",
    only_issue: "str | None",
    jobs: int,
    skip_if_exists: bool,
    no_progress: bool,
    log_level: str,
) -> None:
    """Собирает финальные PDF выпусков из распознанных промежуточных."""
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s: %(message)s")
    params = AssembleParams(
        originals_dir=originals_dir,
        full_pdf_dir=full_pdf_dir,
        pics_only_pdf_dir=pics_only_pdf_dir,
        final_dir=final_dir,
        by_year=by_year,
        jpeg_quality=jpeg_quality,
        skip_if_exists=skip_if_exists,
    )
    stats = run_assemble(
        db_path, pack_name, params, only_year=only_year, only_issue=only_issue, jobs=jobs, progress=not no_progress
    )
    click.echo(stats.summary())
    raise SystemExit(1 if stats.failed else 0)


if __name__ == "__main__":
    main()
