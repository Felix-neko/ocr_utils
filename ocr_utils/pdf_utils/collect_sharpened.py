"""Сбор выгрузки Capture One из подпапок ``sharpened`` в обычное дерево пака.

Capture One складывает обработанные полосы В ПОДПАПКУ рядом с исходными, то есть
``{год}/{выпуск}/sharpened/IMG_0003_2R.jpg`` при исходном
``{год}/{выпуск}/IMG_0003_2R.tif``. Дальше по конвейеру такое дерево неудобно: все прочие
шаги ждут раскладку ``{год}/{выпуск}/полоса``, и относительный путь полосы в базе выглядит
именно так. Этот скрипт переносит картинки в отдельный корень с привычной раскладкой и
убирает за собой опустевшие подпапки.

ПРО ЕЩЁ ПИШУЩИЕСЯ ФАЙЛЫ. Выгрузка идёт часами, и запускать сбор хочется, не дожидаясь её
конца. Опасность в том, что наполовину записанный JPEG на диске неотличим от готового: имя
есть, размер ненулевой, а обрезан он или нет, видно только по концу файла. Поэтому файл
моложе ``--min-age-minutes`` (по умолчанию 10) не трогается ВООБЩЕ — ни на чтение, ни на
перенос — и считается пока не приехавшим.

ПРО ПОЛОВИНЧАТЫЕ ВЫПУСКИ. По той же причине выпуск переносится только целиком: набор
полос в ``sharpened`` сверяется с набором полос выпуска. Половинный выпуск потом ничем не
отличить от целого, и обнаружился бы он в лучшем случае в PDF, где просто нет двадцати
страниц. Отключается ключом ``--no-require-complete``.

Скрипт идемпотентен и рассчитан на повторные запуски: пока Capture One не закончил,
правильный способ им пользоваться — просто запускать его ещё раз, дозабирая доехавшее.

Запуск::

    python -m ocr_utils.pdf_utils.collect_sharpened ИСХОДНЫЙ_КОРЕНЬ КОРЕНЬ_НАЗНАЧЕНИЯ
    python -m ocr_utils.pdf_utils.collect_sharpened ... --db base.sqlite --pack-name пак-1
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import click

from ocr_utils.scan_cleanup.naming import hash_suffix, split_cleaned_stem
from ocr_utils.scan_markup.scan_tree import IGNORED_DIRS, YEAR_RE, is_ignored_dir, issue_images

logger = logging.getLogger(__name__)

DEFAULT_SUBDIR = "sharpened"
# Сколько файл должен пролежать неизменным, чтобы считаться дописанным. Десять минут — с
# запасом: одна полоса 3830x5892 выгружается из Capture One за десятки секунд.
DEFAULT_MIN_AGE_MINUTES = 10.0
DEFAULT_JOBS = 8


@dataclass
class IssuePlan:
    """Что делать с одним выпуском."""

    rel_path: str
    subdir: Path
    dest_dir: Path
    ready: "list[Path]" = field(default_factory=list)
    too_fresh: "list[Path]" = field(default_factory=list)
    missing: "list[str]" = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing and not self.too_fresh


@dataclass
class CollectReport:
    """Итоги прогона."""

    dry_run: bool = False
    moved: int = 0
    issues_done: int = 0
    issues_skipped: int = 0
    too_fresh: int = 0
    missing: int = 0
    dirs_removed: int = 0
    dirs_left: "list[str]" = field(default_factory=list)
    db_filled: int = 0
    db_without_pair: int = 0

    def summary(self) -> str:
        parts = [
            f"перенесено полос {self.moved}",
            f"выпусков собрано {self.issues_done}",
            f"пропущено выпусков {self.issues_skipped}",
            f"ещё пишется {self.too_fresh}",
            f"не выгружено {self.missing}",
        ]
        if self.dirs_left:
            parts.append(f"папок осталось непустыми {len(self.dirs_left)}")
        if self.db_filled or self.db_without_pair:
            parts.append(f"в базе заполнено {self.db_filled}, без пары {self.db_without_pair}")
        # При --dry-run счётчики говорят, что БЫЛО БЫ сделано; без оговорки отчёт врёт.
        return ("ВХОЛОСТУЮ: " if self.dry_run else "") + "; ".join(parts)


def plan_issues(
    source_root: Path, dest_root: Path, subdir: str = DEFAULT_SUBDIR, min_age_minutes: float = DEFAULT_MIN_AGE_MINUTES
) -> "list[IssuePlan]":
    """Что и куда переносить, с разбором «готово / ещё пишется / не выгружено».

    Обход тот же, что у :func:`scan_tree.scan_pack`: год — папка, начинающаяся с четырёх
    цифр, выпуск — папка внутри неё, служебные каталоги (``cache``) отбрасываются. Иначе
    в перенос попали бы миниатюры ScanTailor и проекты ``76_01.ScanTailor``.
    """
    deadline = time.time() - min_age_minutes * 60.0
    plans: "list[IssuePlan]" = []

    for year_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if not YEAR_RE.match(year_dir.name):
            continue
        for issue_dir in sorted(p for p in year_dir.iterdir() if p.is_dir() and not is_ignored_dir(p)):
            source_subdir = issue_dir / subdir
            if not source_subdir.is_dir():
                continue
            plan = IssuePlan(
                rel_path=f"{year_dir.name}/{issue_dir.name}",
                subdir=source_subdir,
                dest_dir=dest_root / year_dir.name / issue_dir.name,
            )
            for path in issue_images(source_subdir):
                # stat, а не open: у файла, который прямо сейчас дописывают, узнаём только
                # время правки и ничего не читаем.
                if path.stat().st_mtime > deadline:
                    plan.too_fresh.append(path)
                else:
                    plan.ready.append(path)

            done = {path.stem for path in plan.ready}
            expected = {path.stem for path in issue_images(issue_dir)}
            plan.missing = sorted(expected - done - {path.stem for path in plan.too_fresh})
            plans.append(plan)

    return plans


def _transfer(source: Path, dest: Path, move: bool) -> None:
    """Перенос одного файла; при ``move`` внутри одной ФС это мгновенный ``rename``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not move:
        shutil.copy2(source, dest)
        return
    try:
        os.replace(source, dest)
    except OSError:
        # Разные файловые системы: ``rename`` между ними невозможен, копируем и удаляем.
        shutil.move(str(source), str(dest))


def collect(
    source_root: Path,
    dest_root: Path,
    *,
    subdir: str = DEFAULT_SUBDIR,
    move: bool = True,
    min_age_minutes: float = DEFAULT_MIN_AGE_MINUTES,
    require_complete: bool = True,
    jobs: int = DEFAULT_JOBS,
    dry_run: bool = False,
) -> "tuple[CollectReport, dict[str, str]]":
    """Переносит готовые выпуски; возвращает отчёт и карту ``rel_path оригинала -> имя копии``.

    Ключ карты — ``выпуск/признак полосы``, где признак это ОТПЕЧАТОК из имени файла
    (``1970/01/a1b2c3d4``), а если отпечатка в имени нет — основа имени (``1970/01/0010``).
    Отпечаток надёжнее: имена полос в разных выпусках повторяются, и именно на этом уже
    один раз сломалась выгрузка Capture One (см. ``scan_cleanup.naming``).

    Имя копии в карте настоящее, взятое с диска, а не собранное подстановкой суффикса:
    следующий пак может выгружаться в другой формат, и подстановка молча промахнулась бы.
    """
    report = CollectReport(dry_run=dry_run)
    moved_names: "dict[str, str]" = {}
    plans = plan_issues(source_root, dest_root, subdir=subdir, min_age_minutes=min_age_minutes)

    for plan in plans:
        report.too_fresh += len(plan.too_fresh)
        report.missing += len(plan.missing)
        if require_complete and not plan.complete:
            report.issues_skipped += 1
            logger.warning(
                "Выпуск %s ещё не готов: ещё пишется %d, не выгружено %d%s",
                plan.rel_path,
                len(plan.too_fresh),
                len(plan.missing),
                f" ({', '.join(plan.missing[:5])}{'...' if len(plan.missing) > 5 else ''})" if plan.missing else "",
            )
            continue
        if not plan.ready:
            report.issues_skipped += 1
            continue

        pairs = [(path, plan.dest_dir / path.name) for path in plan.ready]
        if not dry_run:
            if jobs > 1 and not move:
                # Копирование упирается в диск, а не в счёт, поэтому пул ПОТОКОВ: процессы
                # тут дали бы только накладные расходы на их запуск.
                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    list(pool.map(lambda pair: _transfer(pair[0], pair[1], move), pairs))
            else:
                for source, dest in pairs:
                    _transfer(source, dest, move)

        for source, _dest in pairs:
            stem, digest = split_cleaned_stem(source.stem)
            moved_names[f"{plan.rel_path}/{digest or stem}"] = source.name
        report.moved += len(pairs)
        report.issues_done += 1

        if move and not dry_run:
            # rmdir, а не rm -r: непустая папка должна остаться и попасть в отчёт. В ней
            # может лежать что-то, чего мы не ждали, и стирать это молча нельзя.
            leftovers = [p for p in plan.subdir.iterdir() if p.name.lower() not in IGNORED_DIRS]
            if leftovers:
                report.dirs_left.append(plan.rel_path)
                logger.warning("Папка %s не пуста после переноса (%d файлов)", plan.subdir, len(leftovers))
            else:
                shutil.rmtree(plan.subdir)
                report.dirs_removed += 1

    return report, moved_names


def fill_database(
    db_path: Path, pack_name: str, dest_root: Path, moved_names: "dict[str, str]", report: CollectReport
) -> None:
    """Записывает в базу корень заострённых копий и имена файлов по полосам."""
    from ocr_utils.db.repo import iter_pages, require_pack
    from ocr_utils.db.session import open_db

    session_factory = open_db(db_path)
    with session_factory() as session:
        pack = require_pack(session, pack_name)
        pack.sharpened_text_pics_root = str(dest_root)
        for _year, _issue, page in iter_pages(pack):
            # Ищем сперва по отпечатку, и только потом по имени: у пака, очищенного до
            # появления отпечатков в именах, второй путь остаётся единственным.
            issue_rel = Path(page.source_rel_path).parent.as_posix()
            stem = Path(page.source_rel_path).stem
            name = None
            if page.file_hash:
                name = moved_names.get(f"{issue_rel}/{hash_suffix(page.file_hash)}")
            if name is None:
                name = moved_names.get(f"{issue_rel}/{stem}")
            if name is None:
                if page.sharpened_text_pic_file_name is None:
                    report.db_without_pair += 1
                continue
            page.sharpened_text_pic_file_name = name
            page.sharpened_text_pic_rel_path = f"{Path(page.source_rel_path).parent.as_posix()}/{name}"
            report.db_filled += 1
        session.commit()


@click.command()
@click.argument("source_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("dest_root", type=click.Path(file_okay=False, path_type=Path))
@click.option("--subdir", default=DEFAULT_SUBDIR, show_default=True, help="Имя подпапки с выгрузкой Capture One.")
@click.option("--move/--copy", "move", default=True, show_default=True, help="Переносить или копировать файлы.")
@click.option(
    "--min-age-minutes",
    default=DEFAULT_MIN_AGE_MINUTES,
    show_default=True,
    type=float,
    help="Файл моложе этого возраста считается ещё пишущимся и не трогается.",
)
@click.option(
    "--require-complete/--no-require-complete",
    default=True,
    show_default=True,
    help="Переносить выпуск только целиком: половинный потом не отличить от целого.",
)
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, type=int, help="Потоков на копирование между ФС.")
@click.option("--dry-run", is_flag=True, help="Только показать, что было бы сделано.")
@click.option("--db", "db_path", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pack-name", default=None, help="Имя пака в базе. Не задано — имя папки DEST_ROOT.")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(
    source_root: Path,
    dest_root: Path,
    subdir: str,
    move: bool,
    min_age_minutes: float,
    require_complete: bool,
    jobs: int,
    dry_run: bool,
    db_path: "Path | None",
    pack_name: "str | None",
    log_level: str,
) -> None:
    """Переносит ИСХОДНЫЙ_КОРЕНЬ/{год}/{выпуск}/sharpened/* в КОРЕНЬ_НАЗНАЧЕНИЯ/{год}/{выпуск}/."""
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s: %(message)s")
    report, moved_names = collect(
        source_root,
        dest_root,
        subdir=subdir,
        move=move,
        min_age_minutes=min_age_minutes,
        require_complete=require_complete,
        jobs=jobs,
        dry_run=dry_run,
    )
    if db_path is not None and not dry_run:
        fill_database(db_path, pack_name or dest_root.name, dest_root, moved_names, report)
    click.echo(report.summary())


if __name__ == "__main__":
    main()
