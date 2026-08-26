"""Обход папки пака: пак -> годовой комплект -> выпуск -> полоса.

Правила продиктованы тем, как реально разложены сканы МТС::

    пак-1/1974/01/IMG_0004_1L.tif        полоса
    пак-1/1974/01/74_01.ScanTailor       проект ScanTailor, не полоса
    пак-1/1974/01/cache/thumbs/*.png     миниатюры ScanTailor, не полосы
    пак-1/1975/05 (2)/...                перескан того же выпуска, отдельный выпуск
    пак-1/1966/03/Thumbs.db              мусор Windows
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ocr_utils.scan_cropping.image_io import IMAGE_EXTS

logger = logging.getLogger(__name__)

# Подкаталоги выпуска, которые игнорируются всегда. Нерекурсивного обхода для этого уже
# достаточно, но правило записано ОТДЕЛЬНО и проверяется тестом: ScanTailor кладёт в
# выпуск ``cache/thumbs/*.png`` (252 файла на пак-1), и стоит однажды включить рекурсию
# или сменить способ обхода — миниатюры молча уедут в базу как полосы.
IGNORED_DIRS = frozenset({"cache", ".cache"})

# Имя годового комплекта начинается с четырёх цифр: так отсеиваются служебные папки
# рядом с годами, если они появятся.
YEAR_RE = re.compile(r"^(\d{4})")

# Номер выпуска — ведущие цифры имени: "05" -> 5, "05 (2)" -> 5, "" -> None.
ISSUE_NUMBER_RE = re.compile(r"^(\d+)")


@dataclass
class ScannedPage:
    """Полоса: путь к файлу и его место в выпуске."""

    path: Path
    file_name: str
    rel_path: str  # относительно корня пака, через "/"
    order_index: int


@dataclass
class ScannedIssue:
    """Выпуск: папка с полосами."""

    name: str
    number: int | None
    rel_path: str
    pages: list[ScannedPage] = field(default_factory=list)


@dataclass
class ScannedYear:
    """Годовой комплект: папка с выпусками."""

    name: str
    year: int | None
    rel_path: str
    issues: list[ScannedIssue] = field(default_factory=list)


def is_ignored_dir(path: Path) -> bool:
    """Служебный ли это подкаталог выпуска (``cache``, ``.cache``)."""
    return path.name.lower() in IGNORED_DIRS


def issue_images(issue_dir: Path, recursive: bool = False) -> list[Path]:
    """Файлы-полосы выпуска в устойчивом порядке.

    По умолчанию — только непосредственное содержимое папки. ``recursive=True`` оставлен
    на случай пака с полосами в подпапках; служебные каталоги из :data:`IGNORED_DIRS`
    отсекаются в обоих режимах, поэтому правило не зависит от способа обхода.
    """
    if recursive:
        candidates = (
            p
            for p in issue_dir.rglob("*")
            if not any(is_ignored_dir(parent) for parent in p.relative_to(issue_dir).parents) and not is_ignored_dir(p)
        )
    else:
        candidates = issue_dir.iterdir()

    return sorted(
        p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )


def scan_pack(root: Path, recursive: bool = False) -> list[ScannedYear]:
    """Дерево пака: годы -> выпуски -> полосы. Пустые выпуски и годы отбрасываются."""
    root = Path(root)
    years: list[ScannedYear] = []

    for year_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        match = YEAR_RE.match(year_dir.name)
        if not match:
            logger.debug("Пропускаю папку без года в имени: %s", year_dir.name)
            continue

        year = ScannedYear(name=year_dir.name, year=int(match.group(1)), rel_path=year_dir.name)
        for issue_dir in sorted(p for p in year_dir.iterdir() if p.is_dir() and not is_ignored_dir(p)):
            images = issue_images(issue_dir, recursive)
            if not images:
                logger.debug("Выпуск без картинок, пропускаю: %s", issue_dir)
                continue

            number_match = ISSUE_NUMBER_RE.match(issue_dir.name)
            issue = ScannedIssue(
                name=issue_dir.name,
                number=int(number_match.group(1)) if number_match else None,
                rel_path=f"{year.rel_path}/{issue_dir.name}",
            )
            issue.pages = [
                ScannedPage(
                    path=path, file_name=path.name, rel_path=path.relative_to(root).as_posix(), order_index=index
                )
                for index, path in enumerate(images)
            ]
            year.issues.append(issue)

        if year.issues:
            years.append(year)

    return years


def count_pages(years: list[ScannedYear]) -> int:
    """Сколько всего полос в дереве — для сообщений и прогресс-баров."""
    return sum(len(issue.pages) for year in years for issue in year.issues)
