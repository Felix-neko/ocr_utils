"""Нарезка собранного выпуска внешнего OCR на статьи по маркерам ``<!-- article An -->`` и ``<!-- rubric Rn -->``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Маркеры, которые ставит сборка выпуска (external_ocr_services.reconcile): одна статья — один маркер.
MARKER_RE = re.compile(r"^<!-- (rubric R\d+|article A\d+) -->\s*$", re.MULTILINE)
RUBRIC_RE = re.compile(r"<rubric>(.*?)</rubric>", re.DOTALL)
AUTHOR_RE = re.compile(r"<author>(.*?)</author>", re.DOTALL)
TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
# Имя файла выпуска: ``1968_09.md``.
ISSUE_FILE_RE = re.compile(r"^(\d{4})_(\d{2})\.md$")


def _plain(fragment: str) -> str:
    """Строка без markdown-выделения и лишних пробелов: ``*ПРОБЛЕМЫ, СУЖДЕНИЯ*`` → ``ПРОБЛЕМЫ, СУЖДЕНИЯ``.

    Args:
        fragment: Кусок разметки (содержимое ``<rubric>`` или ``<author>``).

    Returns:
        Текст без ``*`` и ``_`` по краям слов, с одиночными пробелами.
    """
    text = re.sub(r"[*_]+", "", fragment)
    return re.sub(r"\s+", " ", text).strip(" ,")


@dataclass(frozen=True)
class Article:
    """Статья выпуска: где лежит, в какой рубрике, заголовок, авторы и тело.

    Args:
        year: Год выпуска (``"1968"``).
        issue: Номер выпуска двумя знаками (``"09"``).
        article_id: Id статьи из маркера (``"A15"``).
        rubric: Рубрика, под которой стоит статья; ``None`` — до первой рубрики выпуска.
        title: Заголовок ``#`` статьи; пусто, если его нет.
        authors: Авторы из ``<author>`` в порядке появления.
        text: Тело статьи от строки после маркера до следующего маркера (разметка как есть).
        source_path: Путь к md-файлу выпуска.
        line_start: Номер строки (с 1) первой строки ``text`` в файле — для ссылок ``файл:строка``.
    """

    year: str
    issue: str
    article_id: str
    rubric: str | None
    title: str
    authors: tuple[str, ...]
    text: str
    source_path: Path
    line_start: int

    @property
    def key(self) -> str:
        """Короткий ключ статьи для логов и CLI: ``1968_09:A15``."""
        return f"{self.year}_{self.issue}:{self.article_id}"

    @property
    def rel(self) -> Path:
        """Путь записи статьи в кэше: ``1968/09/A15``."""
        return Path(self.year) / self.issue / self.article_id

    @property
    def n_chars(self) -> int:
        """Длина тела в символах."""
        return len(self.text)


def split_issue(path: Path) -> list[Article]:
    """Разрезать собранный выпуск на статьи.

    Текст между маркером рубрики и следующим маркером — сама рубрика (её название запоминается для
    следующих статей), текст между маркером статьи и следующим маркером — статья. Всё до первого
    маркера (обложка, оглавление, редколлегия) пропускается.

    Args:
        path: Файл ``{год}_{выпуск}.md``.

    Returns:
        Статьи в порядке следования в выпуске.
    """
    match = ISSUE_FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"имя файла выпуска не в виде ГГГГ_ВВ.md: {path}")
    year, issue = match.groups()
    text = path.read_text(encoding="utf-8")
    markers = list(MARKER_RE.finditer(text))
    articles: list[Article] = []
    rubric: str | None = None
    for index, marker in enumerate(markers):
        # Тело — от конца строки маркера до начала следующего маркера (или конца файла).
        body_start = marker.end() + 1
        body_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        body = text[body_start:body_end]
        label = marker.group(1)
        if label.startswith("rubric"):
            found = RUBRIC_RE.search(body)
            rubric = _plain(found.group(1)) if found else None
            continue
        title_match = TITLE_RE.search(body)
        articles.append(
            Article(
                year=year,
                issue=issue,
                article_id=label.split()[1],
                rubric=rubric,
                title=_plain(title_match.group(1)) if title_match else "",
                authors=tuple(_plain(found) for found in AUTHOR_RE.findall(body)),
                text=body.strip("\n"),
                source_path=path,
                # Строки считаются с 1; тело начинается со строки, следующей за маркером,
                # а ведущие пустые строки срезаны strip — их тоже надо отсчитать.
                line_start=text.count("\n", 0, body_start) + 1 + (len(body) - len(body.lstrip("\n"))),
            )
        )
    return articles


def issue_files(root: Path, issues: list[str] | None = None) -> list[Path]:
    """Файлы выпусков пака: все или выбранные.

    Args:
        root: Папка пака с подпапками по годам (``пак-1/1968/1968_09.md``).
        issues: Выпуски вида ``1968_09``; ``None`` — все файлы пака.

    Returns:
        Пути в порядке год → выпуск; отсутствующий выпуск из списка — ошибка.
    """
    if issues is None:
        return sorted(path for path in root.glob("*/*.md") if ISSUE_FILE_RE.match(path.name))
    paths = []
    for name in issues:
        path = root / name[:4] / f"{name}.md"
        if not path.is_file():
            raise FileNotFoundError(f"нет выпуска {name}: {path}")
        paths.append(path)
    return paths


def iter_articles(root: Path, issues: list[str] | None = None) -> Iterator[Article]:
    """Статьи пака (или выбранных выпусков) подряд.

    Args:
        root: Папка пака.
        issues: Выпуски вида ``1968_09``; ``None`` — весь пак.

    Yields:
        Статьи в порядке год → выпуск → номер в выпуске.
    """
    for path in issue_files(root, issues):
        yield from split_issue(path)


def parse_article_key(key: str) -> tuple[str, str]:
    """``1968_09:A15`` → (``1968_09``, ``A15``).

    Args:
        key: Ключ статьи в виде ``ГГГГ_ВВ:An``.

    Returns:
        Выпуск и id статьи; неверный вид — ``ValueError``.
    """
    match = re.fullmatch(r"(\d{4}_\d{2}):(A\d+)", key)
    if match is None:
        raise ValueError(f"ключ статьи не в виде ГГГГ_ВВ:An: {key}")
    return match.group(1), match.group(2)
